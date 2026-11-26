#!/usr/bin/env python3
"""按四档场景生成压测请求（JSONL），每个场景、每级并发各一个文件。

为什么要自己造：评审组的「性能压测数据集」还没下发。下发后用 --official 换成官方数据，
压测脚本其余部分不变（见 README）。

怎么模拟缓存命中（智能体 0.70、编程 0.9）：
  - 每级并发 C 造 C 个会话（session，可以理解成 C 个同时在干活的智能体或 C 个代码仓库）。
  - 同一会话里的请求共享开头「命中率 × 本请求长度」个 token，后面是各自独有的内容。
  - 正式压测前先发一轮预热请求，把每个会话的共享前缀灌进前缀缓存（见 run_ladder.py）。
    所以理论命中率 ≈ 设定值；实际命中率由压测时 /metrics 的增量算出，两者对照着看。
  - 不同并发级别用不同的会话，上一级留在缓存里的内容不会帮下一级作弊。

共同 system prompt（--shared-system-tokens N，默认 0 = 不加）：
  真实智能体/编程请求开头往往是一段所有会话都一样的长 system prompt（工具说明、规范）。
  加上它才能测到多副本路由器的关键路径：旧版 router 正是被这种请求打成 100%/0%/0%/0%
  （issue #14）。--shared-system-role system 放成单独的 system 消息（默认）；
  user 放在 user 消息开头（模拟数据集把共同前缀写进 user 消息的情况）。
  注意：这段内容本身也会命中缓存，所以加了之后理论命中率会高于场景目标值（meta 里有数）。

文本来源：本机 Python 标准库的源码和注释（自然的代码 + 英文），不是随机乱码。
规则明确禁止长文本档用乱码测速度（乱码会让投机解码的接受率失真）。

长度控制：给 --tokenizer（模型目录或 tokenizer.json）时按真实 token 切；
不给时按「字符数 / --chars-per-token」估算，并在 meta 里标明「估算」。

用法：
  python3 loadtest/make_datasets.py --out loadtest/data                       # 全量、估算长度
  python3 loadtest/make_datasets.py --out loadtest/data --tokenizer /models/Qwen3.8-27B-FP8
  python3 loadtest/make_datasets.py --out /tmp/lt --scale 0.02 --scenarios agent,stress  # 缩小版调试
"""
import argparse
import json
import math
import os
import random
import sys
import sysconfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenarios import ORDER, SCENARIOS  # noqa: E402


# ---------------------------------------------------------------- 语料与切词

def load_corpus(max_chars):
    """读本机 Python 标准库 .py 文件，按文件名排序拼起来，直到 max_chars。"""
    root = sysconfig.get_paths()["stdlib"]
    files = []
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d not in ("test", "tests", "idlelib", "site-packages", "__pycache__"))
        files.extend(os.path.join(dp, f) for f in sorted(fns) if f.endswith(".py"))
    buf, n = [], 0
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                t = fh.read()
        except (UnicodeDecodeError, OSError):
            continue
        buf.append(f"# ===== file: {os.path.relpath(f, root)} =====\n{t}\n")
        n += len(buf[-1])
        if n >= max_chars:
            break
    return "".join(buf)


class CharTok:
    """没有 tokenizer 时的估算：一个 token ≈ chars_per_token 个字符。"""
    exact = False

    def __init__(self, corpus, cpt):
        self.corpus, self.cpt = corpus, cpt
        self.n = int(len(corpus) / cpt)

    def slice(self, start, length):
        a = int(start * self.cpt)
        return self.corpus[a:a + int(length * self.cpt)]

    def count(self, text):
        return int(len(text) / self.cpt)


class RealTok:
    """用模型自己的 tokenizer（tokenizers 库）切词，长度准确。"""
    exact = True

    def __init__(self, corpus, path):
        from tokenizers import Tokenizer
        if os.path.isdir(path):
            path = os.path.join(path, "tokenizer.json")
        self.tk = Tokenizer.from_file(path)
        # 分段编码，避免一次编码几十 MB 太慢
        ids = []
        step = 200_000
        for i in range(0, len(corpus), step):
            ids.extend(self.tk.encode(corpus[i:i + step], add_special_tokens=False).ids)
        self.ids = ids
        self.n = len(ids)

    def slice(self, start, length):
        return self.tk.decode(self.ids[start:start + length])

    def count(self, text):
        return len(self.tk.encode(text, add_special_tokens=False).ids)


def take(tok, rng, length):
    """从语料里随机取一段长 length 的文本（语料不够长时首尾拼接）。"""
    out, need = [], length
    while need > 0:
        L = min(need, tok.n - 1)
        start = rng.randrange(0, max(1, tok.n - L))
        out.append(tok.slice(start, L))
        need -= L
    return "".join(out)


# ---------------------------------------------------------------- 长度分布

def sample_lengths(rng, n, mean, lo, hi, mixed):
    """混合长度：对数正态（σ=0.8）截断到 [lo, hi]，再缩放使均值≈mean。定长则全等于 mean。"""
    if not mixed:
        return [mean] * n
    raw = [rng.lognormvariate(0, 0.8) for _ in range(n)]
    scale = mean / (sum(raw) / n)
    for _ in range(20):  # 截断会拉偏均值，迭代修正
        xs = [min(hi, max(lo, r * scale)) for r in raw]
        m = sum(xs) / n
        if abs(m - mean) / mean < 0.005:
            break
        scale *= mean / m
    return [int(x) for x in xs]


# ---------------------------------------------------------------- 生成

LONGGEN_ASKS = [
    "Write a detailed, well-structured technical tutorial that explains the code above to a new team member. "
    "Cover its purpose, every public function and class, edge cases, and give several worked examples.",
    "Write a long design review of the code above: strengths, weaknesses, possible bugs, performance "
    "concerns and a step-by-step refactoring plan with example code for each step.",
    "Write comprehensive documentation for the code above in the style of an official library reference, "
    "with an overview, usage guide, API reference for each item, and a FAQ section.",
    "Write a long blog post that walks through the code above line by line, explains the history and "
    "reasoning behind its design, and compares it with alternative approaches in other languages.",
]


def build_level(tok, rng, name, sc, conc, rounds, min_requests, scale, shared_sys="", sys_role="system"):
    """生成某场景某级并发的 (正式请求, 预热请求, 统计)。"""
    n_req = max(min_requests, rounds * conc)
    n_sess = conc if sc["cache_hit"] > 0 else n_req
    mean_in = max(64, int(sc["input_tokens"] * scale))
    hi = max(mean_in, int(sc["max_input_tokens"] * scale))
    lo = max(32, min(mean_in, int(2000 * scale)))
    out_tokens = max(8, int(sc["output_tokens"] * scale))
    lens = sample_lengths(rng, n_req, mean_in, lo, hi, sc["mixed_length"])
    hit = sc["cache_hit"]

    # 请求按会话轮转排列：第 j 轮依次是会话 0..S-1 的第 j 个请求，
    # 这样同时在途的请求大多来自不同会话，同一会话的下一个请求要隔一整轮才来。
    sessions = [[] for _ in range(n_sess)]
    for i, L in enumerate(lens):
        sessions[i % n_sess].append(L)

    reqs, warm = [], []
    tag = f"{name}-c{conc}"
    for s, sl in enumerate(sessions):
        if not sl:
            continue
        sid = f"{tag}-s{s:03d}"
        shared_max = int(max(sl) * hit)
        header = f"### Session {sid}: repository snapshot and prior context\n"
        shared_text = header + take(tok, rng, shared_max) if shared_max > 0 else ""
        if shared_text:
            warm.append({"session_id": sid, "text": shared_text, "output_tokens": 1})
        for j, L in enumerate(sl):
            shared_n = int(L * hit)
            if shared_n > 0:
                # 取共享前缀的前 shared_n 个 token（按比例截字符，估算即可，前缀一致就能命中）
                cut = header + (shared_text[len(header):][:int(len(shared_text[len(header):]) * shared_n / max(1, shared_max))])
            else:
                cut = ""
            uniq_head = f"\n### Request {sid}-r{j:03d}\n"
            if name == "longgen":
                body = take(tok, rng, max(16, L - 80)) + "\n\n" + rng.choice(LONGGEN_ASKS) + \
                    f" The answer should be at least {out_tokens * 3 // 4} words long."
                text = uniq_head + body
            else:
                body = take(tok, rng, max(16, L - shared_n))
                text = cut + uniq_head + body + \
                    "\n\nBased on the context above, continue the task: explain what this code does and propose improvements."
            reqs.append({"session_id": sid, "text": text, "output_tokens": out_tokens,
                         "shared_chars": len(cut)})

    # 共同 system prompt：system 角色单独成一条消息；user 角色拼到 user 消息最前面
    sys_tokens = tok.count(shared_sys) if shared_sys else 0
    for r in reqs + warm:
        if not shared_sys:
            break
        if sys_role == "system":
            r["system"] = shared_sys
        else:
            r["text"] = shared_sys + "\n" + r["text"]
            if "shared_chars" in r:
                r["shared_chars"] += len(shared_sys) + 1
    for w in warm:
        w["input_tokens"] = tok.count(w["text"]) + (sys_tokens if "system" in w else 0)

    # 重排成会话轮转顺序
    by_sess = {}
    for r in reqs:
        by_sess.setdefault(r["session_id"], []).append(r)
    ordered = []
    for j in range(max(len(v) for v in by_sess.values())):
        for v in by_sess.values():
            if j < len(v):
                ordered.append(v[j])

    text_counts = [tok.count(r["text"]) for r in ordered]
    extra_sys = sys_tokens if (shared_sys and sys_role == "system") else 0
    in_counts = [c + extra_sys for c in text_counts]
    for r, c in zip(ordered, in_counts):
        r["input_tokens"] = c
    shared_tokens = sum(int(c * r["shared_chars"] / max(1, len(r["text"]))) + extra_sys
                        for r, c in zip(ordered, text_counts))
    stats = {
        "scenario": name, "concurrency": conc, "requests": len(ordered), "sessions": len(by_sess),
        "warmup_requests": len(warm),
        "input_tokens_mean": round(sum(in_counts) / len(in_counts)),
        "input_tokens_max": max(in_counts), "input_tokens_min": min(in_counts),
        "output_tokens": out_tokens,
        "target_cache_hit": hit,
        "theoretical_cache_hit": round(shared_tokens / max(1, sum(in_counts)), 3),
        "shared_system_tokens": sys_tokens,
        "shared_system_role": sys_role if shared_sys else None,
    }
    return ordered, warm, stats


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="输出目录")
    p.add_argument("--scenarios", default=",".join(ORDER), help="逗号分隔：" + ",".join(ORDER))
    p.add_argument("--concurrency", default="", help="只生成这些并发级别（逗号分隔），默认按场景阶梯")
    p.add_argument("--rounds", type=int, default=3, help="每级请求数 = max(--min-requests, rounds × 并发)")
    p.add_argument("--min-requests", type=int, default=24)
    p.add_argument("--scale", type=float, default=1.0, help="输入/输出长度整体缩放（<1 只用于本地调试）")
    p.add_argument("--tokenizer", default="", help="模型目录或 tokenizer.json；不给则按字符估算")
    p.add_argument("--chars-per-token", type=float, default=3.6, help="估算模式下每 token 字符数")
    p.add_argument("--seed", type=int, default=20261127)
    p.add_argument("--shared-system-tokens", type=int, default=0,
                   help="所有请求开头加同一段共同 system prompt 的 token 数（按 --scale 缩放；0 = 不加）")
    p.add_argument("--shared-system-role", choices=["system", "user"], default="system",
                   help="共同前缀放成单独的 system 消息（默认），还是拼在 user 消息开头")
    a = p.parse_args()

    rng = random.Random(a.seed)
    names = [s.strip() for s in a.scenarios.split(",") if s.strip()]
    for n in names:
        if n not in SCENARIOS:
            sys.exit(f"未知场景 {n}，可选：{ORDER}")
    print("读取语料（本机 Python 标准库源码）...", flush=True)
    corpus = load_corpus(max_chars=40_000_000 if a.scale >= 0.5 else 4_000_000)
    if a.tokenizer:
        print(f"用 tokenizer 切词：{a.tokenizer}", flush=True)
        tok = RealTok(corpus, a.tokenizer)
    else:
        print(f"[提示] 没给 --tokenizer，按 {a.chars_per_token} 字符/token 估算长度", flush=True)
        tok = CharTok(corpus, a.chars_per_token)
    print(f"语料约 {tok.n:,} token", flush=True)

    shared_sys = ""
    if a.shared_system_tokens > 0:
        n_sys = max(16, int(a.shared_system_tokens * a.scale))
        # 单独的随机源：加不加这个选项，其余数据都不变
        shared_sys = "### Shared agent instructions (identical for every session)\n" + \
            take(tok, random.Random(a.seed + 1), n_sys)
        print(f"共同 system prompt：约 {tok.count(shared_sys):,} token，放在 {a.shared_system_role} 消息", flush=True)

    only = {int(x) for x in a.concurrency.split(",") if x.strip()}
    all_stats = []
    for name in names:
        sc = SCENARIOS[name]
        d = os.path.join(a.out, name)
        os.makedirs(d, exist_ok=True)
        for conc in sc["concurrency"]:
            if only and conc not in only:
                continue
            reqs, warm, st = build_level(tok, rng, name, sc, conc, a.rounds, a.min_requests, a.scale,
                                         shared_sys, a.shared_system_role)
            with open(os.path.join(d, f"c{conc}.jsonl"), "w") as f:
                for r in reqs:
                    f.write(json.dumps({k: r[k] for k in ("session_id", "system", "text", "input_tokens", "output_tokens")
                                        if k in r}, ensure_ascii=False) + "\n")
            with open(os.path.join(d, f"c{conc}.warmup.jsonl"), "w") as f:
                for r in warm:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            st["length_mode"] = "tokenizer" if tok.exact else f"估算（{a.chars_per_token} 字符/token）"
            all_stats.append(st)
            print(f"  {name:8s} c{conc:<3d} 请求 {st['requests']:4d}  会话 {st['sessions']:4d}  "
                  f"输入均值 {st['input_tokens_mean']:>7,}  最大 {st['input_tokens_max']:>7,}  "
                  f"输出 {st['output_tokens']:>5,}  理论命中 {st['theoretical_cache_hit']:.2f}", flush=True)
    meta = {"seed": a.seed, "scale": a.scale, "tokenizer": a.tokenizer or None,
            "shared_system_tokens": a.shared_system_tokens, "shared_system_role": a.shared_system_role,
            "levels": all_stats}
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"完成：{a.out}/meta.json", flush=True)


if __name__ == "__main__":
    main()
