#!/usr/bin/env python3
"""按四档场景 × 并发阶梯压测一个（或几个）推理服务，判定 SLO，找出每档的「摸高」并发和 Goodput。

每个场景、每级并发依次做：
  1. 预热：把该级每个会话的共享前缀发一遍（max_tokens=1，不计成绩），让前缀缓存就位
  2. 抓 /metrics（压测前）
  3. 正式压测（引擎二选一：aiperf = 评审组用的 NVIDIA AIPerf；builtin = 内置客户端）
  4. 抓 /metrics（压测后），两次相减得到「实际缓存命中率」
  5. 按 P90 TTFT / P90 TPOT / 错误率判定过线；默认某级不过线就停止该场景的后续级别

结果写到 --out 目录：每级一个 JSON，最后生成 report.md（见 report.py）。

用法：
  # 对本机 8000 端口的服务跑全部场景（数据先用 make_datasets.py 生成）
  python3 loadtest/run_ladder.py --url http://127.0.0.1:8000 --data loadtest/data --out loadtest/results/qwen-dp8

  # 只跑某个场景的某几级，并发阶梯不过线也继续往上测
  python3 loadtest/run_ladder.py --url ... --data ... --out ... --scenarios coding --concurrency 16,32 --no-stop-on-fail

  # 多个副本（各自独立端口、前面没有负载均衡）：同一会话固定发到同一个地址，保住缓存命中
  python3 loadtest/run_ladder.py --url http://h1:8000,http://h2:8000 --engine builtin ...
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import builtin_client  # noqa: E402
from common import cache_hit_between, judge, scrape_metrics, served_model  # noqa: E402
from scenarios import ORDER, SCENARIOS  # noqa: E402


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def pick_url(urls, session_id):
    """同一会话固定到同一个地址（多副本时保住前缀缓存命中）。"""
    if len(urls) == 1:
        return urls[0]
    h = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
    return urls[h % len(urls)]


def scrape_all(urls):
    snaps = [scrape_metrics(u) for u in urls]
    if any(s is None for s in snaps):
        return None
    total = {}
    for s in snaps:
        for k, v in s.items():
            total[k] = total.get(k, 0.0) + v
    return total


def run_builtin_multi(urls, model, records, conc, extra, timeout):
    """多地址时按会话分组，每个地址分到的并发与其请求数成正比；单地址就是普通跑法。"""
    if len(urls) == 1:
        return builtin_client.run(urls[0], model, records, conc, extra, timeout)
    import threading
    groups = {}
    for r in records:
        groups.setdefault(pick_url(urls, r["session_id"]), []).append(r)
    out, walls = [], []
    lock = threading.Lock()

    def go(u, recs, c):
        res, wall = builtin_client.run(u, model, recs, c, extra, timeout)
        with lock:
            out.extend(res)
            walls.append(wall)

    ths = []
    for u, recs in groups.items():
        c = max(1, round(conc * len(recs) / len(records)))
        ths.append(threading.Thread(target=go, args=(u, recs, c)))
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return out, max(walls)


def warmup(urls, model, warm, conc, timeout):
    if not warm:
        return 0.0, 0
    recs = [dict(w, output_tokens=1) for w in warm]
    res, wall = run_builtin_multi(urls, model, recs, conc, {}, timeout)
    errs = sum(1 for r in res if r and r["error"])
    return wall, errs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True, help="服务地址，如 http://127.0.0.1:8000；多个用逗号分隔")
    p.add_argument("--data", required=True, help="make_datasets.py 的输出目录")
    p.add_argument("--out", required=True, help="结果目录")
    p.add_argument("--model", default="", help="请求里的 model 名，默认取 /v1/models 第一个")
    p.add_argument("--scenarios", default=",".join(ORDER))
    p.add_argument("--concurrency", default="", help="只跑这些并发级别（逗号分隔）")
    p.add_argument("--engine", choices=["aiperf", "builtin"], default="aiperf")
    p.add_argument("--aiperf", default="aiperf", help="aiperf 可执行文件路径")
    p.add_argument("--tokenizer", default="", help="给 aiperf 用的 tokenizer（模型目录或 HF 名）；默认用 --model")
    p.add_argument("--no-stop-on-fail", action="store_true", help="某级不过线也继续往上测")
    p.add_argument("--no-warmup", action="store_true", help="不做前缀预热（看冷缓存下的表现）")
    p.add_argument("--no-ignore-eos", action="store_true",
                   help="不加 ignore_eos（默认加，保证输出长度等于场景设定）")
    p.add_argument("--max-error-rate", type=float, default=0.0, help="允许的错误率，默认 0")
    p.add_argument("--timeout", type=int, default=1800, help="单请求超时秒数")
    p.add_argument("--label", default="", help="写进报告的说明，如「Qwen FP8 DP=8 单机」")
    p.add_argument("--force", action="store_true", help="--out 目录已有结果时先清空")
    a = p.parse_args()

    urls = [u.strip().rstrip("/") for u in a.url.split(",") if u.strip()]
    model = a.model or served_model(urls[0])
    names = [s.strip() for s in a.scenarios.split(",") if s.strip()]
    only = {int(x) for x in a.concurrency.split(",") if x.strip()}
    extra = {} if a.no_ignore_eos else {"ignore_eos": True}
    if os.path.exists(os.path.join(a.out, "run_meta.json")):
        if not a.force:
            sys.exit(f"[错误] {a.out} 里已有上一次的结果，混在一起报告会出错。换一个 --out 目录，或加 --force 清空它。")
        import shutil
        shutil.rmtree(a.out)
    os.makedirs(a.out, exist_ok=True)

    run_meta = {
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "urls": urls, "model": model, "engine": a.engine, "label": a.label,
        "warmup": not a.no_warmup, "ignore_eos": not a.no_ignore_eos,
        "max_error_rate": a.max_error_rate, "stop_on_fail": not a.no_stop_on_fail,
        "data_meta": json.load(open(os.path.join(a.data, "meta.json"))) if os.path.exists(
            os.path.join(a.data, "meta.json")) else None,
    }
    with open(os.path.join(a.out, "run_meta.json"), "w") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)
    print(f"服务：{', '.join(urls)}  模型：{model}  引擎：{a.engine}", flush=True)

    if a.engine == "aiperf":
        import aiperf_engine

    for name in names:
        sc = SCENARIOS[name]
        for conc in sc["concurrency"]:
            if only and conc not in only:
                continue
            records = load_jsonl(os.path.join(a.data, name, f"c{conc}.jsonl"))
            if not records:
                print(f"[跳过] {name} c{conc}：没有数据文件", flush=True)
                continue
            warm = [] if a.no_warmup else load_jsonl(os.path.join(a.data, name, f"c{conc}.warmup.jsonl"))
            # 每次运行给所有请求（含预热）开头加同一个随机标记：本轮内的共享前缀照样命中，
            # 但同一份数据重跑时不会白捡上一轮留在缓存里的内容（否则命中率和 TTFT 都会虚高）。
            salt = f"[run {uuid.uuid4().hex[:12]}]\n"
            # 有共同 system prompt 时标记加在 system 最前面（它才是整个请求的开头）
            salted = lambda r: dict(r, system=salt + r["system"]) if r.get("system") else dict(r, text=salt + r["text"])  # noqa: E731
            records = [salted(r) for r in records]
            warm = [salted(w) for w in warm]
            print(f"\n== {sc['title']}（{name}）并发 {conc}：{len(records)} 个请求 ==", flush=True)
            w_s, w_err = warmup(urls, model, warm, conc, a.timeout)
            if warm:
                print(f"   预热 {len(warm)} 个会话前缀，{w_s:.1f}s，失败 {w_err}", flush=True)

            before = scrape_all(urls)
            if a.engine == "builtin":
                res, wall = run_builtin_multi(urls, model, records, conc, extra, a.timeout)
                level = builtin_client.summarize(res, wall, sc)
                raw_path = os.path.join(a.out, name, f"c{conc}.requests.jsonl")
                os.makedirs(os.path.dirname(raw_path), exist_ok=True)
                with open(raw_path, "w") as f:
                    for r in res:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            else:
                if len(urls) > 1:
                    print("   [提示] aiperf 引擎只压第一个地址；多副本请在前面加负载均衡，或改用 --engine builtin",
                          flush=True)
                level = aiperf_engine.run(
                    aiperf=a.aiperf, url=urls[0], model=model, tokenizer=a.tokenizer or model,
                    records=records, conc=conc, sc=sc, extra=extra,
                    art_dir=os.path.join(a.out, name, f"c{conc}.aiperf"), timeout=a.timeout)
            after = scrape_all(urls)

            level.update({
                "scenario": name, "concurrency": conc,
                "warmup_s": w_s, "warmup_errors": w_err,
                "measured_cache_hit": cache_hit_between(before, after),
                "target_cache_hit": sc["cache_hit"],
                "slo": {"ttft_s": sc["ttft_s"], "tpot_ms": sc["tpot_ms"]},
            })
            ok, reasons = judge(level, sc, a.max_error_rate)
            level["pass"], level["fail_reasons"] = ok, reasons
            os.makedirs(os.path.join(a.out, name), exist_ok=True)
            with open(os.path.join(a.out, name, f"c{conc}.json"), "w") as f:
                json.dump(level, f, ensure_ascii=False, indent=2)

            hit = level["measured_cache_hit"]
            fmt = lambda v, s: "—" if v is None else s.format(v)  # noqa: E731
            print(f"   P90 TTFT {fmt(level['p90_ttft_s'], '{:.2f}s')}  P90 TPOT {fmt(level['p90_tpot_ms'], '{:.1f}ms')}  "
                  f"错误 {level['errors']}/{level['requests']}  Goodput {fmt(level['goodput_rps'], '{:.3f} req/s')}  "
                  f"实测命中 {fmt(hit, '{:.2f}')}  → {'✅ 过线' if ok else '❌ ' + '；'.join(reasons)}", flush=True)
            if not ok and not a.no_stop_on_fail:
                print(f"   {name} 在并发 {conc} 不过线，停止该场景后续级别（--no-stop-on-fail 可继续）", flush=True)
                break

    import report
    path = report.write(a.out)
    print(f"\n报告：{path}", flush=True)


if __name__ == "__main__":
    main()
