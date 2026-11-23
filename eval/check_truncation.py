#!/usr/bin/env python3
"""check_truncation.py -- 统计 EvalScope 评测输出的截断率、输出长度分布与精度对照

只用 Python 标准库，无第三方依赖。同时兼容：
- 未开 --reasoning-parser（<think>...</think> 内联在 content 里）
- 开启 --reasoning-parser qwen3（reasoning 独立字段，截断主要看 stop_reason）

判定口径（#5 评审校准后）：
- 每题三态：截断（stop_reason 是 max_tokens/length）、完成（有 stop_reason
  且不是截断）、未知（拿不到 stop_reason，如缺 predictions 文件）。
  未知不再拿 </think> 缺失当截断证据；</think> 计数只作诊断参考。
- 逐题对照按稳定题目 ID（user 消息内容 + target 的 sha256）配对，
  不再按 index；对照时按数据集名逐集配对，不再只比第一集。

用法：
  # 扫描 eval/outputs/ 下所有 EvalScope 轮次
  python3 eval/check_truncation.py

  # 指定一个或多个轮次目录
  python3 eval/check_truncation.py eval/outputs/20261122_065141 eval/outputs/20261121_171331

  # 逐题对比两轮（例如 FP8 vs BF16 同一数据集），附精确 McNemar p 值
  python3 eval/check_truncation.py --compare eval/outputs/20261121_164127 eval/outputs/20261122_064348
"""

import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _stable_qid(row: dict) -> Optional[str]:
    """稳定题目 ID：user 角色消息内容 + target 的 sha256。
    - review 行里 messages[i].id 是每轮随机生成的，不能跨轮配对；
    - index 在同一数据集同一 limit 下碰巧稳定，但 limit 或数据集版本一变就错位；
    - user 消息内容就是题目本身：AIME/GPQA 只有一条 user 消息，
      LCB 的题目在第二条 user 消息（第一条是共享的 system 指令），
      所以拼全部 user 消息内容。
    """
    msgs = row.get("messages") or []
    user_texts = [
        m.get("content", "")
        for m in msgs
        if isinstance(m, dict) and m.get("role") == "user"
    ]
    if not user_texts:
        # 兜底：没有 role 标记时用全部非 assistant 消息。
        user_texts = [
            m.get("content", "")
            for m in msgs
            if isinstance(m, dict) and m.get("role") != "assistant"
        ]
    if not user_texts:
        return None
    key = "\n".join(user_texts) + "|" + json.dumps(row.get("target"), ensure_ascii=False)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _parse_task_config(run_dir: Path) -> Dict[str, str]:
    """轻量解析 configs/task_config.yaml 的关键字段（不依赖 PyYAML）。"""
    cfg_path = run_dir / "configs" / "task_config.yaml"
    info: Dict[str, str] = {}
    if not cfg_path.is_file():
        return info
    text = cfg_path.read_text(encoding="utf-8", errors="replace")
    for key in ("max_tokens", "temperature", "top_p", "top_k"):
        m = re.search(rf"^\s*{key}:\s*([^\s#]+)", text, re.MULTILINE)
        if m:
            info[key] = m.group(1)
    return info


def _is_passed(review_row: dict) -> bool:
    sr = review_row.get("sample_score", {}).get("score", {}).get("value", {})
    vals = sr.values() if isinstance(sr, dict) else [sr]
    return any(v == 1 or v == 1.0 or v is True for v in vals)


def _percentile(sorted_vals: List[int], p: float) -> int:
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, max(0, int(math.ceil(p * len(sorted_vals))) - 1))
    return sorted_vals[idx]


def _mcnemar_exact_p(b: int, c: int) -> float:
    """双侧精确 McNemar 检验（二项分布 p=0.5）。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def analyze_run(run_dir: Path) -> List[dict]:
    """分析单个 EvalScope 输出目录（eval/outputs/<timestamp>）。"""
    review_files = sorted(glob.glob(str(run_dir / "reviews" / "*" / "*.jsonl")))
    if not review_files:
        return []
    cfg = _parse_task_config(run_dir)
    results = []

    for rpath_str in review_files:
        rpath = Path(rpath_str)
        model_name = rpath.parent.name
        dataset_file = rpath.name
        ppath = run_dir / "predictions" / model_name / dataset_file

        # 先读 predictions 获取 stop_reason 与 output_tokens
        pred_by_idx: Dict[int, dict] = {}
        if ppath.is_file():
            with ppath.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    pred_by_idx[d.get("index")] = d

        total = 0
        passed_total = 0
        trunc_stop = 0
        trunc_pass = 0
        done_count = 0
        done_pass = 0
        unknown_count = 0
        unknown_pass = 0
        no_think_end = 0
        out_tokens: List[int] = []
        per_qid_pass: Dict[str, bool] = {}
        qid_index: Dict[str, Optional[int]] = {}

        with rpath.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                idx = row.get("index")
                total += 1
                ok = _is_passed(row)
                qid = _stable_qid(row)
                if qid is None and idx is not None:
                    qid = f"index:{idx}"
                if qid is not None:
                    per_qid_pass[qid] = ok
                    qid_index[qid] = idx
                if ok:
                    passed_total += 1

                pred = pred_by_idx.get(idx, {})
                mo = pred.get("model_output") or {}
                choices = mo.get("choices") or []
                ch0 = choices[0] if choices else {}
                stop_reason = ch0.get("stop_reason") or ch0.get("finish_reason")
                usage = mo.get("usage") or {}
                otok = usage.get("output_tokens") or usage.get("completion_tokens")
                if isinstance(otok, int):
                    out_tokens.append(otok)

                # 诊断计数（只做参考，不参与截断/完成判定）：整行序列化后
                # 有没有 </think>。
                raw_text = json.dumps(row, ensure_ascii=False)
                missing_think_end = "</think>" not in raw_text
                if missing_think_end:
                    no_think_end += 1

                # 三态判定：只有拿到 stop_reason 才能定截断/完成；
                # 拿不到（缺 predictions 文件、或 choices 为空）就是未知，
                # 不再拿 </think> 缺失当截断证据——之前会把无 predictions
                # 的轮次整批误判成截断。
                if stop_reason is not None:
                    state = "trunc" if stop_reason in ("max_tokens", "length") else "done"
                else:
                    state = "unknown"

                if state == "trunc":
                    trunc_stop += 1
                    if ok:
                        trunc_pass += 1
                elif state == "done":
                    done_count += 1
                    if ok:
                        done_pass += 1
                else:
                    unknown_count += 1
                    if ok:
                        unknown_pass += 1

        out_tokens.sort()
        results.append(
            {
                "run_dir": str(run_dir),
                "dataset": dataset_file.replace("_default.jsonl", "").replace(".jsonl", ""),
                "max_tokens": cfg.get("max_tokens", "?"),
                "temperature": cfg.get("temperature", "?"),
                "total": total,
                "passed": passed_total,
                "acc": (passed_total / total * 100.0) if total else 0.0,
                "trunc": trunc_stop,
                "trunc_rate": (trunc_stop / total * 100.0) if total else 0.0,
                "trunc_pass": trunc_pass,
                "done": done_count,
                "done_pass": done_pass,
                "done_acc": (done_pass / done_count * 100.0) if done_count else 0.0,
                "unknown": unknown_count,
                "unknown_pass": unknown_pass,
                "no_think_end": no_think_end,
                "avg_tok": (sum(out_tokens) / len(out_tokens)) if out_tokens else 0.0,
                "p50_tok": _percentile(out_tokens, 0.50),
                "p90_tok": _percentile(out_tokens, 0.90),
                "max_tok": out_tokens[-1] if out_tokens else 0,
                "per_qid_pass": per_qid_pass,
                "qid_index": qid_index,
            }
        )
    return results


def print_table(rows: List[dict]) -> None:
    if not rows:
        print("未找到任何 EvalScope reviews/*.jsonl 文件。")
        return
    header = (
        f"{'目录':<28} {'数据集':<16} {'maxT':>5} {'temp':>4} "
        f"{'总分':>11} {'截断(占比)':>13} {'截断对':>6} {'未知':>5} {'完成正确率':>13} "
        f"{'out_tok(avg/p90/max)':>21}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        run_short = Path(r["run_dir"]).name
        score_s = f"{r['passed']}/{r['total']}({r['acc']:.1f}%)"
        trunc_s = f"{r['trunc']}/{r['total']}({r['trunc_rate']:.1f}%)"
        done_s = f"{r['done_pass']}/{r['done']}({r['done_acc']:.1f}%)"
        tok_s = f"{r['avg_tok']:.0f}/{r['p90_tok']}/{r['max_tok']}"
        print(
            f"{run_short:<28} {r['dataset']:<16} {str(r['max_tokens']):>5} {str(r['temperature']):>4} "
            f"{score_s:>11} {trunc_s:>13} {r['trunc_pass']:>6} {r['unknown']:>5} {done_s:>13} {tok_s:>21}"
        )


def _disp(qid: str, qid_index: Dict[str, Optional[int]]) -> str:
    """对照输出里的题目显示：有 index 就显示 index（和旧输出一致），
    没有才显示稳定 ID 前 8 位。"""
    idx = qid_index.get(qid)
    return str(idx) if idx is not None else qid[:8]


def _compare_pair(a: dict, b: dict) -> None:
    print_table([a, b])
    print()
    pa, pb = a["per_qid_pass"], b["per_qid_pass"]
    common = sorted(set(pa) & set(pb))
    only_a = sorted(set(pa) - set(pb))
    only_b = sorted(set(pb) - set(pa))
    both_ok = sum(1 for i in common if pa[i] and pb[i])
    a_only = [i for i in common if pa[i] and not pb[i]]
    b_only = [i for i in common if not pa[i] and pb[i]]
    both_fail = sum(1 for i in common if not pa[i] and not pb[i])
    pval = _mcnemar_exact_p(len(a_only), len(b_only))
    print(f"数据集 {a['dataset']}：逐题对照（共同 {len(common)} 题，"
          f"仅 {Path(a['run_dir']).name} 有 {len(only_a)} 题，"
          f"仅 {Path(b['run_dir']).name} 有 {len(only_b)} 题）：")
    print(f"  同对：{both_ok} | 同错：{both_fail}")
    print(f"  仅 {Path(a['run_dir']).name} 对：{len(a_only)} 题 "
          f"{[_disp(i, a['qid_index']) for i in a_only]}")
    print(f"  仅 {Path(b['run_dir']).name} 对：{len(b_only)} 题 "
          f"{[_disp(i, b['qid_index']) for i in b_only]}")
    print(f"  精确 McNemar 检验（双侧）：p = {pval:.4f}")


def compare_runs(dir_a: Path, dir_b: Path) -> None:
    rows_a = analyze_run(dir_a)
    rows_b = analyze_run(dir_b)
    if not rows_a or not rows_b:
        print(f"[错误] 无法读取对照目录：{dir_a} / {dir_b}", file=sys.stderr)
        sys.exit(1)
    # 按数据集名配对，不再只比每目录的第一行（一个轮次目录里常有三集）。
    by_ds_b = {r["dataset"]: r for r in rows_b}
    for a in rows_a:
        b = by_ds_b.get(a["dataset"])
        if b is None:
            print(f"[跳过] 数据集 {a['dataset']} 只在 {dir_a} 出现，{dir_b} 里没有")
            continue
        _compare_pair(a, b)


def main() -> None:
    parser = argparse.ArgumentParser(description="统计 EvalScope 截断率与精度对照")
    parser.add_argument(
        "dirs",
        nargs="*",
        help="EvalScope 输出目录（默认扫描 eval/outputs/*）",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("DIR_A", "DIR_B"),
        help="逐题对比两个输出目录并计算 McNemar p 值",
    )
    args = parser.parse_args()

    if args.compare:
        compare_runs(Path(args.compare[0]), Path(args.compare[1]))
        return

    if args.dirs:
        targets = [Path(d) for d in args.dirs]
    else:
        targets = sorted(Path("eval/outputs").glob("*"))

    all_rows: List[dict] = []
    for t in targets:
        if t.is_dir():
            all_rows.extend(analyze_run(t))
    print_table(all_rows)


if __name__ == "__main__":
    main()
