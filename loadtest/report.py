#!/usr/bin/env python3
"""把 run_ladder.py 的结果目录汇总成 report.md（也可单独运行重新生成）。

用法：python3 loadtest/report.py loadtest/results/<某次>
     python3 loadtest/report.py 结果A 结果B   # 多次结果并排对比「摸高」
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenarios import ORDER, SCENARIOS  # noqa: E402


def load(out_dir):
    meta_p = os.path.join(out_dir, "run_meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    levels = {}
    for p in glob.glob(os.path.join(out_dir, "*", "c*.json")):
        d = json.load(open(p))
        levels.setdefault(d["scenario"], []).append(d)
    for v in levels.values():
        v.sort(key=lambda d: d["concurrency"])
    return meta, levels


def peak(levels):
    """摸高：从低到高连续过线的最高一级（中间断了就以断点前为准）。"""
    best = None
    for d in levels:
        if d["pass"]:
            best = d
        else:
            break
    return best


def f(v, spec, dash="—"):
    return dash if v is None else spec.format(v)


def write(out_dir):
    meta, levels = load(out_dir)
    L = []
    L.append(f"# 压测报告：{meta.get('label') or os.path.basename(os.path.abspath(out_dir))}\n")
    L.append(f"- 开始时间：{meta.get('started', '?')}")
    L.append(f"- 服务：{', '.join(meta.get('urls', []))}；模型名：`{meta.get('model', '?')}`；引擎：{meta.get('engine', '?')}")
    dm = meta.get("data_meta") or {}
    L.append(f"- 数据：scale={dm.get('scale')}，长度口径={((dm.get('levels') or [{}])[0]).get('length_mode', '?')}，"
             f"预热={'开' if meta.get('warmup') else '关'}，ignore_eos={'开' if meta.get('ignore_eos') else '关'}，"
             f"允许错误率={meta.get('max_error_rate', 0):.0%}")
    if dm.get("scale") not in (None, 1, 1.0):
        L.append(f"- ⚠️ scale={dm.get('scale')}：长度被缩小过，**结果只用于调试脚本，不能当成绩**")
    L.append("- 判定：P90 TTFT 和 P90 TPOT 都不超过 SLO、错误率不超标，才算该级过线；摸高 = 从低到高连续过线的最高一级")
    L.append("- 口径：自测，复测为准；数字必须注明在哪台机器、什么部署方式下测的\n")

    L.append("## 总览（摸高）\n")
    L.append("| 场景 | SLO（P90） | 摸高并发 | 该级 Goodput (req/s) | 该级输出吞吐 (tok/s) | 实测命中 / 目标 |")
    L.append("|---|---|---|---|---|---|")
    for name in ORDER:
        if name not in levels:
            continue
        sc = SCENARIOS[name]
        b = peak(levels[name])
        slo = f"TTFT≤{sc['ttft_s']:g}s，TPOT≤{sc['tpot_ms']:g}ms"
        if b:
            L.append(f"| {sc['title']} | {slo} | **{b['concurrency']}** | {f(b['goodput_rps'], '{:.3f}')} | "
                     f"{f(b['output_tps'], '{:,.0f}')} | {f(b['measured_cache_hit'], '{:.2f}')} / {sc['cache_hit']:.2f} |")
        else:
            L.append(f"| {sc['title']} | {slo} | ❌ 最低一级就不过线 | — | — | — |")

    for name in ORDER:
        if name not in levels:
            continue
        sc = SCENARIOS[name]
        L.append(f"\n## {sc['title']}（{name}）：输入 {sc['input_tokens']:,} / 输出 {sc['output_tokens']:,} / 命中 {sc['cache_hit']}\n")
        L.append("| 并发 | 请求 | 错误 | P50 TTFT | **P90 TTFT** | P50 TPOT | **P90 TPOT** | 输入均值 | 输出均值 | Goodput (req/s) | 输出 tok/s | 实测命中 | 结果 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for d in levels[name]:
            res = "✅" if d["pass"] else "❌ " + "；".join(d["fail_reasons"])
            L.append(f"| {d['concurrency']} | {d['requests']} | {d['errors']} | {f(d['p50_ttft_s'], '{:.2f}s')} | "
                     f"{f(d['p90_ttft_s'], '{:.2f}s')} | {f(d['p50_tpot_ms'], '{:.1f}ms')} | {f(d['p90_tpot_ms'], '{:.1f}ms')} | "
                     f"{f(d.get('input_tokens_mean'), '{:,.0f}')} | {f(d.get('output_tokens_mean'), '{:,.0f}')} | "
                     f"{f(d['goodput_rps'], '{:.3f}')} | {f(d['output_tps'], '{:,.0f}')} | "
                     f"{f(d['measured_cache_hit'], '{:.2f}')} | {res} |")
        off = [d for d in levels[name] if d.get("measured_cache_hit") is not None
               and abs(d["measured_cache_hit"] - sc["cache_hit"]) > 0.10]
        if off:
            L.append(f"\n⚠️ 实测命中率和目标 {sc['cache_hit']:.2f} 差了 0.1 以上（并发 "
                     + "、".join(f"{d['concurrency']}→{d['measured_cache_hit']:.2f}" for d in off)
                     + "）。偏低：KV 缓存装不下这么多会话、被挤掉了；偏高：可能有别的流量或缓存没隔离。这一级的数和比赛口径不一致，要先查原因。")
        if any(d.get("measured_cache_hit") is None for d in levels[name]):
            L.append("\n（部分级别没读到 /metrics 的 prefix_cache 计数，实测命中率显示为 —）")
        errs = sorted({e for d in levels[name] for e in d.get("error_samples", [])})
        if errs:
            L.append("\n错误样例：")
            L.extend(f"- `{e}`" for e in errs[:5])
        early = sum(d.get("stopped_early") or 0 for d in levels[name])
        if early:
            L.append(f"\n⚠️ 有 {early} 个请求没到设定输出长度就自己结束了（ignore_eos 没生效？），TPOT 和吞吐会偏乐观。")

    path = os.path.join(out_dir, "report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return path


def compare(dirs):
    """多次结果并排：每个场景的摸高并发和 Goodput。"""
    rows = ["| 场景 | " + " | ".join(os.path.basename(os.path.abspath(d)) for d in dirs) + " |",
            "|---|" + "---|" * len(dirs)]
    loaded = [load(d)[1] for d in dirs]
    for name in ORDER:
        cells = []
        for lv in loaded:
            b = peak(lv.get(name, []))
            cells.append("—" if name not in lv else (f"c{b['concurrency']} / {f(b['goodput_rps'], '{:.3f}')} req/s" if b else "❌"))
        rows.append(f"| {SCENARIOS[name]['title']} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if len(sys.argv) == 2:
        print(write(sys.argv[1]))
    else:
        print(compare(sys.argv[1:]))
