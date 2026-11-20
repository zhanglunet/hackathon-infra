#!/usr/bin/env python3
"""下载四项公开评测集到 eval/data/（自测代理，非官方 190 题）。

官方 190 题由组织方持有、不公开；这里下的是四项公开数据集的全集，
只做自测参考。用法（在租来的实例或本机上都行）：

    pip install datasets huggingface_hub
    python3 eval/fetch-benchmarks.py [--out eval/data]

数据集（2026-11-20 核对 ID 存在）：
- AIME 2025：MathArena/aime_2025，train 30 题
- GPQA：Idavidrein/gpqa，gpqa_main.csv（主集 448 题）
- LiveCodeBench Lite：livecodebench/code_generation_lite，
  test.jsonl ~ test6.jsonl（六个版本）
- BFCL v3：gorilla-llm/Berkeley-Function-Calling-Leaderboard，
  BFCL_v3_*.json 共 25 个文件
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files


def fetch_aime(out: Path) -> int:
    ds = load_dataset("MathArena/aime_2025", split="train")
    rows = [dict(r) for r in ds]
    p = out / "aime_2025.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def fetch_gpqa(out: Path) -> int:
    p = Path(hf_hub_download(repo_id="Idavidrein/gpqa",
                              filename="gpqa_main.csv", repo_type="dataset"))
    dest = out / "gpqa_main.csv"
    dest.write_bytes(p.read_bytes())
    with open(dest, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.reader(f)) - 1  # 去表头


def fetch_livecodebench(out: Path) -> int:
    d = out / "livecodebench"
    d.mkdir(parents=True, exist_ok=True)
    total = 0
    for i in range(1, 7):
        name = "test.jsonl" if i == 1 else f"test{i}.jsonl"
        p = Path(hf_hub_download(repo_id="livecodebench/code_generation_lite",
                                  filename=name, repo_type="dataset"))
        dest = d / name
        dest.write_bytes(p.read_bytes())
        with open(dest, encoding="utf-8") as f:
            total += sum(1 for _ in f)
    return total


def fetch_bfcl(out: Path) -> int:
    d = out / "bfcl_v3"
    d.mkdir(parents=True, exist_ok=True)
    files = [f for f in list_repo_files(
        "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        repo_type="dataset")
        if f.startswith("BFCL_v3_") and f.endswith(".json")]
    for name in sorted(files):
        p = Path(hf_hub_download(
            repo_id="gorilla-llm/Berkeley-Function-Calling-Leaderboard",
            filename=name, repo_type="dataset"))
        (d / name).write_bytes(p.read_bytes())
    return len(files)


JOBS = [
    ("AIME 2025", fetch_aime),
    ("GPQA main", fetch_gpqa),
    ("LiveCodeBench Lite", fetch_livecodebench),
    ("BFCL v3", fetch_bfcl),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="下载四项公开评测集（自测用）")
    ap.add_argument("--out", default="eval/data",
                    help="输出目录（默认 eval/data）")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    failed = []
    for name, fn in JOBS:
        try:
            n = fn(out)
            print(f"[ok] {name}: {n}")
        except Exception as e:  # noqa: BLE001 - 逐项失败不中断其余下载
            failed.append(name)
            print(f"[fail] {name}: {e}", file=sys.stderr)
    if failed:
        print(f"失败 {len(failed)} 项：{', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"[done] 全部下到 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
