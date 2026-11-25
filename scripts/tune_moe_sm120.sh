#!/usr/bin/env bash
# tune_moe_sm120.sh — 在比赛机（RTX 6000D SM120）上生成 vLLM 0.34.0 Triton MoE 调优表
#
# 背景（Issue #1 源码调研新 P0）：
#   1. vLLM 0.34.0 内置 Triton MoE 调优表靠 torch.cuda.get_device_name() 精确匹配，
#      没有 RTX 6000D 条目，会回落默认慢配置（警告 Using default MoE config）。
#   2. benchmarks/kernels/benchmark_moe.py 不在 pip wheel 里，需从 v0.34.0 源码拉取。
#   3. 原版 benchmark_moe.py 的 get_model_params 不认识 Glm5NextForCausalLM 与
#      DeepseekV41ForCausalLM（会错读 num_local_experts 报 has no MoE experts），
#      需自动打补丁归入读取 n_routed_experts 的分支。
#
# 用法（在比赛机激活 vLLM 0.34.0 环境后执行）：
#   # 仅打印显卡设备名并准备打好补丁的 benchmark_moe.py（不跑调优）
#   bash scripts/tune_moe_sm120.sh --prep-only
#
#   # 为 GLM-5.3-Flash（TP=8）生成调优表到 ./tuned_configs
#   MODEL_PATH=/data/models/GLM-5.3-Flash TP_SIZE=8 SAVE_DIR=./tuned_configs bash scripts/tune_moe_sm120.sh
#
#   # 为 DeepSeek-V4.1-Flash-1017（TP=8）生成调优表
#   MODEL_PATH=/data/models/DeepSeek-V4.1-Flash-1017 TP_SIZE=8 SAVE_DIR=./tuned_configs bash scripts/tune_moe_sm120.sh
#
# 生成后启动服务时传入：
#   VLLM_TUNED_CONFIG_FOLDER=./tuned_configs bash serve-glm.sh

set -euo pipefail

PREP_ONLY=0
if [ "${1:-}" = "--prep-only" ]; then PREP_ONLY=1; fi

VLLM_TAG="${VLLM_TAG:-v0.34.0}"
WORK_DIR="${WORK_DIR:-./.cache/moe_tuner}"
SAVE_DIR="${SAVE_DIR:-./tuned_configs}"
MODEL_PATH="${MODEL_PATH:-}"
TP_SIZE="${TP_SIZE:-8}"
DTYPE="${DTYPE:-fp8_w8a8}"   # fp8_w8a8 | float16 | bfloat16
BENCH_PY="$WORK_DIR/benchmark_moe.py"

echo "== 1/4 检查显卡设备名（用于确认 vLLM MoE 配置表键名）=="
python3 - <<'PYEOF'
try:
    import torch
    if torch.cuda.is_available():
        print("GPU 数量：", torch.cuda.device_count())
        print("torch.cuda.get_device_name(0)：", repr(torch.cuda.get_device_name(0)))
        print("Compute Capability：", torch.cuda.get_device_capability(0))
    else:
        print("[提示] 当前环境无可用 CUDA GPU（仅离线准备脚本）。")
except Exception as e:
    print(f"[提示] 未能导入 torch：{e}")
PYEOF

echo "== 2/4 准备 benchmark_moe.py（优先用仓库内已打补丁的副本，离线可用）=="
mkdir -p "$WORK_DIR" "$SAVE_DIR"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
BUNDLED="$SCRIPT_DIR/moe_tuner/benchmark_moe.py"
if [ -f "$BUNDLED" ]; then
    echo "用仓库 bundled 副本：$BUNDLED（已打 Glm5Next/DeepseekV41 补丁）"
    cp "$BUNDLED" "$BENCH_PY"
elif [ ! -f "$BENCH_PY" ]; then
    RAW_URL="https://raw.githubusercontent.com/vllm-project/vllm/${VLLM_TAG}/benchmarks/kernels/benchmark_moe.py"
    echo "下载：$RAW_URL"
    curl -fsSL "$RAW_URL" -o "$BENCH_PY"
else
    echo "已存在：$BENCH_PY（跳过下载）"
fi

echo "== 3/4 为 Glm5Next / DeepseekV41 架构自动打补丁（n_routed_experts）=="
python3 - "$BENCH_PY" <<'PYEOF'
from pathlib import Path
import sys

p = Path(sys.argv[1])
src = p.read_text(encoding="utf-8")

# Issue #1 核实：原脚本把 DeepSeek 系列归在一组读 n_routed_experts，
# 但缺 Glm5NextForCausalLM / Glm5NextForConditionalGeneration / DeepseekV41*。
targets = [
    '"Glm5NextForCausalLM"',
    '"Glm5NextForConditionalGeneration"',
    '"DeepseekV41ForCausalLM"',
    '"DeepseekV41ForConditionalGeneration"',
]
anchor = '"DeepseekV3ForCausalLM"'
if anchor not in src:
    anchor = "'DeepseekV3ForCausalLM'"

if all(t.strip('"') in src for t in targets):
    print("[跳过] 补丁已存在。")
elif anchor in src:
    addition = ", ".join([anchor] + [t for t in targets if t.strip('"') not in src])
    patched = src.replace(anchor, addition, 1)
    p.write_text(patched, encoding="utf-8")
    print("[完成] 已将 Glm5Next / DeepseekV41 架构加入 n_routed_experts 分支。")
else:
    print("[警告] 未找到锚点 DeepseekV3ForCausalLM，请人工核对 benchmark_moe.py 的 get_model_params。")
PYEOF

if [ "$PREP_ONLY" -eq 1 ]; then
    echo "[完成] --prep-only 模式结束。打好补丁的脚本位于：$BENCH_PY"
    exit 0
fi

if [ -z "$MODEL_PATH" ]; then
    echo "[错误] 请指定 MODEL_PATH（例如 MODEL_PATH=/data/models/GLM-5.3-Flash）。"
    exit 1
fi

echo "== 4/4 运行 Triton MoE 调优（MODEL=$MODEL_PATH, TP=$TP_SIZE, DTYPE=$DTYPE, SAVE_DIR=$SAVE_DIR）=="
python3 "$BENCH_PY" \
    --model "$MODEL_PATH" \
    --tp-size "$TP_SIZE" \
    --dtype "$DTYPE" \
    --tune \
    --save-dir "$SAVE_DIR"

echo
echo "[完成] 调优配置已写入：$SAVE_DIR"
echo "启动服务时请加上：VLLM_TUNED_CONFIG_FOLDER=$SAVE_DIR"
