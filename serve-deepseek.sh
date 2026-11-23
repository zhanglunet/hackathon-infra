#!/usr/bin/env bash
# serve-deepseek.sh — 用 vLLM 0.34.0 在 8 卡机上起 DeepSeek-V4.1-Flash-1017 服务
#
# 状态：基于 vLLM 官方 recipe（recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash，
#       2026-11-23 核对）编写的裸机骨架；尚未在 RTX 6000D-84G 真机验证。
#
# 用法：
#   bash serve-deepseek.sh                                 # 默认 8 卡 TP + Engram CPU 卸载 + 272K
#   ENGRAM_CPU_OFFLOAD=false bash serve-deepseek.sh        # 尝试将 183 GiB Engram 驻留显存（8×84G 需实测余量）
#   LANGUAGE_MODEL_ONLY=1 bash serve-deepseek.sh           # 纯文本赛题跳过 32 层 ViT 省显存
#   DSPARK_TOKENS=5 bash serve-deepseek.sh                 # 开启 DSpark 推测解码（V4.1 已弃用 MTP）
#   TP_SIZE=1 DP_SIZE=8 ENABLE_EP=1 bash serve-deepseek.sh # DEP8 数据+专家并行模式
#
# ── 关键核实项（vLLM 官方 DeepSeek-V4.1-Flash recipe）───────────────
# 1. --tokenizer-mode deepseek_v41：官方权重不含 Jinja chat template，必传！
# 2. --tool-call-parser deepseek_v41 & --reasoning-parser deepseek_v41：
#    V4.1 输出带空格的 DSML 标签，不能用 deepseek_v4（否则静默解析失败、全当普通文本）。
# 3. Engram 内存账本（8×RTX 6000D-84G = 672 GB 总显存）：
#    权重总计约 476 GiB（511 GB），其中 Engram 哈希表占 183.1 GiB。
#    - 默认 ENGRAM_CPU_OFFLOAD=true：Engram 放 pinned 主机内存（UVA 读取、输出无损），
#      显存驻留仅 ~293 GiB（每卡 ~36.6 GiB），留给激活和 KV 的余量充裕。
#    - 若主机内存不足或想测更高吞吐，可试 ENGRAM_CPU_OFFLOAD=false（每卡 ~59.5 GiB 权重）。
# 4. 上下文：原生支持 1M（1,048,576），272000 无需额外 YaRN override；全局 KV 仅 890 B/tok。
# 5. 首次启动耗时较长（加载 511 GB + JIT），设置 VLLM_ENGINE_READY_TIMEOUT_S=3600。
#
# 环境变量（均可覆盖）：
#   MODEL_DIR            默认 ./models
#   MODEL_NAME           默认 DeepSeek-V4.1-Flash-1017
#   HOST / PORT          默认 0.0.0.0 / 8000
#   SERVED_NAME          默认 deepseek-v4.1-flash
#   TP_SIZE              默认 8
#   DP_SIZE              默认不传（1）；DEP 模式可设 8（配合 TP_SIZE=1 ENABLE_EP=1）
#   ENABLE_EP            默认 0；设 1 加 --enable-expert-parallel
#   MAX_MODEL_LEN        默认 272000（比赛要求 272K）
#   KV_CACHE_DTYPE       默认 fp8
#   GPU_MEM_UTIL         默认 0.90
#   MAX_NUM_SEQS         默认 128（recipe 默认值；高并发压测再按 P90 调）
#   MAX_NUM_BATCHED_TOKENS  默认不传（走 vLLM 默认；压测时可设 8192/16384）
#   ENGRAM_CPU_OFFLOAD   默认 true（可设 false 或空字符串跳过 --engram-config）
#   TOKENIZER_MODE       默认 deepseek_v41（设空则不传）
#   TOOL_CALL_PARSER     默认 deepseek_v41（同时开 --enable-auto-tool-choice；设空关闭）
#   REASONING_PARSER     默认 deepseek_v41（设空关闭）
#   ATTENTION_CONFIG     默认不传（让 vLLM 自动选 SM120 适配后端；Blackwell 也可显式传
#                        '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}'）
#   LANGUAGE_MODEL_ONLY  默认 0；设 1 加 --language-model-only（纯文本省显存）
#   DSPARK_TOKENS        默认不传；设 5 开启 DSpark 推测解码（V4.1 专用，取代 MTP）
#   VLLM_USE_RUST_FRONTEND 默认 0（真机验证稳定后可设 1 测高并发收益）

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-./models}"
MODEL_NAME="${MODEL_NAME:-DeepSeek-V4.1-Flash-1017}"
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_NAME="${SERVED_NAME:-deepseek-v4.1-flash}"
TP_SIZE="${TP_SIZE:-8}"
DP_SIZE="${DP_SIZE:-}"
ENABLE_EP="${ENABLE_EP:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-272000}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
ENGRAM_CPU_OFFLOAD="${ENGRAM_CPU_OFFLOAD-true}"
TOKENIZER_MODE="${TOKENIZER_MODE-deepseek_v41}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER-deepseek_v41}"
REASONING_PARSER="${REASONING_PARSER-deepseek_v41}"
ATTENTION_CONFIG="${ATTENTION_CONFIG:-}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-0}"
DSPARK_TOKENS="${DSPARK_TOKENS:-}"

export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
if [ "${VLLM_USE_RUST_FRONTEND:-0}" = "1" ]; then
    export VLLM_USE_RUST_FRONTEND=1
fi

echo "== DeepSeek-V4.1-Flash 预检 =="
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "[错误] 找不到 $MODEL_PATH/config.json，请先运行 bash download-deepseek.sh。"
    exit 1
fi
if ! command -v vllm >/dev/null 2>&1; then
    echo "[错误] 找不到 vllm 命令。请先激活虚拟环境或确认容器内 vllm 可用。"
    exit 1
fi

GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -dc '0-9' || echo 0)"
GPU_TOTAL_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)"
NEED_GPUS=$(( TP_SIZE * ${DP_SIZE:-1} ))
echo "检测到显卡：${GPU_COUNT} 张（单卡 ${GPU_TOTAL_MIB} MiB）；需要卡数：${NEED_GPUS}（TP=${TP_SIZE}, DP=${DP_SIZE:-1}）"
if [ "${GPU_COUNT:-0}" -lt "$NEED_GPUS" ]; then
    echo "[错误] 可用显卡数（$GPU_COUNT）小于 TP×DP（$NEED_GPUS）。"
    exit 1
fi

EXTRA_ARGS=()
if [ -n "$DP_SIZE" ] && [ "$DP_SIZE" != "1" ]; then
    EXTRA_ARGS+=(--data-parallel-size "$DP_SIZE")
fi
if [ "$ENABLE_EP" = "1" ]; then
    EXTRA_ARGS+=(--enable-expert-parallel)
fi
if [ -n "$MAX_NUM_BATCHED_TOKENS" ]; then
    EXTRA_ARGS+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi
if [ -n "$TOKENIZER_MODE" ]; then
    EXTRA_ARGS+=(--tokenizer-mode "$TOKENIZER_MODE")
fi
if [ -n "$TOOL_CALL_PARSER" ]; then
    EXTRA_ARGS+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
fi
if [ -n "$REASONING_PARSER" ]; then
    EXTRA_ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi
if [ -n "$ENGRAM_CPU_OFFLOAD" ]; then
    EXTRA_ARGS+=(--engram-config "{\"cpu_offload\":${ENGRAM_CPU_OFFLOAD}}")
fi
if [ -n "$ATTENTION_CONFIG" ]; then
    EXTRA_ARGS+=(--attention-config "$ATTENTION_CONFIG")
fi
if [ "$LANGUAGE_MODEL_ONLY" = "1" ]; then
    EXTRA_ARGS+=(--language-model-only)
fi
if [ -n "$DSPARK_TOKENS" ]; then
    EXTRA_ARGS+=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${DSPARK_TOKENS}}")
fi

mkdir -p logs
LOG_FILE="logs/vllm-deepseek-$(date +%Y%m%d-%H%M%S).log"
echo "== 启动 DeepSeek-V4.1-Flash（日志写到 $LOG_FILE）=="
echo "模型：$MODEL_PATH | 上下文：$MAX_MODEL_LEN | Engram CPU offload：$ENGRAM_CPU_OFFLOAD"
echo "服务地址：http://<本机 IP>:$PORT/v1/chat/completions"
echo "指标地址：http://<本机 IP>:$PORT/metrics"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --enable-prefix-caching \
    --enable-per-request-metrics \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"
