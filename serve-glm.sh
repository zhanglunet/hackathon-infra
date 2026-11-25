#!/usr/bin/env bash
# serve-glm.sh — 用 vLLM 0.34.0 在 8 卡机上起 GLM-5.3-Flash 服务（第二款自选模型）
#
# 状态：基于 vLLM 官方 recipe（recipes.vllm.ai/zai-org/GLM-5.3-Flash，
#       2026-11-24 核对）编写的裸机骨架；尚未在 RTX 6000D-84G 真机验证。
#
# 用法：
#   bash serve-glm.sh                                    # 默认 8 卡 TP + 272K + FP8 KV + glm47 parser
#   LANGUAGE_MODEL_ONLY=1 bash serve-glm.sh              # 纯文本赛题跳过视觉编码器省显存
#   MTP_TOKENS=3 bash serve-glm.sh                       # 开启内置 MTP 推测解码（须重测精度）
#   MOE_BACKEND=triton bash serve-glm.sh                 # 对比不同 MoE 后端（triton/deep_gemm/cutlass/flashinfer_cutlass）
#   TP_SIZE=1 DP_SIZE=8 ENABLE_EP=1 bash serve-glm.sh    # DEP8 数据+专家并行模式
#
# ── 关键核实项（vLLM 官方 GLM-5.3-Flash recipe & Issue #1 调研）─────
# 1. --tool-call-parser glm47 & --reasoning-parser glm47：
#    注意不能用 glm（会静默解析失败、吞输出且 tool_calls 为空），必须用 glm47。
# 2. 显存账本（8×RTX 6000D-84G = 672 GB 总显存）：
#    原生 FP8 权重约 306 GiB，8 卡 TP 每卡仅 ~38.3 GiB，留给 34 层 KDA 状态与 11 层
#    NoPE 稀疏 MLA KV 的余量充裕（注：KDA 状态按并发序列分配，OOM 时先降 MAX_NUM_SEQS）。
# 3. Blackwell（SM120）支持 --kv-cache-dtype fp8（Hopper 不支持、须设 auto/bf16）。
# 4. Issue #1 核实：SM120 用不了 DeepGEMM MegaMoE（限 SM100），真机务必先跑
#    scripts/tune_moe_sm120.sh 生成 Triton MoE 调优表并用 VLLM_TUNED_CONFIG_FOLDER 加载。
# 5. 上下文：原生 1M（1,048,576），272000 无需 YaRN override。
#
# 环境变量（均可覆盖）：
#   MODEL_DIR            默认 ./models
#   MODEL_NAME           默认 GLM-5.3-Flash
#   HOST / PORT          默认 0.0.0.0 / 8000
#   SERVED_NAME          默认 glm-5.3-flash
#   TP_SIZE              默认 8
#   DP_SIZE              默认不传（1）；DEP 模式可设 8（配合 TP_SIZE=1 ENABLE_EP=1）
#   ENABLE_EP            默认 0；设 1 加 --enable-expert-parallel
#   MAX_MODEL_LEN        默认 272000（比赛要求 272K）
#   KV_CACHE_DTYPE       默认 fp8（Blackwell 支持；Hopper 须改为 auto）
#   GPU_MEM_UTIL         默认 0.90
#   MAX_NUM_SEQS         默认 64（34 层 KDA 状态随并发增长；压测时配合 80 阶梯实测调整）
#   MAX_NUM_BATCHED_TOKENS  默认不传（走 vLLM 默认；压测时可设 8192/16384）
#   TOKENIZER_MODE       默认不传（Python frontend 用 auto；开 Rust frontend 时可设 glm5_next）
#   TOOL_CALL_PARSER     默认 glm47（同时开 --enable-auto-tool-choice；设空关闭）
#   REASONING_PARSER     默认 glm47（设空关闭）
#   MOE_BACKEND          默认不传（可选 triton / deep_gemm / cutlass / flashinfer_cutlass，真机对照用）
#   LANGUAGE_MODEL_ONLY  默认 0；设 1 加 --language-model-only（纯文本省显存）
#   MTP_TOKENS           默认不传；设 3 开启内置 MTP 推测解码（--speculative-config）
#   VLLM_USE_RUST_FRONTEND 默认 0（真机纯文本验证稳定后可设 1 + TOKENIZER_MODE=glm5_next）

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-./models}"
MODEL_NAME="${MODEL_NAME:-GLM-5.3-Flash}"
# NFS 上的目录名不一定和 MODEL_NAME 一致（可能多一层组织名），允许直接传 MODEL_PATH（issue #14 ①-1）
MODEL_PATH="${MODEL_PATH:-$MODEL_DIR/$MODEL_NAME}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_NAME="${SERVED_NAME:-glm-5.3-flash}"
TP_SIZE="${TP_SIZE:-8}"
DP_SIZE="${DP_SIZE:-}"
ENABLE_EP="${ENABLE_EP:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-272000}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
TOKENIZER_MODE="${TOKENIZER_MODE:-}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER-glm47}"
REASONING_PARSER="${REASONING_PARSER-glm47}"
MOE_BACKEND="${MOE_BACKEND:-}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-0}"
MTP_TOKENS="${MTP_TOKENS:-}"

export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
if [ "${VLLM_USE_RUST_FRONTEND:-0}" = "1" ]; then
    export VLLM_USE_RUST_FRONTEND=1
fi
# 预热与调优缓存（响应 #1 调研：SM120 需自调 Triton MoE 配置表 + 持久化 FlashInfer 缓存）
[ -n "${VLLM_CACHE_ROOT:-}" ] && export VLLM_CACHE_ROOT
[ -n "${VLLM_TUNED_CONFIG_FOLDER:-}" ] && export VLLM_TUNED_CONFIG_FOLDER
[ -n "${VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR:-}" ] && export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR
[ -n "${VLLM_DEEP_GEMM_WARMUP:-}" ] && export VLLM_DEEP_GEMM_WARMUP

echo "== GLM-5.3-Flash 预检 =="
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "[错误] 找不到 $MODEL_PATH/config.json，请确认 MODEL_DIR / MODEL_PATH 是否指向 NFS 上的权重目录（比赛权重已预置，不要自己下载）。"
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
if [ -n "$MOE_BACKEND" ]; then
    EXTRA_ARGS+=(--moe-backend "$MOE_BACKEND")
fi
if [ "$LANGUAGE_MODEL_ONLY" = "1" ]; then
    EXTRA_ARGS+=(--language-model-only)
fi
if [ -n "$MTP_TOKENS" ]; then
    EXTRA_ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}")
fi

# 日志目录可配置：容器里代码目录只读时，默认 ./logs 写不进去会直接退出（issue #14 ①-2）
LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/vllm-glm-$(date +%Y%m%d-%H%M%S).log"
echo "== 启动 GLM-5.3-Flash（日志写到 $LOG_FILE）=="
echo "模型：$MODEL_PATH | 上下文：$MAX_MODEL_LEN | KV：$KV_CACHE_DTYPE | MAX_NUM_SEQS：$MAX_NUM_SEQS"
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
    --enable-prompt-tokens-details \
    --enable-per-request-metrics \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"
