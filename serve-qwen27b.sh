#!/usr/bin/env bash
# serve-qwen27b.sh — 用 vLLM 起 Qwen3.8-27B 的 OpenAI 兼容服务
#
# 用法：
#   bash serve-qwen27b.sh            # 前台运行，Ctrl-C 停止；日志写到 logs/
#   nohup bash serve-qwen27b.sh > logs/nohup.out 2>&1 &   # 后台运行
#
# 环境变量（都可改）：
#   MODEL_DIR      权重根目录，默认 ./models
#   MODEL_NAME     子目录名，默认 Qwen3.8-27B-FP8（改 BF16 时改为 Qwen3.8-27B）
#   PORT           默认 8000；HOST 默认 0.0.0.0（局域网/公网可访问，见安全提示）
#   SERVED_NAME    API 里显示的模型名，默认 qwen3.8-27b
#   MAX_MODEL_LEN  上下文上限，默认 262144（= 模型原生上限，见下）
#   KV_CACHE_DTYPE KV 缓存精度，默认 fp8
#   GPU_MEM_UTIL   显存占用比，默认 0.90
#   MAX_NUM_SEQS   最大并发序列数，默认 32
#   MAX_NUM_BATCHED_TOKENS  chunked prefill 步长，默认 65536（已验证，见下）
#   ATTENTION_BACKEND  全注意力后端，默认 FLASHINFER（已验证最快，见下）；
#     设为空字符串则不传该参数、走 vLLM 默认（FA3）
#   GDN_PREFILL_BACKEND  GDN prefill 后端，默认不传（auto）；可设 triton 等
#   COMPILATION_MODE  编译模式，默认不传；设 3 = VLLM_COMPILE（本模型上是空转，见下）
#   VLLM_USE_BREAKABLE_CUDAGRAPH  默认 0（退出自动开 breakable 图的路径，见下）
#
# ── 已验证的调优项（2026-11-21～11-23，Modal H100/H200，见 devlog）───
# 以下默认值照抄 modal/modal_vllm_qwen27b.py 里的已验证值，不要自己改数字；
# 每项括号里是来源。注意：这些值是在 Modal 上验证的，裸机/Blackwell 上
# 仍须重测，见各条“待真机验证”。
#   VLLM_ALLOW_LONG_MAX_MODEL_LEN=1：MAX_MODEL_LEN=272000 时 vLLM 0.34.0
#     会先校验 max_model_len ≤ max_position_embeddings，不设它直接报
#     ValueError——2026-11-21 Modal 首次部署因此循环重启约 11 分钟。
#     （待真机验证：裸机 vllm serve 同样走这段校验，逻辑相同，但未实跑。）
#   --max-num-batched-tokens 65536：271,500 token 的提示只切 2 片；
#     8192→65536 只快 1.5%（50.92 秒 vs 51.72 秒），分片不是瓶颈；
#     262144 初始化失败。副作用：KV 容量 2,955,530→2,592,357 token。
#   --attention-backend FLASHINFER：第 6 轮最快，冷 41.09/43.8 秒，
#     比 FA3（52.52/54.94 秒）快约 21%。前提：flashinfer + nvcc 已装
#     （setup-vllm.sh 第 6 步）；没装时引擎在 warmup JIT 编译处报错，
#     像第 5 轮那样起不来——此时设 ATTENTION_BACKEND="" 回落 FA3。
#   VLLM_USE_BREAKABLE_CUDAGRAPH=0：第 4 轮核实，vLLM 对 Qwen3.5 架构
#     默认自动开 breakable cudagraph，会把 compilation mode 置回 NONE；
#     设 0 退出该路径。但 mode=3 本身对本模型也是空转（不支持
#     torch.compile），所以 COMPILATION_MODE 默认不传。
#   --gdn-prefill-backend：默认不传（auto）。GDN 全部内核仅约 1.4 秒
#     （冷请求 GPU 时间的 3%），triton 轮 53.04 秒无改善，不是瓶颈。
#
# ── 上下文长度的关键说明 ──────────────────────────────────────────
# Qwen3.8-27B 原生上下文是 262144 token（config.json 的 max_position_embeddings）。
# 比赛要求 272K（272000），比原生多 3.8%，必须用 YaRN 外推：
#   把 MAX_MODEL_LEN 设成 272000，脚本会自动加 --hf-overrides 传 YaRN 参数
#   （factor≈1.04）。这是实验性路径：首次使用请先跑 smoke-test.sh，
#   再用 EvalScope 跑精度对照，确认 ≤2% 的精度门后再用于正式自测。
#   如 YaRN 报错或精度不达标，退回 262144。
#
# ── 显存账本（按 config.json 实测算的，不是估的）─────────────────
# 模型 64 层，其中全注意力层每 4 层 1 个，共 16 层；其余 48 层是线性注意力
# （Gated DeltaNet），状态固定大小、不随 token 数增长。
# KV 缓存只算 16 个全注意力层：每 token 每层 = 2(K/V) × 4 头 × 256 维：
#   BF16：4 KB/token/层 → 16 层 = 64 KB/token → 262144 token ≈ 17.2 GB
#   FP8 ：2 KB/token/层 → 16 层 = 32 KB/token → 262144 token ≈  8.6 GB
# FP8 权重约 30.9 GB + FP8 KV 约 8.6 GB ≈ 40 GB，80 GB 卡绰绰有余。
# BF16 权重约 55.5 GB + BF16 KV 17.2 GB ≈ 73 GB，几乎顶满——
# BF16 时请把 KV_CACHE_DTYPE 保持 fp8，或把 MAX_MODEL_LEN 降到 131072。
#
# ── 安全提示 ──────────────────────────────────────────────────────
# HOST=0.0.0.0 表示任何能连到这台机器的人都能调接口。
# AutoDL 实例通常只有你能访问；如实例有公网 IP 且不想公开，
# 请把 HOST 改成 127.0.0.1，或在 AutoDL 控制台只映射你自己的端口。
#
# 标志核对（vLLM 0.34.0 官方 CLI 参考 docs.vllm.ai，2026-11-20 核对；
#   新增 4 项 2026-11-23 按 modal_vllm_qwen27b.py 已验证值补）：
#   --host/--port/--served-model-name/--tensor-parallel-size/--max-model-len/
#   --quantization/--kv-cache-dtype/--gpu-memory-utilization/
#   --enable-prefix-caching/--enable-per-request-metrics/--max-num-seqs/
#   --max-num-batched-tokens/--attention-backend/--gdn-prefill-backend/
#   --compilation-config 均存在。
#   注意：vLLM 0.34 没有 --enable-metrics 这个参数——/metrics 端口默认就开，
#   不用加参数；--enable-per-request-metrics 是把每请求的耗时写进响应里。

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-./models}"
MODEL_NAME="${MODEL_NAME:-Qwen3.8-27B-FP8}"
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_NAME="${SERVED_NAME:-qwen3.8-27b}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
TP_SIZE="${TP_SIZE:-1}"
# 已验证的调优项（2026-11-23 补，默认值照抄 modal 已验证值，见文件头说明）：
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASHINFER}"
GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-}"
COMPILATION_MODE="${COMPILATION_MODE:-}"
# vLLM 0.34.0 校验 max_model_len ≤ max_position_embeddings：272000 超原生
# 262144 时必须设此变量放行，否则直接报 ValueError（2026-11-21 Modal
# 首次部署因此循环重启约 11 分钟；modal_vllm_qwen27b.py _CONTAINER_ENV 同款）。
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# vLLM 对 Qwen3.5 架构默认自动开 breakable cudagraph，会把 compilation
# mode 置回 NONE（第 4 轮源码定位）；设 0 退出该路径。默认 0。
export VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}"
NATIVE_LEN=262144

echo "== 预检 =="
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "[错误] 找不到 $MODEL_PATH/config.json，请先运行 bash download-model.sh。"
    exit 1
fi
if ! command -v vllm >/dev/null 2>&1; then
    echo "[错误] 找不到 vllm 命令。请先 source .venv/bin/activate，再运行本脚本。"
    echo "       （或先跑 bash setup-vllm.sh 安装）"
    exit 1
fi
GPU_TOTAL_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)"
echo "显卡总显存：${GPU_TOTAL_MIB} MiB；模型：$MODEL_PATH；上下文：$MAX_MODEL_LEN；KV 精度：$KV_CACHE_DTYPE"
if [ "${GPU_TOTAL_MIB:-0}" -lt 70000 ]; then
    echo "[警告] 显存小于 70 GB，单卡跑 27B 可能 OOM。先试 --max-model-len 32768 再说。"
fi

# FP8 权重：config.json 里有 quant_method=fp8 时显式告诉 vLLM；
# BF16 权重则不传 --quantization，让 vLLM 按 dtype 加载。
QUANT_ARGS=()
IS_FP8=0
if grep -q '"quant_method": *"fp8"' "$MODEL_PATH/config.json" 2>/dev/null; then
    QUANT_ARGS=(--quantization fp8)
    IS_FP8=1
    echo "检测到 FP8 权重，使用 --quantization fp8。"
else
    echo "未检测到 FP8 量化配置，按 BF16 加载（显存吃紧时请降 MAX_MODEL_LEN）。"
fi

# FP8 硬件检查：FP8 张量核需要 Hopper（算力 9.0，如 H100/H800/H20）
# 或 Blackwell（12.0，如 RTX 6000D）。A100/A800 是 Ampere（8.0），
# 没有 FP8 张量核，跑不了 FP8 权重——此时请改下 BF16 版再跑。
if [ "$IS_FP8" -eq 1 ]; then
    COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || echo 0)"
    echo "显卡算力：$COMPUTE_CAP（FP8 需要 ≥ 9.0）"
    if ! python3 -c "import sys; sys.exit(0 if float('$COMPUTE_CAP') >= 9.0 else 1)" 2>/dev/null; then
        echo "[错误] 这张卡算力 $COMPUTE_CAP，不支持 FP8 计算，跑不了 FP8 权重。"
        echo "       两个办法：① 换 H800/H100/H20/RTX 6000D 这类 9.0 以上的卡；"
        echo "       ② 改下 BF16 版：bash download-model.sh --bf16，再把 KV_CACHE_DTYPE 保持 fp8、MAX_MODEL_LEN 降到 131072。"
        exit 1
    fi
fi

# 272K 超出原生 262144 时自动加 YaRN 外推（实验性，见文件头说明）。
ROPE_ARGS=()
if [ "$MAX_MODEL_LEN" -gt "$NATIVE_LEN" ]; then
    YARN_FACTOR="$(python3 -c "print(f'{$MAX_MODEL_LEN / $NATIVE_LEN:.4f}')")"
    echo "[注意] MAX_MODEL_LEN=$MAX_MODEL_LEN 超出原生 $NATIVE_LEN，启用 YaRN 外推（factor=$YARN_FACTOR）。"
    echo "       首次使用请务必跑精度对照；如启动报错或精度不达标，退回 MAX_MODEL_LEN=262144。"
    ROPE_ARGS=(--hf-overrides "{\"rope_parameters\": {\"rope_type\": \"yarn\", \"factor\": $YARN_FACTOR, \"original_max_position_embeddings\": $NATIVE_LEN, \"rope_theta\": 10000000, \"partial_rotary_factor\": 0.25, \"mrope_section\": [11, 11, 10], \"mrope_interleaved\": true}}")
fi

# 已验证的调优 flag（2026-11-23 补，默认值照抄 modal 已验证值）：
# ATTENTION_BACKEND 为空时不传参数（走 vLLM 默认 FA3）；非空时传
# --attention-backend。FLASHINFER 须 flashinfer + nvcc 已装，否则引擎在
# warmup JIT 编译处报错（第 5 轮教训）——此时设 ATTENTION_BACKEND="" 回落。
ATTN_ARGS=()
if [ -n "$ATTENTION_BACKEND" ]; then
    ATTN_ARGS=(--attention-backend "$ATTENTION_BACKEND")
fi
# GDN prefill 后端：默认不传（auto）；显式指定时才传。
GDN_ARGS=()
if [ -n "$GDN_PREFILL_BACKEND" ]; then
    GDN_ARGS=(--gdn-prefill-backend "$GDN_PREFILL_BACKEND")
fi
# 编译模式：默认不传。设了才传 --compilation-config '{"mode": N}'。
# 注意：第 4 轮已核实 mode=3 对本模型是空转（不支持 torch.compile），
# 传了也不提速；留这个开关只为以后换模型时复用。
COMP_ARGS=()
if [ -n "$COMPILATION_MODE" ]; then
    COMP_ARGS=(--compilation-config "{\"mode\": $COMPILATION_MODE}")
fi

mkdir -p logs
LOG_FILE="logs/vllm-serve-$(date +%Y%m%d-%H%M%S).log"
echo "== 启动 vLLM（日志同时写到 $LOG_FILE）=="
echo "服务地址：http://<本机 IP>:$PORT/v1/chat/completions"
echo "指标地址：http://<本机 IP>:$PORT/metrics"
echo "按 Ctrl-C 停止服务。"

# 前台运行 + tee 写日志。想后台跑：nohup bash serve-qwen27b.sh > logs/nohup.out 2>&1 &
vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    "${QUANT_ARGS[@]}" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    "${ATTN_ARGS[@]}" \
    "${GDN_ARGS[@]}" \
    "${COMP_ARGS[@]}" \
    --enable-prefix-caching \
    --enable-per-request-metrics \
    "${ROPE_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"

# ── 备选命令（出问题时抄过去用）────────────────────────────────────
# 短上下文兜底（显存不够 / 只想先验证服务能起来）：
#   MAX_MODEL_LEN=32768 KV_CACHE_DTYPE=fp8 bash serve-qwen27b.sh
# BF16 权重 + 短上下文：
#   MODEL_NAME=Qwen3.8-27B MAX_MODEL_LEN=131072 KV_CACHE_DTYPE=fp8 bash serve-qwen27b.sh
# 调试用 eager 模式（慢，但能绕过 CUDA 图相关报错）：
#   在上面命令后追加 --enforce-eager
# FlashInfer 起不来时（warmup 报找不到 nvcc）：先确认 setup-vllm.sh 第 6 步
# 已装 flashinfer + nvcc；应急可回落 FA3（第 5 轮默认后端，冷 52.52 秒）：
#   ATTENTION_BACKEND= bash serve-qwen27b.sh
# 只看文本、不加载视觉塔（比赛只测文本时可省一点显存）：
#   追加 --language-model-only
