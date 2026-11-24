#!/usr/bin/env bash
# download-glm.sh — 下载 GLM-5.3-Flash 权重（ModelScope 优先，HF 备用）
#
# 用法：
#   bash download-glm.sh                              # 默认下官方 FP8 版到 ./models/GLM-5.3-Flash
#   MODEL_DIR=/data/models bash download-glm.sh       # 挂数据盘（推荐，需 ≥ 420 GB 空闲）
#   nohup bash download-glm.sh > logs/download-glm.log 2>&1 &
#
# 权重规格（2026-11-24 核对自 vLLM 官方 recipe：recipes.vllm.ai/zai-org/GLM-5.3-Flash）：
#   - 仓库：zai-org/GLM-5.3-Flash（默认即原生 FP8 版；BF16 版为 zai-org/GLM-5.3-Flash-BF16）
#   - 体积：约 328 GB（306 GiB）
#   - 结构：321B 总参 / 18B 激活 MoE，45 层（34 层 KDA 线性注意力 + 11 层 NoPE 稀疏 MLA），
#           8/288 专家，原生 1M 上下文，内置 1 层 MTP 草稿头
#
# 环境变量（可按组织方实际指定权重名覆盖）：
#   MODEL_DIR        权重根目录，默认 ./models
#   MODEL_ID         ModelScope/HF 仓库名，默认 zai-org/GLM-5.3-Flash
#   LOCAL_NAME       本地子目录名，默认取 MODEL_ID 最后一段
#   EXPECTED_SHARDS  预期 safetensors 分片数，默认 0（从 model.safetensors.index.json 动态核对，避免硬编码错位）
#   EXPECTED_GB      预期体积（GB），默认 328
#   NEED_FREE_GB     下载前要求空闲磁盘（GB），默认 420

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-./models}"
MODEL_ID="${MODEL_ID:-zai-org/GLM-5.3-Flash}"
LOCAL_NAME="${LOCAL_NAME:-${MODEL_ID##*/}}"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-0}"
EXPECTED_GB="${EXPECTED_GB:-328}"
NEED_FREE_GB="${NEED_FREE_GB:-420}"
SHARD_GLOB="*.safetensors"
TARGET_DIR="$MODEL_DIR/$LOCAL_NAME"

echo "== 目标：$MODEL_ID，下载到 $TARGET_DIR =="
echo "== 1/4 检查磁盘空间（至少需要 ${NEED_FREE_GB} GB 空闲） =="
mkdir -p "$MODEL_DIR"
FREE_GB="$(df -BG --output=avail "$MODEL_DIR" | tail -1 | tr -dc '0-9')"
echo "可用磁盘：${FREE_GB} GB"
if [ "${FREE_GB:-0}" -lt "$NEED_FREE_GB" ]; then
    echo "[错误] 磁盘空间不足（需要 ${NEED_FREE_GB} GB，当前仅 ${FREE_GB} GB）。请指定大容量数据盘：MODEL_DIR=/data/models bash download-glm.sh"
    exit 1
fi

expected_shards_from_index() {
    python3 - "$TARGET_DIR/model.safetensors.index.json" <<'PYEOF'
import json, sys
idx = json.load(open(sys.argv[1]))
wm = idx.get("weight_map", {})
print(len(set(wm.values())))
PYEOF
}

weights_complete() {
    [ -f "$TARGET_DIR/config.json" ] || return 1
    [ -f "$TARGET_DIR/model.safetensors.index.json" ] || return 1
    local want="$EXPECTED_SHARDS"
    if [ "$want" -le 0 ]; then
        want="$(expected_shards_from_index 2>/dev/null || echo 0)"
    fi
    [ "$want" -gt 0 ] || return 1
    local n
    n="$(find "$TARGET_DIR" -maxdepth 1 -name "$SHARD_GLOB" | wc -l)"
    [ "$n" -eq "$want" ]
}

if weights_complete; then
    echo "[提示] $TARGET_DIR 已有完整权重（与 index.json 分片数一致），跳过下载，直接校验。"
else
    echo "== 2/4 下载（ModelScope 优先，支持断点续下） =="
    mkdir -p "$TARGET_DIR"
    if command -v modelscope >/dev/null 2>&1 && modelscope download "$MODEL_ID" --local-dir "$TARGET_DIR"; then
        echo "[完成] ModelScope 下载成功。"
    else
        echo "[警告] ModelScope 下载失败或未安装，改走 Hugging Face（hf-mirror 镜像，保留已下分片续下）。"
        HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
            hf download "$MODEL_ID" --local-dir "$TARGET_DIR"
        echo "[完成] Hugging Face 下载成功。"
    fi
fi

echo "== 3/4 校验权重完整性 =="
FAIL=0
for f in config.json model.safetensors.index.json; do
    if [ ! -f "$TARGET_DIR/$f" ]; then
        echo "[错误] 缺少关键文件：$f"
        FAIL=1
    fi
done
WANT_SHARDS="$EXPECTED_SHARDS"
if [ "$WANT_SHARDS" -le 0 ] && [ -f "$TARGET_DIR/model.safetensors.index.json" ]; then
    WANT_SHARDS="$(expected_shards_from_index 2>/dev/null || echo 0)"
fi
SHARD_COUNT="$(find "$TARGET_DIR" -maxdepth 1 -name "$SHARD_GLOB" | wc -l)"
echo "分片数量：$SHARD_COUNT（预期 $WANT_SHARDS，按 $SHARD_GLOB 计数）"
if [ "$WANT_SHARDS" -gt 0 ] && [ "$SHARD_COUNT" -ne "$WANT_SHARDS" ]; then
    echo "[错误] 分片数量不对，下载可能不完整。请重新运行本脚本续下（不会删已下分片）。"
    FAIL=1
elif [ "$SHARD_COUNT" -eq 0 ]; then
    echo "[错误] 未找到任何 safetensors 分片。"
    FAIL=1
fi
ACTUAL_GB="$(du -sBG "$TARGET_DIR" | cut -f1 | tr -dc '0-9')"
echo "总大小：约 ${ACTUAL_GB} GB（预期约 ${EXPECTED_GB} GB）"
LOW=$((EXPECTED_GB * 85 / 100))
HIGH=$((EXPECTED_GB * 115 / 100))
if [ "${ACTUAL_GB:-0}" -lt "$LOW" ] || [ "${ACTUAL_GB:-0}" -gt "$HIGH" ]; then
    echo "[错误] 总大小与预期差太多（${LOW}~${HIGH} GB 才算正常），可能下载不完整。"
    FAIL=1
fi
if [ "$FAIL" -ne 0 ]; then
    exit 1
fi

echo "== 4/4 校验 config 关键字段 =="
python3 - "$TARGET_DIR/config.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
t = d.get("text_config", d)
print("模型结构：", d.get("architectures"))
print("模型类型：", d.get("model_type"), "/", t.get("model_type"))
print("原生上下文 max_position_embeddings：", t.get("max_position_embeddings"))
print("层数 / 路由专家数：", t.get("num_hidden_layers"), "/", t.get("n_routed_experts"))
qc = d.get("quantization_config") or t.get("quantization_config")
if qc:
    print("量化配置：", {k: qc.get(k) for k in ("quant_method", "fmt", "activation_scheme", "weight_block_size") if k in qc})
PYEOF

echo
echo "[完成] 权重就绪：$TARGET_DIR"
echo "接下来运行：bash serve-glm.sh"
