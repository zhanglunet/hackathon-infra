#!/usr/bin/env bash
# download-deepseek.sh — 下载 DeepSeek-V4.1-Flash-1017 权重（ModelScope 优先，HF 备用）
#
# 用法：
#   bash download-deepseek.sh                          # 默认下 1017 版到 ./models/DeepSeek-V4.1-Flash-1017
#   MODEL_DIR=/data/models bash download-deepseek.sh   # 挂数据盘（推荐，需 ≥ 650 GB 空闲）
#   nohup bash download-deepseek.sh > logs/download-deepseek.log 2>&1 &
#
# 权重规格（2026-11-23 核对自 vLLM 官方 recipe：recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash）：
#   - 仓库：deepseek-ai/DeepSeek-V4.1-Flash-1017（默认；0909 首发版为 deepseek-ai/DeepSeek-V4.1-Flash）
#   - 体积：约 511 GB（476 GiB），48 个 safetensors 分片
#   - 结构：552B MoE 骨干 + 196B Engram 记忆表（约 183 GiB）+ ~14B DSpark 草稿头
#   - 量化：路由专家 MXFP4，其余 MXFP8 block 量化（UE8M0 scale），Embedding/LM head BF16
#   - 注意：官方仓库不含 Jinja chat template，启动时必须加 --tokenizer-mode deepseek_v41
#
# 环境变量（可按组织方实际指定权重名覆盖）：
#   MODEL_DIR        权重根目录，默认 ./models
#   MODEL_ID         ModelScope/HF 仓库名，默认 deepseek-ai/DeepSeek-V4.1-Flash-1017
#   LOCAL_NAME       本地子目录名，默认取 MODEL_ID 最后一段
#   EXPECTED_SHARDS  预期 safetensors 分片数，默认 48（设为 0 则只校验 >0 不强卡具体数）
#   EXPECTED_GB      预期体积（GB），默认 511
#   NEED_FREE_GB     下载前要求空闲磁盘（GB），默认 650

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-./models}"
MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-V4.1-Flash-1017}"
LOCAL_NAME="${LOCAL_NAME:-${MODEL_ID##*/}}"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-48}"
EXPECTED_GB="${EXPECTED_GB:-511}"
NEED_FREE_GB="${NEED_FREE_GB:-650}"
SHARD_GLOB="*.safetensors"
TARGET_DIR="$MODEL_DIR/$LOCAL_NAME"

echo "== 目标：$MODEL_ID，下载到 $TARGET_DIR =="
echo "== 1/4 检查磁盘空间（至少需要 ${NEED_FREE_GB} GB 空闲） =="
mkdir -p "$MODEL_DIR"
FREE_GB="$(df -BG --output=avail "$MODEL_DIR" | tail -1 | tr -dc '0-9')"
echo "可用磁盘：${FREE_GB} GB"
if [ "${FREE_GB:-0}" -lt "$NEED_FREE_GB" ]; then
    echo "[错误] 磁盘空间不足（需要 ${NEED_FREE_GB} GB，当前仅 ${FREE_GB} GB）。请指定大容量数据盘：MODEL_DIR=/data/models bash download-deepseek.sh"
    exit 1
fi

weights_complete() {
    [ -f "$TARGET_DIR/config.json" ] || return 1
    [ -f "$TARGET_DIR/model.safetensors.index.json" ] || return 1
    local n
    n="$(find "$TARGET_DIR" -maxdepth 1 -name "$SHARD_GLOB" | wc -l)"
    if [ "$EXPECTED_SHARDS" -gt 0 ]; then
        [ "$n" -eq "$EXPECTED_SHARDS" ]
    else
        [ "$n" -gt 0 ]
    fi
}

if weights_complete; then
    echo "[提示] $TARGET_DIR 已有完整权重（分片数匹配），跳过下载，直接校验。"
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
SHARD_COUNT="$(find "$TARGET_DIR" -maxdepth 1 -name "$SHARD_GLOB" | wc -l)"
echo "分片数量：$SHARD_COUNT（预期 $EXPECTED_SHARDS，按 $SHARD_GLOB 计数）"
if [ "$EXPECTED_SHARDS" -gt 0 ] && [ "$SHARD_COUNT" -ne "$EXPECTED_SHARDS" ]; then
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
print("层数：", t.get("num_hidden_layers"))
qc = d.get("quantization_config") or t.get("quantization_config")
if qc:
    print("量化配置：", {k: qc.get(k) for k in ("quant_method", "fmt", "activation_scheme", "weight_block_size") if k in qc})
PYEOF

echo
echo "[完成] 权重就绪：$TARGET_DIR"
echo "接下来运行：bash serve-deepseek.sh"
