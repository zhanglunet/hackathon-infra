#!/usr/bin/env bash
# download-model.sh — 下载 Qwen3.8-27B 权重（ModelScope 优先，HF 备用）
#
# 用法：
#   bash download-model.sh            # 默认下载官方 FP8 版（约 31 GB）
#   bash download-model.sh --bf16     # 下载 BF16 原版（约 55 GB）
#   MODEL_DIR=/data/models bash download-model.sh
#
# 流程：
#   1. 检查磁盘空间
#   2. 先试 ModelScope（modelscope.cn，国内直连快）
#   3. 失败则回退到 Hugging Face（走 hf-mirror.com 国内镜像）
#   4. 校验：config.json 存在、分片数量对得上、总大小在预期范围内
#
# 权重说明（2026-11-20 实测，2026-11-21/23 HF API 与源码复核）：
#   - 官方 FP8 版 Qwen/Qwen3.8-27B-FP8：约 30.9 GB，66 个 safetensors 分片
#     （按层拆分为 layers-0~63 + mtp + outside，非 model-*.safetensors），
#     config.json 里 quant_method=fp8、fmt=e4m3、per-tensor 权重量化、
#     activation_scheme=dynamic（无 weight_block_size），
#     官方称精度与 BF16 原版“几乎相同”。比赛精度门是相对官方权重损失 ≤2%，
#     用官方 FP8 权重本身不算“再量化”，自测时仍要跑 EvalScope 对照。
#   - BF16 原版 Qwen/Qwen3.8-27B：约 51.7 GiB（约 55.5 GB），18 个分片。
#     只有在 FP8 版出问题、或想做精度对照时才下 BF16。

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-./models}"
QUANT="${QUANT:-fp8}"   # fp8 | bf16

if [ "${1:-}" = "--bf16" ]; then QUANT="bf16"; fi
if [ "${1:-}" = "--fp8" ]; then QUANT="fp8"; fi

if [ "$QUANT" = "fp8" ]; then
    MODEL_ID="Qwen/Qwen3.8-27B-FP8"
    LOCAL_NAME="Qwen3.8-27B-FP8"
    EXPECTED_GB=31        # 官方 FP8 权重约 30.9 GB
    EXPECTED_SHARDS=66
    SHARD_GLOB="*.safetensors"
    NEED_FREE_GB=45       # 下载中需要一点余量
else
    MODEL_ID="Qwen/Qwen3.8-27B"
    LOCAL_NAME="Qwen3.8-27B"
    EXPECTED_GB=56        # BF16 约 51.7 GiB ≈ 55.5 GB
    EXPECTED_SHARDS=18
    SHARD_GLOB="model-*.safetensors"
    NEED_FREE_GB=75
fi
TARGET_DIR="$MODEL_DIR/$LOCAL_NAME"

echo "== 目标：$MODEL_ID（$QUANT），下载到 $TARGET_DIR =="
echo "== 1/4 检查磁盘空间（至少需要 ${NEED_FREE_GB} GB 空闲） =="
# 先建目录再 df：全新环境没有 models/ 时，df 不存在的路径会失败，
# 在 set -e + pipefail 下直接让脚本在创建目录前退出（#5 评审发现）。
# 之前兜底 df 的是当前目录 .，和目标盘可能不是同一文件系统，数字会错。
mkdir -p "$MODEL_DIR"
FREE_GB="$(df -BG --output=avail "$MODEL_DIR" | tail -1 | tr -dc '0-9')"
echo "可用磁盘：${FREE_GB} GB"
if [ "${FREE_GB:-0}" -lt "$NEED_FREE_GB" ]; then
    echo "[错误] 磁盘空间不足。AutoDL 租机时请挂数据盘，或先清理空间。"
    exit 1
fi

# 只有分片数对得上才算“已有完整权重”：只看 index.json 会在中断续下时
# 误判——半成品也可能已有 index.json，跳过下载后校验必失败，还得整删重下。
weights_complete() {
    [ -f "$TARGET_DIR/model.safetensors.index.json" ] || return 1
    local n
    n="$(find "$TARGET_DIR" -maxdepth 1 -name "$SHARD_GLOB" | wc -l)"
    [ "$n" -eq "$EXPECTED_SHARDS" ]
}

if weights_complete; then
    echo "[提示] $TARGET_DIR 已有完整权重（$EXPECTED_SHARDS 个分片），跳过下载，直接校验。"
else
    echo "== 2/4 下载（ModelScope 优先） =="
    mkdir -p "$TARGET_DIR"
    # modelscope CLI（1.40.3 实测）：仓库 ID 是位置参数，目录用 --local-dir（带连字符）
    #   modelscope download <owner/name> --local-dir <目录>
    if modelscope download "$MODEL_ID" --local-dir "$TARGET_DIR"; then
        echo "[完成] ModelScope 下载成功。"
    else
        echo "[警告] ModelScope 下载失败，改走 Hugging Face（hf-mirror 镜像）。"
        # 不删 TARGET_DIR：半成品留在原地，hf download 会跳过已下完的分片续下；
        # 删掉等于逼用户重下几十 GB（#5 评审发现）。
        # hf CLI（huggingface_hub 2.x）：老名字 huggingface-cli 已被 hf 取代
        #   hf download <owner/name> --local-dir <目录>
        HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
            hf download "$MODEL_ID" --local-dir "$TARGET_DIR"
        echo "[完成] Hugging Face 下载成功。"
    fi
fi

echo "== 3/4 校验权重完整性 =="
FAIL=0
for f in config.json model.safetensors.index.json tokenizer.json; do
    if [ ! -f "$TARGET_DIR/$f" ]; then
        echo "[错误] 缺少关键文件：$f"
        FAIL=1
    fi
done
SHARD_COUNT="$(find "$TARGET_DIR" -maxdepth 1 -name "$SHARD_GLOB" | wc -l)"
echo "分片数量：$SHARD_COUNT（预期 $EXPECTED_SHARDS，按 $SHARD_GLOB 计数）"
if [ "$SHARD_COUNT" -ne "$EXPECTED_SHARDS" ]; then
    echo "[错误] 分片数量不对，下载可能不完整。请删掉 $TARGET_DIR 后重新下载。"
    FAIL=1
fi
ACTUAL_GB="$(du -sBG "$TARGET_DIR" | cut -f1 | tr -dc '0-9')"
echo "总大小：约 ${ACTUAL_GB} GB（预期约 ${EXPECTED_GB} GB）"
# 允许 ±15% 误差（不同统计口径）
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
print("原生上下文 max_position_embeddings：", t.get("max_position_embeddings"))
print("层数 / 全注意力层：", t.get("num_hidden_layers"), "/", sum(1 for x in t.get("layer_types", []) if x == "full_attention"))
qc = d.get("quantization_config")
if qc:
    # 只打印关键字段：modules_to_not_convert 有几百项，全打印会刷屏
    print("量化配置：", {k: qc.get(k) for k in ("quant_method", "fmt", "activation_scheme", "weight_block_size")})
else:
    print("量化配置：无（BF16 原版）")
PYEOF

echo
echo "[完成] 权重就绪：$TARGET_DIR"
echo "接下来运行：bash serve-qwen27b.sh"
