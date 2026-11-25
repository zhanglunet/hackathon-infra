#!/usr/bin/env bash
# distribute-image.sh — 把预打包的推理镜像分发到 4 台比赛机（issue #14 ②-5）
#
# 用法：
#   bash scripts/distribute-image.sh \
#       --image hackathon-vllm:0.34.0 \
#       --nodes "node-user@10.x.x.11,node-user@10.x.x.12,node-user@10.x.x.13,node-user@10.x.x.14"
#
#   节点地址比赛当天才发（私发给组长），也可以用环境变量传：
#     R2_NODES="node-user@10.x.x.11,..." IMAGE=hackathon-vllm:0.34.0 bash scripts/distribute-image.sh
#
# 流程（本机执行）：docker save | zstd 打包 → 记下包大小 → scp 到每台机 →
# 每台机上 docker load → 每台机上 docker images 核对镜像在位。
# 要求：本机与各节点都能 ssh 免密登录（比赛机账号私发给组长时一并确认）；
# 各节点上 docker 可用（普通用户有 docker 权限，见组织方 2026-11-24 答复）。
#
# 注意：镜像包十几 GB，scp 走内网；包大小会打印出来，交卷时填进「环境与优化说明表」。
set -euo pipefail

IMAGE="${IMAGE:-hackathon-vllm:0.34.0}"
NODES="${R2_NODES:-}"
PKG_DIR="${PKG_DIR:-./image-packages}"
DRY_RUN="${DRY_RUN:-0}"

while [ $# -gt 0 ]; do
    case "$1" in
        --image) IMAGE="$2"; shift 2 ;;
        --nodes) NODES="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "[错误] 未知参数：$1（用法：--image <tag> --nodes <user@host,...>）" >&2; exit 2 ;;
    esac
done

[ -n "$NODES" ] || { echo "[错误] 没给节点：传 --nodes 或设 R2_NODES（4 台比赛机地址，逗号分隔）。" >&2; exit 2; }
command -v docker >/dev/null 2>&1 || { echo "[错误] 本机找不到 docker。" >&2; exit 2; }
command -v zstd >/dev/null 2>&1 || { echo "[错误] 本机找不到 zstd（apt install zstd）。" >&2; exit 2; }

SAFE_TAG="$(echo "$IMAGE" | tr '/:' '__')"
PKG="$PKG_DIR/$SAFE_TAG.tar.zst"
mkdir -p "$PKG_DIR"

echo "== 打包 $IMAGE → $PKG =="
if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] docker save \"$IMAGE\" | zstd -T0 -19 -o \"$PKG\""
else
    docker save "$IMAGE" | zstd -T0 -19 -o "$PKG"
fi
PKG_SIZE="$(du -h "$PKG" | cut -f1)"
PKG_BYTES="$(stat -c%s "$PKG")"
echo "镜像包：$PKG（$PKG_SIZE，$PKG_BYTES 字节）——交卷时把这个数填进环境与优化说明表"

IFS=',' read -ra NODE_LIST <<< "$NODES"
for node in "${NODE_LIST[@]}"; do
    echo "== 分发 → $node =="
    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] scp $PKG $node:/tmp/ && ssh $node 'zstd -dc /tmp/$(basename "$PKG") | docker load && docker images $IMAGE'"
        continue
    fi
    scp "$PKG" "$node:/tmp/"
    ssh "$node" "zstd -dc /tmp/$(basename "$PKG") | docker load && docker images '$IMAGE'"
done

echo "== 完成：$IMAGE 已分发到 ${#NODE_LIST[@]} 台机，包 $PKG（$PKG_SIZE）=="
