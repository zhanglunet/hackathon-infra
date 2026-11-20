#!/usr/bin/env bash
# setup-vllm.sh — 在全新 AutoDL Ubuntu + CUDA 实例上安装 vLLM 推理环境
#
# 用法：
#   bash setup-vllm.sh            # 使用默认版本安装
#   VLLM_VERSION=0.34.0 bash setup-vllm.sh
#
# 做什么：
#   1. 检查 GPU（nvidia-smi）、Python 版本、磁盘空间
#   2. 创建 Python 虚拟环境 .venv
#   3. 安装 vLLM（已验证版本，见下方）+ modelscope + huggingface_hub
#   4. 校验 vllm 可正常 import
#
# 注意：
#   - vLLM 0.34.0 会把 torch 换成它要求的版本（torch 2.14.0），
#     首次安装要下载约 3 GB，请耐心等待，不要中途 Ctrl-C。
#   - 本脚本不装 flashinfer：A100 上 vLLM 默认后端即可；
#     到了比赛用的 RTX 6000D（Blackwell）上如需 flashinfer 再另装。
#   - 全程不联网下载模型权重，权重下载见 download-model.sh。

set -euo pipefail

VLLM_VERSION="${VLLM_VERSION:-0.34.0}"
VENV_DIR="${VENV_DIR:-.venv}"
# 安装约需：vLLM + torch 约 8 GB 磁盘
MIN_DISK_GB="${MIN_DISK_GB:-15}"

echo "== 1/5 检查 GPU =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[错误] 找不到 nvidia-smi：这台机器没有 NVIDIA 驱动或没有显卡。"
    echo "       AutoDL 租机时请选带 GPU 的镜像（PyTorch 镜像自带驱动）。"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
GPU_MEM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
if [ "${GPU_MEM_MIB:-0}" -lt 70000 ]; then
    echo "[警告] 显存只有 ${GPU_MEM_MIB} MiB，小于 70 GB。"
    echo "       Qwen3.8-27B FP8 单卡排练至少需要 1 张 80 GB 卡（A100/H100/RTX 6000D）。"
    echo "       仍继续安装，但 serve 时大概率会 OOM。"
fi

echo "== 2/5 检查 Python =="
if ! command -v python3 >/dev/null 2>&1; then
    echo "[错误] 找不到 python3，请先装 Python 3.11+。"
    exit 1
fi
PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "Python 版本：$PY_VER"
python3 -c 'import sys; assert sys.version_info >= (3, 11), "vLLM 要求 Python >= 3.11"'

echo "== 3/5 检查磁盘空间 =="
FREE_GB="$(df -BG --output=avail . | tail -1 | tr -dc '0-9')"
echo "当前目录可用磁盘：${FREE_GB} GB（至少需要 ${MIN_DISK_GB} GB）"
if [ "${FREE_GB:-0}" -lt "$MIN_DISK_GB" ]; then
    echo "[错误] 磁盘空间不足。请在 AutoDL 控制台给实例扩数据盘，或换更大磁盘的机型。"
    exit 1
fi

echo "== 4/5 创建虚拟环境并安装 vLLM ${VLLM_VERSION} =="
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
# 国内机器走 PyPI 官方源通常也可达；如下载极慢，可改用清华源：
#   pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install "vllm==${VLLM_VERSION}" modelscope huggingface_hub

echo "== 5/5 校验安装 =="
python -c "import vllm; print('vllm 版本：', vllm.__version__)"
python -c "import torch; print('torch 版本：', torch.__version__, '| CUDA 可用：', torch.cuda.is_available())"
python -c "import modelscope; print('modelscope 版本：', modelscope.__version__)"
python -c "import huggingface_hub; print('huggingface_hub 版本：', huggingface_hub.__version__)"
hf --version

echo
echo "[完成] 环境已就绪。接下来运行："
echo "  source $VENV_DIR/bin/activate"
echo "  bash download-model.sh    # 下载 Qwen3.8-27B-FP8 权重（约 31 GB）"
