#!/usr/bin/env bash
# setup-vllm.sh — 在全新 AutoDL Ubuntu + CUDA 实例上安装 vLLM 推理环境
#
# 用法：
#   bash setup-vllm.sh            # 使用默认版本安装
#   VLLM_VERSION=0.34.0 bash setup-vllm.sh
#
# 做什么：
#   1. 检查 GPU（nvidia-smi）
#   2. 检查 Python 版本（>= 3.11）
#   3. 检查磁盘空间
#   4. 创建 Python 虚拟环境 .venv，安装 vLLM + modelscope + huggingface_hub
#   5. 安装 flashinfer 与 CUDA toolkit（nvcc，FlashInfer 后端 JIT 编译要用）
#   6. 校验 vllm 可正常 import、flashinfer 可用、nvcc 可用
#
# 注意：
#   - vLLM 0.34.0 会把 torch 换成它要求的版本（torch 2.14.0），
#     首次安装要下载约 3 GB，请耐心等待，不要中途 Ctrl-C。
#   - 第 5 步装 flashinfer + nvcc：第 6 轮验证 FlashInfer 是最快后端
#     （冷 41.09 秒），但它的 batch-prefill 内核要在 warmup 时 JIT 编译，
#     没 nvcc 引擎起不来（第 5 轮教训；Modal 是用 CUDA devel 基础镜像
#     解决的）。nvcc 没装上也不报错——FA3 后端不用 nvcc，serve 时设
#     ATTENTION_BACKEND="" 即可回落（第 5 轮默认后端，冷 52.52 秒）。
#   - 全程不联网下载模型权重，权重下载见 download-model.sh。

set -euo pipefail

VLLM_VERSION="${VLLM_VERSION:-0.34.0}"
VENV_DIR="${VENV_DIR:-.venv}"
# 安装约需：vLLM + torch 约 8 GB 磁盘
MIN_DISK_GB="${MIN_DISK_GB:-15}"

echo "== 1/6 检查 GPU =="
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

echo "== 2/6 检查 Python =="
if ! command -v python3 >/dev/null 2>&1; then
    echo "[错误] 找不到 python3，请先装 Python 3.11+。"
    exit 1
fi
PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "Python 版本：$PY_VER"
python3 -c 'import sys; assert sys.version_info >= (3, 11), "vLLM 要求 Python >= 3.11"'

echo "== 3/6 检查磁盘空间 =="
FREE_GB="$(df -BG --output=avail . | tail -1 | tr -dc '0-9')"
echo "当前目录可用磁盘：${FREE_GB} GB（至少需要 ${MIN_DISK_GB} GB）"
if [ "${FREE_GB:-0}" -lt "$MIN_DISK_GB" ]; then
    echo "[错误] 磁盘空间不足。请在 AutoDL 控制台给实例扩数据盘，或换更大磁盘的机型。"
    exit 1
fi

echo "== 4/6 创建虚拟环境并安装 vLLM ${VLLM_VERSION} =="
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
# 国内机器走 PyPI 官方源通常也可达；如下载极慢，可改用清华源：
#   pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install "vllm==${VLLM_VERSION}" modelscope huggingface_hub

echo "== 5/6 安装 flashinfer 与 CUDA toolkit（nvcc）=="
# flashinfer：FlashInfer 注意力后端（第 6 轮最快，冷 41.09 秒）要它；
# vLLM 0.34.0 安装时通常会连带装 flashinfer-python，这里只做显式核对，
# 缺了再补装。
if python -c "import flashinfer" 2>/dev/null; then
    python -c "import flashinfer; print('flashinfer 已装：', getattr(flashinfer, '__version__', '未知版本'))"
else
    echo "没找到 flashinfer，补装 flashinfer-python ..."
    pip install flashinfer-python
    python -c "import flashinfer; print('flashinfer 已装：', getattr(flashinfer, '__version__', '未知版本'))"
fi
# nvcc：FlashInfer batch-prefill 要在 warmup 时 JIT 编译 CUDA 内核，没 nvcc
# 引擎起不来（第 5 轮教训；Modal 是用 CUDA devel 基础镜像解决的）。
# 裸机上走 apt 装 cuda-nvcc-12-8（与 vLLM 0.34.0 pip 轮子的 CUDA 12.8
# 匹配；第 6 轮 torch 实际是 cu130，nvcc 12.8 与之错位但 JIT 仍通，
# DeepGEMM 被禁——见 ENVIRONMENT.md §4）。前提：需要 sudo，且 NVIDIA
# CUDA apt 源已配好；源没配好时只警告、不报错——FA3 后端不用 nvcc，
# serve 时设 ATTENTION_BACKEND="" 即可回落。
if command -v nvcc >/dev/null 2>&1; then
    echo "nvcc 已装："
    nvcc --version | tail -1
else
    echo "[提示] 没找到 nvcc，尝试 apt 安装 cuda-nvcc-12-8 ..."
    if ls /etc/apt/sources.list.d/ 2>/dev/null | grep -qi cuda; then
        if command -v sudo >/dev/null 2>&1; then
            sudo apt-get update && sudo apt-get install -y cuda-nvcc-12-8 || echo "[警告] nvcc 安装失败，见上（不报错，继续）。"
        else
            apt-get update && apt-get install -y cuda-nvcc-12-8 || echo "[警告] nvcc 安装失败，见上（不报错，继续）。"
        fi
    else
        echo "[警告] 没配 NVIDIA CUDA apt 源，跳过 nvcc 安装（不报错）。"
        echo "       后果：FlashInfer 后端用不了，只能回落 FA3（serve 时设 ATTENTION_BACKEND=\"\"）。"
        echo "       手动装法：按 https://developer.nvidia.com/cuda-downloads 配好 CUDA apt 源后，"
        echo "       再跑：sudo apt-get install -y cuda-nvcc-12-8"
        echo "       然后用 nvcc --version 校验。"
    fi
    if command -v nvcc >/dev/null 2>&1; then
        echo "nvcc 已装："
        nvcc --version | tail -1
    else
        echo "[警告] nvcc 仍不可用，继续安装其余部分（见上）。FlashInfer 后端暂不可用。"
    fi
fi

echo "== 6/6 校验安装 =="
python -c "import vllm; print('vllm 版本：', vllm.__version__)"
python -c "import torch; print('torch 版本：', torch.__version__, '| CUDA 可用：', torch.cuda.is_available())"
python -c "import modelscope; print('modelscope 版本：', modelscope.__version__)"
python -c "import huggingface_hub; print('huggingface_hub 版本：', huggingface_hub.__version__)"
python -c "import flashinfer; print('flashinfer 可用：', getattr(flashinfer, '__version__', '未知版本'))"
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version | tail -1
else
    echo "[提示] nvcc 不可用：FlashInfer 后端暂不可用，FA3 可用（serve 时设 ATTENTION_BACKEND=\"\"）。"
fi
hf --version || echo "[警告] hf 命令不可用（不影响 vLLM 运行，ModelScope 仍可下载权重）。"

echo
echo "[完成] 环境已就绪。接下来运行："
echo "  source $VENV_DIR/bin/activate"
echo "  bash download-model.sh    # 下载 Qwen3.8-27B-FP8 权重（约 31 GB）"
