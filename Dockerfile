# Dockerfile — 算力优化黑客松标准化推理镜像（草稿，待 Blackwell 真机构建验证）
#
# 基础镜像选型：
#   nvidia/cuda:13.0.0-devel-ubuntu24.04（含 nvcc 13.0，对齐 pip install vllm==0.34.0
#   拉取的 torch 2.14.0+cu130，解决此前 CUDA 12.8 devel 镜像与 cu130 错位导致
#   DeepGEMM 被禁、Blackwell JIT 存在错位风险的问题）。
#
# 构建（镜像还从来没构建过，构建地点待定，见 issue #14）：
#   docker build --platform linux/amd64 -t hackathon-vllm:0.34.0 .
#   # 国内构建加速可传 PyPI 镜像：
#   docker build --platform linux/amd64 --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -t hackathon-vllm:0.34.0 .
#
# 运行（挂载 NFS 只读权重目录到 /models，缓存目录挂到本地 NVMe）：
#   # 1. 启动 Qwen3.8-27B-FP8（单卡或 DP=8）
#   docker run --gpus all --ipc=host --net=host \
#     --device /dev/infiniband --ulimit memlock=-1 \
#     -v /data/models:/models:ro \
#     -v /data/nvme1n1/cache:/cache \
#     -e MODEL_DIR=/models \
#     -e NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7 \
#     -e NCCL_IB_GID_INDEX=3 -e NCCL_SOCKET_IFNAME=ib0 \
#     hackathon-vllm:0.34.0 bash serve-qwen27b.sh
#
#   # 2. 启动 DeepSeek-V4.1-Flash-1017（8 卡 TP）
#   docker run --gpus all --ipc=host --net=host \
#     --device /dev/infiniband --ulimit memlock=-1 \
#     -v /data/models:/models:ro \
#     -v /data/nvme1n1/cache:/cache \
#     -e MODEL_DIR=/models \
#     -e NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7 \
#     -e NCCL_IB_GID_INDEX=3 -e NCCL_SOCKET_IFNAME=ib0 \
#     hackathon-vllm:0.34.0 bash serve-deepseek.sh
#
#   上面 NCCL_IB_HCA 的 8 个口是占位写法：上机后用 `ibv_devinfo` / `rdma link`
#   筛出真正 ACTIVE 的 8 个口再填（见 DAY1-RUNBOOK.md 10:30–11:00 那步）。

ARG BASE_IMAGE=nvidia/cuda:13.0.0-devel-ubuntu24.04
# 明确构建平台：Dockerfile 本身保证不了 amd64，ARM 机器上直接 build 会出 arm64 镜像
FROM --platform=linux/amd64 ${BASE_IMAGE}

ARG VLLM_VERSION=0.34.0
ARG PIP_INDEX_URL=""

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/models \
    VLLM_CACHE_ROOT=/cache/vllm \
    VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/cache/vllm/flashinfer_autotune_cache \
    VLLM_TUNED_CONFIG_FOLDER=/cache/vllm/tuned_configs

RUN mkdir -p /cache/vllm/flashinfer_autotune_cache /cache/vllm/tuned_configs

# Ubuntu 24.04 自带 Python 3.12；rdma-core / libibverbs1 / ibverbs-utils /
# iproute2 是跨机 PD/CP 与 NCCL IB 用的（每台只起本机副本时用不上，装上成本低）
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-dev \
        python3-pip \
        curl \
        git \
        ca-certificates \
        rdma-core \
        libibverbs1 \
        ibverbs-utils \
        iproute2 \
    && rm -rf /var/lib/apt/lists/*

# 使用独立 venv，避免 --break-system-packages 污染系统 Python
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# 注意：绝不能把 flashinfer 的回退接在同一条 && 链上写 `... || pip install ...`。
# shell 里 && 和 || 同优先级、从左往右结合，前面任何一步失败（包括 vLLM 没装上）
# 都会去执行 pip install flashinfer-python，它成功的话整条 RUN 退出码就是 0，
# 于是得到一个「构建成功、但里面没有 vLLM」的镜像（issue #14 P0，已验证）。
# 所以 vllm 安装与 flashinfer 回退拆成两个 RUN；flashinfer 锁 0.7.3
#（vLLM 0.34.0 要求，见 #1 的调研）；最后再加一行构建期自检。
RUN if [ -n "${PIP_INDEX_URL}" ]; then \
        pip config set global.index-url "${PIP_INDEX_URL}"; \
    fi \
    && pip install --upgrade pip \
    && pip install "vllm==${VLLM_VERSION}" modelscope huggingface_hub
RUN { python -c "import flashinfer" 2>/dev/null || pip install "flashinfer-python==0.7.3"; }
RUN python -c "import vllm, flashinfer; print('vllm', vllm.__version__)"

WORKDIR /app
COPY serve-qwen27b.sh serve-deepseek.sh serve-glm.sh download-model.sh download-deepseek.sh download-glm.sh smoke-test.sh ./
COPY router/ ./router/
COPY scripts/ ./scripts/
RUN chmod +x serve-qwen27b.sh serve-deepseek.sh serve-glm.sh download-model.sh download-deepseek.sh download-glm.sh smoke-test.sh router/prefix_router.py scripts/tune_moe_sm120.sh

EXPOSE 8000

# 默认入口为 Qwen，启动 DeepSeek 时覆盖 CMD 为 `bash serve-deepseek.sh`
CMD ["bash", "serve-qwen27b.sh"]
