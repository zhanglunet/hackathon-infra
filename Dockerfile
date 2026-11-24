# Dockerfile — 算力优化黑客松标准化推理镜像（草稿，待 Blackwell 真机构建验证）
#
# 基础镜像选型：
#   nvidia/cuda:13.0.0-devel-ubuntu24.04（含 nvcc 13.0，对齐 pip install vllm==0.34.0
#   拉取的 torch 2.14.0+cu130，解决此前 CUDA 12.8 devel 镜像与 cu130 错位导致
#   DeepGEMM 被禁、Blackwell JIT 存在错位风险的问题）。
#
# 构建：
#   docker build -t hackathon-vllm:0.34.0 .
#   # 国内构建加速可传 PyPI 镜像：
#   docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -t hackathon-vllm:0.34.0 .
#
# 运行（挂载本地权重目录到 /models）：
#   # 1. 启动 Qwen3.8-27B-FP8（单卡或 DP=8）
#   docker run --gpus all --ipc=host --net=host \
#     -v /data/models:/models \
#     -e MODEL_DIR=/models \
#     hackathon-vllm:0.34.0 bash serve-qwen27b.sh
#
#   # 2. 启动 DeepSeek-V4.1-Flash-1017（8 卡 TP）
#   docker run --gpus all --ipc=host --net=host \
#     -v /data/models:/models \
#     -e MODEL_DIR=/models \
#     hackathon-vllm:0.34.0 bash serve-deepseek.sh

ARG BASE_IMAGE=nvidia/cuda:13.0.0-devel-ubuntu24.04
FROM ${BASE_IMAGE}

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

# Ubuntu 24.04 自带 Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-dev \
        python3-pip \
        curl \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 使用独立 venv，避免 --break-system-packages 污染系统 Python
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN if [ -n "${PIP_INDEX_URL}" ]; then \
        pip config set global.index-url "${PIP_INDEX_URL}"; \
    fi \
    && pip install --upgrade pip \
    && pip install "vllm==${VLLM_VERSION}" modelscope huggingface_hub \
    && python -c "import flashinfer" 2>/dev/null || pip install flashinfer-python

WORKDIR /app
COPY serve-qwen27b.sh serve-deepseek.sh download-model.sh download-deepseek.sh smoke-test.sh ./
RUN chmod +x serve-qwen27b.sh serve-deepseek.sh download-model.sh download-deepseek.sh smoke-test.sh

EXPOSE 8000

# 默认入口为 Qwen，启动 DeepSeek 时覆盖 CMD 为 `bash serve-deepseek.sh`
CMD ["bash", "serve-qwen27b.sh"]
