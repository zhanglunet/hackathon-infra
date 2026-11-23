"""modal_vllm_qwen27b.py -- 在 Modal 上跑 Qwen3.8-27B 的 vLLM 服务

用法（先在 Modal 控制台取 token 做 modal setup，见 modal/README.md）：
  modal run modal/modal_vllm_qwen27b.py::download_weights   # 下权重进 Volume（约 31 GB，只需一次）
  modal serve modal/modal_vllm_qwen27b.py                    # 临时服务（开发调试，Ctrl-C 停）
  modal deploy modal/modal_vllm_qwen27b.py                   # 长期服务（固定 URL）

  MODAL_GPU=H100 modal deploy modal/modal_vllm_qwen27b.py   # FP8 冒烟切 H100
  MODAL_GPU=H200 MAX_MODEL_LEN=272000 modal deploy modal/modal_vllm_qwen27b.py  # 测 272K

  modal run modal/modal_vllm_qwen27b.py --url https://<部署地址>   # 冒烟

参数来源：serve-qwen27b.sh / download-model.sh（模型名、启动参数照抄，
不要自己编）。vLLM 0.34.0；served-model-name qwen3.8-27b；端口 8000。
服务暴露 OpenAI 兼容接口 /v1/chat/completions 和 /metrics。
"""

import json
import os
import subprocess

import modal

APP_NAME = "qwen27b-vllm"
VLLM_VERSION = "0.34.0"
VOLUME_NAME = "qwen27b-weights"
MODELS_DIR = "/models"
SERVED_NAME = "qwen3.8-27b"
NATIVE_LEN = 262144

# GPU 开关：默认 A100-80GB（最便宜，先做短上下文冒烟）；
# 测 272K 切 H200 141GB；FP8 冒烟切 H100 或 H200（见下面的算力检查）。
GPU = os.environ.get("MODAL_GPU", "A100-80GB")
QUANT = os.environ.get("QUANT", "fp8")  # fp8 | bf16
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", str(NATIVE_LEN)))
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "32"))
# chunked prefill 步长：vLLM 0.34.0 默认 8192。271K 提示会被切成约
# 34 片，片间有调度开销；调大它让单步吃更多 token（262144 时
# 271,500 token 的提示只切 2 片，基本不分片）。调大后单步激活显存
# 涨，vLLM 会自动少分 KV 块——单请求只占 KV 约 9%，够用。
MAX_NUM_BATCHED_TOKENS = int(os.environ.get("MAX_NUM_BATCHED_TOKENS", "8192"))
# 编译模式：空 = 默认（vLLM 0.34.0 实测跑 eager，mode=0，CUDA 图照常开）；
# 3 = VLLM_COMPILE（vLLM 定制的 Inductor 后端，分段编译 + 形状特化），
# 1 = 标准 torch.compile，2 = 单次 Dynamo 追踪。见 vLLM 官方文档
# Engine Arguments 的 --compilation-config 一节（-cc.mode=3 等价于
# -cc='{"mode":3}'）。2026-11-21 实测：默认 eager 下 272K 冷 TTFT
# 约 51 秒（约 15% MFU），编译是下一轮主攻方向。
COMPILATION_MODE = os.environ.get("COMPILATION_MODE", "")  # "" | "1" | "2" | "3"
# breakable CUDA 图开关：vLLM 0.34.0 对 Qwen3.5 架构默认自动开 breakable
# cudagraph，开启后会把 compilation_config.mode 置回 NONE（vllm/config/
# vllm.py:833），mode=3 因此静默失效（日志原文："Auto-enabling
# VLLM_USE_BREAKABLE_CUDAGRAPH=1. Set VLLM_USE_BREAKABLE_CUDAGRAPH=0 to
# opt out."）。设 "0" 可退出该路径再试 mode=3。默认 "1" 保持旧行为。
VLLM_USE_BREAKABLE_CUDAGRAPH = os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
GPU_MEM_UTIL = os.environ.get("GPU_MEM_UTIL", "0.90")
# GDN（线性注意力）prefill 后端：空 = 默认（auto→Hopper 上是 FlashInfer，
# JIT 编译内核）；"triton" = Triton/FLA 后端，跳过 JIT。2026-11-21 源码核实：
# vLLM 0.34.0 的 --gdn-prefill-backend（choices flashinfer/triton/cutedsl/
# aiter_flydsl/sycl/hip），默认 auto 在 H200（SM90）上选 flashinfer。
GDN_PREFILL_BACKEND = os.environ.get("GDN_PREFILL_BACKEND", "")  # "" | "triton" | ...
# 全注意力后端：空 = 默认（auto→Hopper 上是 FA3，profiling 轮实测
# _vllm_fa3_C::fwd 占冷请求 GPU 时间约 65%）；"FLASHINFER" / "TRITON_ATTN"
# 对照用。2026-11-22 源码核实（vllm 0.34.0）：AttentionBackendEnum 含两者；
# FlashInfer 要求 head_size ∈ {64,128,256,512}（本模型 256）、支持 fp8 KV、
# 算力 SM80–SM12.1（H200 是 SM90）；Triton 要求 head_size ≥ 32、支持 fp8 KV、
# 算力不限。显式指定时无效会直接报 ValueError（不静默回退），日志
# "Using %s backend." 可作切换证据。只影响 16 层全注意力，GDN 层走自己的后端。
ATTENTION_BACKEND = os.environ.get("ATTENTION_BACKEND", "")  # "" | "FLASHINFER" | "TRITON_ATTN"
# CUDA 镜像开关：第 5 轮 FlashInfer 引擎起不来，报错
# "Could not find nvcc"——debian_slim + pip 装 vllm 的镜像里没有 CUDA
# toolkit，FlashInfer batch-prefill 要在 warmup 时 JIT 编译 CUDA 内核。
# 设 "1" 时走 nvidia/cuda:12.8.0-devel-ubuntu24.04 官方基础镜像
# （含 nvcc 12.8，与 vllm 0.34.0 pip 轮子的 CUDA 12.8 匹配）；
# 默认 "0" 走 debian_slim，已验证的 FA3 路径一动不动。
USE_CUDA_IMAGE = os.environ.get("USE_CUDA_IMAGE", "0")  # "0" | "1"
# torch profiler 配置（JSON 字符串）：空 = 不开。开时写
# --profiler-config '{"profiler":"torch","torch_profiler_dir":"/models/profiler",...}'，
# trace 落进 Volume（/models 挂载点），随后 /start_profile、/stop_profile
# 控制采集，trace 用 `modal volume get` 取回本地分析。
PROFILER_CONFIG = os.environ.get("PROFILER_CONFIG", "")

# 工具调用与思考输出解析器（核对自 vLLM 官方 Qwen3.8-27B recipe）：
# 设为空字符串 "" 可显式关闭对应参数。
TOOL_CALL_PARSER = os.environ.get("TOOL_CALL_PARSER", "qwen3_coder")
REASONING_PARSER = os.environ.get("REASONING_PARSER", "qwen3")
# Modal 单容器最大并发请求数（比赛阶梯最高 80，默认放宽到 128，避免压测被 Modal 限流）。
MODAL_MAX_INPUTS = int(os.environ.get("MODAL_MAX_INPUTS", "128"))

MODELS = {
    # (HF 仓库, 本地目录名, 预期分片数, 分片匹配模式)
    # 注意 FP8 版的分片不叫 model-*.safetensors，而是按层拆的
    # layers-0~63（64 个）+ mtp + outside = 66 个（2026-11-21 用 HF API 核实）。
    "fp8": ("Qwen/Qwen3.8-27B-FP8", "Qwen3.8-27B-FP8", 66, "*.safetensors"),
    "bf16": ("Qwen/Qwen3.8-27B", "Qwen3.8-27B", 18, "model-*.safetensors"),
}
MODEL_ID, LOCAL_NAME, EXPECTED_SHARDS, SHARD_GLOB = MODELS[QUANT]
MODEL_PATH = f"{MODELS_DIR}/{LOCAL_NAME}"

# 容器里重导入本文件时，本地 shell 的环境变量已不存在；必须把配置经
# @app.function(env=...) 显式传进容器，容器里的模块级变量才能读到
# 同样的值。缺这一步时，H100 部署在容器里会回落成 A100 默认值，
# fail-fast 报错导致循环重启（2026-11-21 实测）。
_CONTAINER_ENV = {
    "MODAL_GPU": GPU,
    "QUANT": QUANT,
    "MAX_MODEL_LEN": str(MAX_MODEL_LEN),
    "MAX_NUM_SEQS": str(MAX_NUM_SEQS),
    "MAX_NUM_BATCHED_TOKENS": str(MAX_NUM_BATCHED_TOKENS),
    "COMPILATION_MODE": COMPILATION_MODE,
    "VLLM_USE_BREAKABLE_CUDAGRAPH": VLLM_USE_BREAKABLE_CUDAGRAPH,
    "GPU_MEM_UTIL": GPU_MEM_UTIL,
    "GDN_PREFILL_BACKEND": GDN_PREFILL_BACKEND,
    "ATTENTION_BACKEND": ATTENTION_BACKEND,
    "PROFILER_CONFIG": PROFILER_CONFIG,
    "TOOL_CALL_PARSER": TOOL_CALL_PARSER,
    "REASONING_PARSER": REASONING_PARSER,
    "MODAL_MAX_INPUTS": str(MODAL_MAX_INPUTS),
    # vLLM 0.34.0 会先校验 max_model_len ≤ config 的 max_position_embeddings，
    # 再应用 --hf-overrides；超原生上限必须另设此变量才放行（2026-11-21
    # 实测：不设它时 vLLM 直接报 ValueError，容器循环重启）。
    # YaRN 外推本身仍由 --hf-overrides 的 rope_parameters 负责。
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
}

app = modal.App(APP_NAME)

if USE_CUDA_IMAGE == "1":
    # FlashInfer 轮专用：官方 CUDA devel 镜像自带 nvcc 12.8（与 vllm
    # 0.34.0 pip 轮子的 CUDA 12.8 匹配），FlashInfer warmup 时 JIT 编译
    # 内核要用到它。add_python="3.12" 在该镜像里装 Python 3.12。
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.0-devel-ubuntu24.04", add_python="3.12"
        ).pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub")
    )
else:
    # 默认：debian_slim，已验证的 FA3 / BF16 路径都跑在这上面，别动。
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub")
    )

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _check_fp8_gpu(gpu: str, quant: str) -> None:
    """FP8 权重需要 Hopper（算力 9.0+）或 Blackwell，张量核才算得动。
    A100/A800 是 Ampere（8.0），跑不了 FP8——和 serve-qwen27b.sh 里
    的算力检查同一条规则。"""
    if quant == "fp8" and gpu.upper().startswith("A100"):
        raise ValueError(
            "A100 算力 8.0，不支持 FP8 计算，跑不了 FP8 权重。"
            "两个办法：① 切 H100/H200 再跑：MODAL_GPU=H100；"
            "② A100 上改下 BF16 版：QUANT=bf16（短上下文冒烟够用）。"
        )


@app.function(image=image, env=_CONTAINER_ENV, volumes={MODELS_DIR: volume}, timeout=3600)
def download_weights():
    """把权重下到 Modal Volume（约 31 GB FP8 / 约 55 GB BF16），只跑一次。
    不挂 GPU，只花 CPU 时间。配置经 env=_CONTAINER_ENV 传进容器，
    见上面注释。"""
    from pathlib import Path

    from huggingface_hub import snapshot_download

    print(f"下载 {MODEL_ID} 到 {MODEL_PATH} ...")
    snapshot_download(
        repo_id=MODEL_ID, local_dir=MODEL_PATH, local_dir_use_symlinks=False
    )
    target = Path(MODEL_PATH)
    for f in ("config.json", "model.safetensors.index.json", "tokenizer.json"):
        assert (target / f).exists(), f"缺少关键文件：{f}"
    shards = len(list(target.glob(SHARD_GLOB)))
    print(f"分片数量：{shards}（预期 {EXPECTED_SHARDS}，按 {SHARD_GLOB} 计数）")
    assert shards == EXPECTED_SHARDS, "分片数量不对，下载可能不完整，请重跑。"
    print(f"[完成] 权重就绪：{MODEL_PATH}")


@app.function(image=image, timeout=600)
def check_nvcc():
    """不挂 GPU，只验证容器里 nvcc 可用（第 6 轮 FlashInfer 前置检查）。
    用法：USE_CUDA_IMAGE=1 python3 -m modal run modal/modal_vllm_qwen27b.py::check_nvcc
    预期：nvcc --version 打印 12.8.x；torch.version.cuda 也是 12.8。"""
    import shutil

    for cmd in (
        ["which", "nvcc"],
        ["nvcc", "--version"],
        ["python3", "-c", "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"],
    ):
        print(f"\n=== $ {' '.join(cmd)} ===")
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        print(out.stdout, end="")
        if out.stderr:
            print("[stderr]", out.stderr, end="")
        print(f"[exit {out.returncode}]")
    assert shutil.which("nvcc"), "容器里找不到 nvcc，USE_CUDA_IMAGE 镜像没生效。"
    print("\n[完成] nvcc 可用")


@app.function(
    image=image,
    env=_CONTAINER_ENV,
    gpu=GPU,
    volumes={MODELS_DIR: volume},
    timeout=60 * 60 * 4,
    scaledown_window=600,
    # 只留 1 个容器：Modal 默认会按需扩容，多个容器时请求会被负载均衡
    # 到不同容器，前缀缓存测不准（2026-11-21 实测：热请求被分到空缓存
    # 容器，TTFT 50.88 秒假阴性），还会顶满工作区 10 GPU 并发上限。
    max_containers=1,
    # 关键：Modal 默认每个容器一次只处理 1 个输入（实测 vLLM 引擎恒为
    # Running: 1 reqs，并发 curl 全部排队超时），EvalScope 批量评测时
    # 请求被串行化。用 @modal.concurrent 放开并发（allow_concurrent_inputs
    # 参数在 modal 1.6 已废弃、会直接报 DeprecationError）。
)
@modal.concurrent(max_inputs=MODAL_MAX_INPUTS)
@modal.web_server(port=8000, startup_timeout=600)
def serve():
    """起 vLLM OpenAI 兼容服务：/v1/chat/completions + /metrics，端口 8000。
    配置经 env=_CONTAINER_ENV 传进容器（见上面注释）；不要给 web_server
    装饰的函数加参数，Modal 不允许。"""
    _check_fp8_gpu(GPU, QUANT)
    cmd = [
        "vllm", "serve", MODEL_PATH,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--served-model-name", SERVED_NAME,
        "--tensor-parallel-size", "1",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--kv-cache-dtype", "fp8",
        "--gpu-memory-utilization", GPU_MEM_UTIL,
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--max-num-batched-tokens", str(MAX_NUM_BATCHED_TOKENS),
        "--enable-prefix-caching",
        "--enable-per-request-metrics",
    ]
    if QUANT == "fp8":
        cmd += ["--quantization", "fp8"]
    if COMPILATION_MODE:
        # 只在显式指定时加；空值走 vLLM 默认（eager + CUDA 图）。
        cmd += ["--compilation-config", json.dumps({"mode": int(COMPILATION_MODE)})]
    if GDN_PREFILL_BACKEND:
        cmd += ["--gdn-prefill-backend", GDN_PREFILL_BACKEND]
    if ATTENTION_BACKEND:
        cmd += ["--attention-backend", ATTENTION_BACKEND]
    if TOOL_CALL_PARSER:
        cmd += ["--enable-auto-tool-choice", "--tool-call-parser", TOOL_CALL_PARSER]
    if REASONING_PARSER:
        cmd += ["--reasoning-parser", REASONING_PARSER]
    if PROFILER_CONFIG:
        cmd += ["--profiler-config", PROFILER_CONFIG]
    if MAX_MODEL_LEN > NATIVE_LEN:
        factor = round(MAX_MODEL_LEN / NATIVE_LEN, 4)
        overrides = {
            "rope_parameters": {
                "rope_type": "yarn",
                "factor": factor,
                "original_max_position_embeddings": NATIVE_LEN,
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
                "mrope_section": [11, 11, 10],
                "mrope_interleaved": True,
            }
        }
        print(f"[注意] MAX_MODEL_LEN={MAX_MODEL_LEN} 超出原生 {NATIVE_LEN}，"
              f"启用 YaRN 外推（factor={factor}）。")
        cmd += ["--hf-overrides", json.dumps(overrides)]
    print("启动：", " ".join(cmd))
    subprocess.Popen(cmd)


@app.local_entrypoint()
def smoke(url: str):
    """对已部署的服务跑冒烟（复用 smoke-test.sh 的 4 项，只用标准库）。
    用法：modal run modal/modal_vllm_qwen27b.py --url https://<部署地址>"""
    import time
    import urllib.request

    base = url.rstrip("/")
    passed, failed = 0, 0

    def ok(msg):
        nonlocal passed
        print(f"[通过] {msg}")
        passed += 1

    def bad(msg):
        nonlocal failed
        print(f"[失败] {msg}")
        failed += 1

    print(f"== 冒烟测试：{base}（模型 {SERVED_NAME}）==")
    try:
        with urllib.request.urlopen(base + "/health", timeout=10) as r:
            r.read()
        ok("服务活着（/health 返回 200）")
    except Exception as e:
        bad(f"连不上 /health：{e}")

    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=10) as r:
            models = r.read().decode()
        if SERVED_NAME in models:
            ok(f"模型列表里有 {SERVED_NAME}")
        else:
            bad(f"模型列表里没有 {SERVED_NAME}：{models[:200]}")
    except Exception as e:
        bad(f"/v1/models 请求出错：{e}")

    try:
        payload = json.dumps({
            "model": SERVED_NAME,
            "messages": [{"role": "user", "content": "用一句话介绍你自己。"}],
            "max_tokens": 64,
            "temperature": 0,
            "stream": True,
        }).encode()
        req = urllib.request.Request(
            base + "/v1/chat/completions", data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        t0 = time.time()
        t_first = t_last = None
        n_tokens = 0
        text_parts = []
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                token_text = (
                    delta.get("content")
                    or delta.get("reasoning_content")
                    or delta.get("reasoning")
                )
                if token_text:
                    now = time.time()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    n_tokens += 1
                    text_parts.append(token_text)
        if t_first is None:
            bad("没收到任何 token，服务可能没正常生成。")
        else:
            ttft = t_first - t0
            tpot = ((t_last - t_first) / (n_tokens - 1) * 1000) if n_tokens > 1 else 0.0
            print(f"首 token 耗时 TTFT：{ttft:.2f} 秒")
            print(f"平均每 token 耗时 TPOT：{tpot:.1f} 毫秒（共 {n_tokens} 个 token）")
            print("模型回答：" + "".join(text_parts)[:200])
            if ttft < 15 and tpot < 75:
                ok("流式生成正常，TTFT/TPOT 在短提问门槛内")
            else:
                bad("TTFT ≥ 15 秒或 TPOT ≥ 75 毫秒，短提问都不该这么慢。")
    except Exception as e:
        bad(f"流式请求出错：{e}")

    try:
        with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
            metrics = r.read().decode()
        if "vllm:num_requests_running" in metrics:
            ok("/metrics 有 vllm 指标（vllm:num_requests_running）")
        else:
            bad("/metrics 里没有 vllm 指标。")
    except Exception as e:
        bad(f"/metrics 请求出错：{e}")

    print(f"\n== 结果：通过 {passed} / 失败 {failed} ==")
    raise SystemExit(0 if failed == 0 else 1)
