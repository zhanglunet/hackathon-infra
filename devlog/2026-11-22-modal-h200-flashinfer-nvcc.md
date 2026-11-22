# Modal H200 冷 TTFT 调优第 6 轮：给镜像加 nvcc，再测 FlashInfer 后端

- 日期：2026-11-22
- 部署：Modal H200 141GB，Qwen3.8-27B-FP8，vLLM 0.34.0，MAX_MODEL_LEN=272000（YaRN 外推）
- 配置：`MAX_NUM_BATCHED_TOKENS=65536`，`VLLM_USE_BREAKABLE_CUDAGRAPH=0`，`ATTENTION_BACKEND=FLASHINFER`，`USE_CUDA_IMAGE=1`；不设 COMPILATION_MODE；max_containers=1
- 全部署已停并复核：`modal app list` 0 deployed、/health 404

## 结论先行

- **FlashInfer 终于跑起来了，而且比 FA3 快：冷 41.09 / 43.8 秒、热 2.14 秒、TPOT 约 12.1–12.4 毫秒。** 对照 FA3（第 5 轮：冷 52.52 / 54.94 秒、热 2.29 秒、TPOT 约 15.7 毫秒），冷 TTFT 快约 11.4 秒（约 21%），TPOT 快约 22%。
- **15 秒目标仍远。** 冷 TTFT 41–44 秒，仍是目标的约 2.8 倍。FlashInfer 是目前测过最快的全注意力后端，但只啃掉了注意力 prefill 的一部分；剩下的缺口须换别的打法（见 §6）。
- **镜像开关已落地**：`USE_CUDA_IMAGE=1` 走 CUDA devel 基础镜像（含 nvcc），默认仍走 debian_slim，FA3 / BF16 的已验证路径一动不动。

## 代码改动

`modal/modal_vllm_qwen27b.py`，三处：

- 新增 `USE_CUDA_IMAGE` 环境变量（默认 `"0"`）。设 `"1"` 时镜像改为 `modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu24.04", add_python="3.12").pip_install("vllm==0.34.0", "huggingface_hub")`；默认仍走 `debian_slim`。
- 新增 `check_nvcc` 函数（不挂 GPU）：跑 `which nvcc`、`nvcc --version`、torch 版本，验证容器里 nvcc 可用。用法规：`USE_CUDA_IMAGE=1 python3 -m modal run modal/modal_vllm_qwen27b.py::check_nvcc`。
- 其余默认值未动（`ATTENTION_BACKEND` 是第 5 轮加的，本轮只传值 `FLASHINFER`）。

## nvcc 验证（先于部署，不挂 GPU）

`check_nvcc` 一次成功（App `ap-nPogR7s3q1JCrhrjSK4DQH`；约 5 分钟，基本是首次拉 CUDA devel 镜像的时间）：

- `which nvcc` → `/usr/local/cuda/bin/nvcc`
- `nvcc --version` → release 12.8, V12.8.61
- torch → `2.14.0+cu130`，cuda 13.0

**注意一个版本错位**：基础镜像的 nvcc 是 12.8，但 pip 装 vllm 0.34.0 时拉到的 torch 是 cu130（CUDA 13.0）——任务简报里“vllm 轮子对应 CUDA 12.8”的假设不准，pip 解析到的是最新 torch 2.14 cu130。错位没有拦住 FlashInfer JIT（见下），但有一个副作用：DeepGEMM 被禁用（日志原文：`DeepGEMM kernels are disabled: DeepGEMM's JIT needs nvcc 12.9 or newer, found 12.8`）。Profiling 轮的 GEMM 是 `cutlass_scaled_mm`，DeepGEMM 本来也没在用，所以本轮结果不受影响；想开 DeepGEMM 须换 12.9+（或 13.0）的 devel 镜像，见 §6。

## 后端切换与 JIT 编译的日志证据

引擎启动日志（App `ap-smGNMeauqSrWhxYDsM86PL`，部署 2026-11-22 08:11 UTC，/health 200 于 08:16:56，约 5.5 分钟启动，大头是 JIT 编译）：

- `Using AttentionBackendEnum.FLASHINFER backend.`（`cuda.py:489`）——后端确实切过去了，不是静默回退。
- `FlashInfer resolved query dtypes: prefill=torch.float8_e4m3fn, decode=torch.bfloat16, decode_backend=xqa, kv_cache_dtype=torch.float8_e4m3fn, arch=sm90`
- `flashinfer.jit: Building JIT module batch_prefill_with_kv_cache_dtype_q_e4m3_dtype_kv_e4m3_dtype_o_bf16_dtype_idx_i32_head_dim_qk_256_head_dim_vo_256_posenc_0_use_swa_False_use_logits_cap_False_f16qk_False_sm90`——第 5 轮卡住的那个 batch-prefill 模块，这次正常编译（08:15:08 开始，约 1.5 分钟后引擎继续 warmup）。
- 另有 `FlashInfer's precompiled kernels are missing ... FlashInfer then downloads and compiles kernels at startup` 的警告（flashinfer-cubin / flashinfer-jit-cache 未预装），首次启动编译是预期行为；`vllm download-kernels` 可预装，未做（只影响启动时间，不影响测量）。
- 其余与 FA3 轮一致：`torch.compile is turned on, but the model does not support it`（第 4 轮已结案）；GDN prefill 走 FlashInfer 内核（auto）；KV 缓存布局 LBNHC；FP8 注意力未校准 q_scale 的警告仍在（与 FA3 轮相同）。

## 延迟结果

提示：《白鲸》正文，271,500 token（templated），max_tokens=64，temperature 0。脚本：`probe/run_probe.py`（冷+热连发）与 `probe/cold_only.py`（换开头不同的 `prompt_272k_v2.txt` 再测一次冷）。结果文件：`probe/probe_result_flashinfer.json`、`probe/cold_probe_result_flashinfer.json`。

| 请求 | TTFT | TPOT | FA3 对照 | 差值 |
|---|---|---|---|---|
| 冷 1 | **41.09 秒** | 12.1 毫秒 | 52.52 秒 | 快 11.43 秒（约 22%） |
| 冷 2（换开头防缓存命中） | **43.8 秒** | 12.3 毫秒 | 54.94 秒 | 快 11.14 秒（约 20%） |
| 热 | **2.14 秒** | 12.4 毫秒 | 2.29 秒 | 快 0.15 秒 |

两次冷请求差 2.7 秒（41.09 vs 43.8），在 FA3 轮的两次冷差（52.52 vs 54.94，差 2.4 秒）同一量级，是正常波动。

**分解估算**：Profiling 轮的账是 FA3 prefill 约 33.3 秒 + FP8 GEMM 约 13 秒 + 其他约 6 秒 ≈ 52 秒。FlashInfer 冷 TTFT 约 41–44 秒，若 GEMM 与其他不变（约 19 秒），注意力 prefill 约 22–25 秒——FlashInfer 的 prefill 内核比 FA3 快约 30%。TPOT 从约 15.7 毫秒降到约 12.2 毫秒，decode 也快了约 22%（decode_backend xqa）。

**口径提醒**：冷 1 只出了 25 个 token（模型提前停），TPOT 是按 25 个 token 算的；冷 2 出了 57 个 token。TTFT 不受影响。

/metrics 读取在探针里两次报 `IncompleteRead`（chunked 编码的老问题，FA3 轮也有），只影响 KV 占用读数，不影响延迟数字。

## 部署与费用

- 部署 `ap-smGNMeauqSrWhxYDsM86PL`（H200，`USE_CUDA_IMAGE=1`）：创建 08:11 UTC，/health 200 于 08:16:56 UTC，停 08:19 UTC，已复核 stopped、0 tasks、/health 404。
- H200 墙钟约 8 分钟（08:11 → 08:19 UTC），按 $4.54/时估算约 **$0.61**；另有 `check_nvcc` 的 CPU 函数约 5 分钟（镜像拉取 + 验证），约 $0.03–0.05。本轮合计估算约 **$0.65**，预算 $1.50 内。无 ModelScope 联网，未动 BF16 对照。
- 部署前余额 $10.19，剩约 $9.54 估算，账单页未复核。

## 下一步建议

1. **FlashInfer 是当前最快的后端**，后续 272K 测量默认走它（`ATTENTION_BACKEND=FLASHINFER` + `USE_CUDA_IMAGE=1`）。启动多约 2 分钟 JIT 编译是固定成本，只影响冷启动测量，不影响 TTFT 本身。
2. **冷 TTFT 仍差约 26–29 秒**。按分解账，剩下的是 FP8 GEMM 约 13 秒 + 其他约 6 秒 + FlashInfer prefill 约 22–25 秒。候选方向：① FP8 GEMM 侧优化（DeepGEMM 须 nvcc 12.9+，换 13.0 devel 镜像可一并解决 torch 13.0 与 nvcc 的版本错位，但 DeepGEMM 能否接管本模型的线性层未核实）；② 接受 272K 冷 TTFT 超标，把赌注放在 P90 口径（真实门槛取决于组织方长度分布，未拿到）。
3. `MAX_NUM_BATCHED_TOKENS=65536` 的 KV 容量副作用（2.96M → 2.59M token）仍在，后续轮次若无明确收益改回 8192。
4. 镜像开关的回归：默认 `USE_CUDA_IMAGE` 为空/`"0"` 时仍走 debian_slim，FA3 / BF16 路径未受影响（本轮没重跑回归，改动只在 image 分支条件里，风险低）。
