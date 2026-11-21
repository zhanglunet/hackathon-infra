# 2026-11-21 Modal H200 profiling：272K 冷 TTFT 51 秒的瓶颈定位

> 日期：2026-11-21（JST 23:50–2026-11-22 00:03，UTC 14:50–15:03）
> 模型：Qwen3.8-27B-FP8（`Qwen/Qwen3.8-27B-FP8`，served 名 `qwen3.8-27b`），vLLM 0.34.0，Modal H200
> 探测输入：同一份《白鲸》正文，tokenizer 数准 271,500 token（模板化后）+ 64 输出 token；冷请求为全新部署后的首请求（缓存为空）
> 部署地址：`https://zhanglunet--qwen27b-vllm-serve.us-east.modal.run`（每轮测完即停）

## 结论先说

- **瓶颈不是 GDN 线性注意力，而是全注意力（full attention）**：torch profiler 实测，51 秒 prefill 里全注意力 FA3 内核占约 33.3 秒（65%）、FP8 GEMM 占约 13.0 秒（25%）、GDN 全部内核合计只占约 1.4 秒（3%）。GDN 嫌疑排除。
- **GDN triton 轮验证了这一点**：`--gdn-prefill-backend triton` 冷 TTFT 53.04 秒，比基线 51.72 秒还慢一点（轮间噪声约 ±1 秒），无改善。
- **mode=3 被静默降级的原因已在源码里定位**：Qwen3.5 在 CUDA 上属于默认启用 breakable CUDA graphs 的架构，`_maybe_enable_breakable_cudagraph()` 自动开启后把 `compilation_config.mode` 置为 NONE。下一轮可用 `VLLM_USE_BREAKABLE_CUDAGRAPH=0` 退出该路径再试 mode=3（见 §4）。
- **纠正上轮的一个口径**：devlog 里“约 15% MFU”是错的——它只算了线性层 FLOPs（2×27e9×271,500），漏了注意力 FLOPs。271K 下 16 层全注意力的 QK^T+AV 约 1.8e16 FLOPs，占总 FLOPs 约一半以上；按完整账算，有效算力约 520–620 TFLOPS，混合峰值利用率约 35–45%，仍偏低，但不是 15%。

## 目标（不变）

冷 TTFT ≤ 15 秒（P90，长上下文场景档；272K 单请求适用档）。基线冷 TTFT 51.72 / 53.39 秒。

## 模型结构（实测前已核对 HF config.json）

Qwen3.8-27B-FP8（`Qwen3_5ForConditionalGeneration`，稠密 27B，无 MoE）：64 层，其中 48 层线性注意力（GDN）+ 16 层全注意力（每 4 层一层：3、7、11、…、63）；hidden 5120，Q 头 24 × head_dim 256，KV 头 4（GQA 6:1）；原生上下文 262144。

## 第 1 轮：torch profiler 定位（冷 TTFT 51.94 秒）

- 配置：`MAX_NUM_BATCHED_TOKENS=65536`，其余与基线一致；另加 `--profiler-config '{"profiler":"torch","torch_profiler_dir":"/models/profiler","torch_profiler_with_flops":true,"torch_profiler_record_shapes":true,"torch_profiler_with_stack":false,"ignore_frontend":true}'`（trace 写进 Volume `/models/profiler`，随后用 `modal volume get` 取回）。
- 流程：部署 14:50:43 UTC，/health 200 于 14:53:22（try 7，初始化约 2.6 分钟）→ POST `/start_profile` 200 → 单次冷请求 → POST `/stop_profile`（第一次 POST 超时 000，重试一次 200）→ 停部署 14:54:58。墙钟 GPU 时间约 4.25 分钟。
- 实测：冷 TTFT **51.94 秒**（TPOT 15.1 毫秒，59 token），与基线一致——profiling 本身没拖慢请求。
- 采集产物：Volume `qwen27b-weights` 下 `profiler/rank0.1795272871826462114.pt.trace.json.gz`（Chrome trace，解压后约 139 MB，285,303 个事件）与 `profiler/rank0.profiler_out.txt`（算子汇总表）。本地副本在 /tmp（临时文件，随 VM 清理）。

### profiler 结果：时间花在哪（两份数据交叉核对一致）

算子汇总表（`rank0.profiler_out.txt`，self CUDA 时间，去掉 CPU op 包装行的重复计数）与 Chrome trace 内核时间求和互相印证（FA3：表 33.26 秒 vs trace 33.61 秒；GEMM：表 12.95 秒 vs trace 13.42 秒，含 decode 的 0.47 秒；内核总时间 51.0 秒 vs TTFT 51.94 秒——GPU 占用约 98%，完全 GPU  bound）。

| 阶段 | 内核 | 时间 | 占比 |
| --- | --- | --- | --- |
| 全注意力 prefill | FA3（`_vllm_fa3_C::fwd`，80 次 = 16 层 × 5 个 chunk） | **约 33.3 秒** | **约 65%** |
| 线性层 | FP8 GEMM（`cutlass_scaled_mm`，1280 次） | **约 13.0 秒** | **约 25%** |
| GDN 线性注意力 | GDN 全部内核（FlashInfer delta_rule + conv1d + post_conv + gated_rms_norm） | 约 1.4 秒 | 约 3% |
| 其他 | FP8 量化（`per_token_group_fp8_quant`）约 1.1 秒，激活/归一化等约 2.2 秒 | 约 3.3 秒 | 约 6% |

- chunk 切分印证：profile 标注显示 271,500 token 被切成 5 个 chunk（64000 × 4 + 15500 × 1），与 65536 步长一致。
- FA3 单次调用约 416 毫秒（一层全注意力处理一个 64K-token chunk，KV 最长 271K）。按 QK^T+AV 的 FLOPs 估算，有效算力约 530 TFLOPS（未扣因果掩码省掉的部分；扣掉后更低），相对 BF16 峰值 989 TFLOPS 有缺口，但 flash attention 在 head_dim 256 + GQA 6:1 的形状下本就不满——这是结构性瓶颈，不是某个开关能一下解决的。
- 附：CPU 侧 `cudaLaunchKernelExC` 占 CPU 总时间 46%、`aten::mm` 自身 CPU 时间 12.45 秒，但 GPU 全程忙碌，CPU 启动开销被掩盖，不是瓶颈。

## 第 2 轮：`--gdn-prefill-backend triton`（无改善，假设证伪）

- 配置：`MAX_NUM_BATCHED_TOKENS=65536` + `--gdn-prefill-backend triton`，其余与基线一致。`--gdn-prefill-backend` 是 vLLM 0.34.0 的真实 CLI 选项（choices：flashinfer/triton/cutedsl/aiter_flydsl/sycl/hip），`arg_utils.py:2902` 把它写进 `additional_config["gdn_prefill_backend"]`，解析器再读——链路已在源码里核实（日志里的 "Using … GDN prefill kernel" 一行因 `modal app logs` 只取最近 100 条、初始化日志已滚出，未直接看到，不作断言）。
- 流程：部署 14:58:28 UTC，/health 200 于 15:01:56（try 9，初始化约 3.4 分钟）→ `run_probe.py` 冷+热 → 停部署 15:03:35。墙钟 GPU 时间约 5.1 分钟。
- 实测：冷 TTFT **53.04 秒**（TPOT 14.0 毫秒），热 TTFT 2.43 秒（TPOT 16.2 毫秒）。与基线 51.72 秒、profiling 轮 51.94 秒相比无改善（53.04 秒在轮间噪声约 ±1 秒的范围外沿，偏慢，不作“变慢”的断言）。
- 结论：与 profiling 一致——GDN 只占约 3%，换后端最多省约 1 秒。**GDN 后端不是调优方向。**

## §4 mode=3 被降级的原因（源码定位，未用 GPU 验证）

上轮实测：启动命令行确有 `--compilation-config '{"mode": 3}'`，APIServer 解析为 `mode=VLLM_COMPILE (3)`，但 EngineCore 最终配置是 `mode=NONE (0)`，且无任何警告。本轮在 vLLM 0.34.0 源码里找到链路：

1. `vllm/config/vllm.py:138` `default_breakable_cudagraph_architectures()`：CUDA 平台上返回 `DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES | _QWEN3_5_ARCHITECTURES`，注释原话 "Qwen3.5 migrates to breakable CUDA graphs on NVIDIA only"——**Qwen3.5 在默认启用 breakable CUDA graphs 的架构名单里**。
2. `VllmConfig.post_init` 里 `_maybe_enable_breakable_cudagraph()`（`vllm/config/vllm.py:810`）：`_uses_breakable_cudagraph_by_default()` 为真时自动开启 breakable cudagraph，并执行 `self.compilation_config.mode = CompilationMode.NONE`（`:833`）——**无论用户传了什么 mode，都会被置回 NONE，且只记一条 info 级日志**（"Auto-enabling VLLM_USE_BREAKABLE_CUDAGRAPH=1. Set VLLM_USE_BREAKABLE_CUDAGRAPH=0 to opt out."）。
3. 旁证：上轮日志里 CUDA 图以 PIECEWISE 模式捕获了 11 张图——正是 breakable cudagraph 的行为，与该路径启用一致。
4. 其他重置路径均已排除：`enforce_eager` 未设置（`vllm/config/model.py:1516` 的 `enforce_eager=True` 只针对 ROCm 编解码模型）；`TORCH_COMPILE_DISABLE` 未设置；`models/config.py:470/1538` 的重置只针对 Unlimited-OCR-2 / LongCat-Flash，与 Qwen3.5 无关。

**下一轮可试**：`VLLM_USE_BREAKABLE_CUDAGRAPH=0` + mode=3（或 mode=1），看 Inductor 编译是否真正接通。风险：Qwen3.5 被迁到 breakable graphs 想必有原因（PIECEWISE 编译图与 GDN 内核的兼容性），退出后图捕获可能失败——先备好失败即停。

## 两轮汇总

| 轮次 | 配置 | 冷 TTFT | 热 TTFT | TPOT | 墙钟 GPU 时间 | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| 基线 | 8192，eager | 51.72 / 53.39 秒 | 2.36 秒 | 14.7 毫秒 | — | — |
| profiling | 65536 + torch profiler | 51.94 秒 | —（只测冷） | 15.1 毫秒 | 约 4.25 分钟 | 瓶颈：FA3 全注意力 65% + FP8 GEMM 25%，GDN 仅约 3% |
| GDN triton | 65536 + `--gdn-prefill-backend triton` | 53.04 秒 | 2.43 秒 | 14.0 毫秒 | 约 5.1 分钟 | 无改善，GDN 假设证伪 |

**目标未达成：冷 TTFT 仍约 51–53 秒，距 15 秒目标差 3.4 倍。** 但瓶颈已定位，不再是猜参数。

## 费用

两轮墙钟合计约 9.4 分钟 H200，约 **$0.71**（$4.54/时；部署等待与初始化时间都算钱，但本轮两次初始化都一次成功、无重试）。剩余额度约 $19.90（本轮开始前 $20.61，账单口径待核）。

## 下一步（按优先级）

1. **mode=3（或 mode=1）+ `VLLM_USE_BREAKABLE_CUDAGRAPH=0`**：验证 Inductor 编译能否真正接通。编译若生效，GEMM 与逐元素算子的融合可能带来改善；若图捕获失败，记现象即停。这是 mode 降级定位后的直接跟进。
2. **全注意力后端对照**：FA3 占 65%，是最大头。单变量试 `--attention-backend` 的其他后端（FlashInfer / Triton attention，需先核实 vLLM 0.34.0 里 Qwen3.5 可用的后端列表），看 prefill 能否比 FA3 快。
3. **核对 P90 的提示长度口径**：若 P90 按场景分布算（智能体平均 50K、编程平均 100K），50K 提示按当前速度约 9.6 秒已进 15 秒——15 秒目标是否真需 3.4 倍提速，取决于 P90 长度，未核实。
4. **不要做的**：更大 `max_num_batched_tokens`（262144 已证实初始化失败）；更多 GDN 后端尝试（已证伪）；并发压测、精度测试（不在本轮范围）；升级 Modal 套餐（10 GPU 并发上限靠 `max_containers=1` + 单部署操作已可避开）。

## 附：本轮代码改动（随本日志推 main）

- `modal/modal_vllm_qwen27b.py`：新增 `GDN_PREFILL_BACKEND` 环境变量（空=默认；`triton` 等值时启动命令加 `--gdn-prefill-backend`）与 `PROFILER_CONFIG` 环境变量（JSON 字符串，非空时加 `--profiler-config`），均进 `_CONTAINER_ENV`。
- `modal/README.md`：环境变量表补 `GDN_PREFILL_BACKEND`、`PROFILER_CONFIG` 两行。
- `probe/cold_only.py`（新增）：单次冷请求脚本，配合 `/start_profile`、`/stop_profile` 用；`probe/cold_probe_result.json`（profiling 轮冷请求实测）、`probe/probe_result.json`（GDN 轮冷+热实测）。
