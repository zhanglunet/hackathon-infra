# 环境与优化说明表（初稿）

> 状态：**初稿，2026-11-23 编**。Qwen3.8-27B-FP8 一栏是 Modal H100/H200 上已验证的配置；
> DeepSeek-V4.1-Flash、第二款模型、Blackwell 真机上的数字全部是 **TBD（待真机验证）**，
> 比赛机到手复测后重填。凡标 TBD 的都不是实测数，不要当结论用。
>
> 规则依据：飞书《算力优化黑客松测试标准说明》§五第 3 项（引擎版本、上下文与并行切分、
> 量化与缓存/加速策略）。来源：`devlog/` 14 篇 + `modal/README.md`；逐项来源见 §8。

## 1. 总览（按模型）

| 模型 | 状态 | 引擎 | 上下文 | 并行切分 | 量化 |
|---|---|---|---|---|---|
| Qwen3.8-27B-FP8（自选 1） | 已验证（Modal H100/H200） | vLLM 0.34.0 | 272000（YaRN 外推） | 单卡 TP=1 | FP8 权重 + FP8 KV |
| DeepSeek-V4.1-Flash-1017（必选） | **TBD**：从没上过机 | vLLM 0.34.0（计划） | 272000（计划） | **TBD**：8 卡 TP 方案未定 | **TBD**（官方权重 FP8，待验证） |
| 第二款模型（自选 2） | **TBD**：未定 | TBD | TBD | TBD | TBD |

## 2. 引擎与软件环境（已验证，Qwen 排练环境）

| 项目 | 值 | 来源 |
|---|---|---|
| 推理引擎 | vLLM 0.34.0 | `modal/modal_vllm_qwen27b.py`（`VLLM_VERSION`） |
| torch | 2.14.0+cu130（CUDA 13.0 运行时） | 第 6 轮 `check_nvcc` 实测（`pip install vllm==0.34.0` 实际拉取 `cu130`，非 12.8） |
| Python | 3.12 | Modal 镜像 `debian_slim(python_version="3.12")` / CUDA devel 镜像 `add_python="3.12"` |
| nvcc | 12.8.61（CUDA 12.8 devel 镜像） | 第 6 轮 `check_nvcc` 实测；与 torch `cu130` 版本错位，Hopper 上 FlashInfer JIT 仍通但 DeepGEMM 被禁；**Blackwell 上须验证并优先对齐 CUDA 13.0 toolkit** |
| 排练硬件 | Modal H100 80GB / H200 141GB | 第 1–8 轮 devlog |
| 比赛硬件 | 4 台 × 8×RTX 6000D-Blackwell-84G | 规则 §一；**TBD**：真机上全部重测 |
| 服务接口 | `/v1/chat/completions` + `/metrics`，端口 8000 | `modal_vllm_qwen27b.py` / `serve-qwen27b.sh` |
| 服务模型名 | `qwen3.8-27b` | `--served-model-name` |

## 3. 上下文与并行切分

### 3.1 Qwen3.8-27B-FP8（已验证）

| 项目 | 值 | 来源 |
|---|---|---|
| 原生上下文 | 262144 token（`config.json` 的 `max_position_embeddings`） | `serve-qwen27b.sh` 文件头 |
| 比赛上下文 | 272000 token，YaRN 外推（factor 1.0376 ≈ 1.04） | 第 2 轮（H200 272K 探测）；`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` 必设，否则 vLLM 0.34.0 报 ValueError（2026-11-21 实测：不设它时容器循环重启约 11 分钟） |
| 模型结构 | 64 层：16 层全注意力 + 48 层 Gated DeltaNet（线性注意力） | 第 5 轮源码核实 |
| 并行切分 | 单卡 TP=1；无多机、无 PD 分离 | 排练只用单卡 |
| KV 缓存总容量（H200） | 2,955,530 token（步长 8192 时）；步长 65536 时 2,592,357 token | 第 2 轮 / 第 4 轮实测；单请求 272K 约占 9–10% |
| 最大并发序列数 | 32（`--max-num-seqs 32`） | Modal 配置 |

### 3.2 DeepSeek-V4.1-Flash-1017（必选，脚本骨架已按官方 recipe 就绪，待真机验证）

| 项目 | 值 | 状态 |
|---|---|---|
| 权重 | `deepseek-ai/DeepSeek-V4.1-Flash-1017`：约 511 GB（476 GiB），48 分片；552B MoE 骨干 + 196B Engram（183.1 GiB）+ ~14B DSpark 草稿头；8B prefill / 16B decode 激活；1M 原生上下文；全局 KV 890 字节/token | 2026-11-23 核对自 vLLM 官方 recipe（待真机核实） |
| 必选参数 | `--tokenizer-mode deepseek_v41`（仓库无 Jinja template）、`--enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41`（V4.1 带空格 DSML 标签）、`VLLM_ENGINE_READY_TIMEOUT_S=3600` | 已写入 `serve-deepseek.sh`（待真机验证） |
| 并行切分与 Engram | 默认 8 卡 TP + `--engram-config '{"cpu_offload":true}'`（显存驻留 ~36.6 GiB/卡）；支持切换 `ENGRAM_CPU_OFFLOAD=false`、DEP（`TP_SIZE=1 DP_SIZE=8 ENABLE_EP=1`）、`--language-model-only`、DSpark 推测解码（V4.1 已弃用 MTP） | **TBD**：待真机验证与选优 |
| 上下文 | 272000（原生 1M 上下文，无需 YaRN） | 待真机验证 |
| 脚本与镜像 | `download-deepseek.sh`、`serve-deepseek.sh`、`Dockerfile`（CUDA 13.0 devel 草稿） | 已入库，待真机构建与实跑 |

### 3.3 第二款模型（TBD）

未定。候选：GLM-5.3-Flash（建议优先，见就绪检查 devlog §3）、GLM-5.3、Kimi K3.1 Pro、MiniMax-H3。
定下来后按 §3.1 的格式补一行。

## 4. 量化与缓存/加速策略（Qwen3.8-27B-FP8，已验证）

| 策略 | 值 | 验证来源与效果 |
|---|---|---|
| 权重精度 | FP8（`--quantization fp8`，per-tensor；激活动态 per-token） | 第 7 轮源码核实：`quantization_config={quant_method: fp8, activation_scheme: dynamic, fmt: e4m3}`，无 `weight_block_size` |
| KV 缓存精度 | FP8（`--kv-cache-dtype fp8`） | Modal 配置；BF16 权重时也保持 fp8 KV（`serve-qwen27b.sh` 文件头显存账本） |
| 前缀缓存 | 开（`--enable-prefix-caching`） | Modal 配置 |
| 全注意力后端 | **FlashInfer**（`--attention-backend FLASHINFER`） | 第 6 轮：冷 41.09/43.8 秒，比 FA3 快约 21%，目前最快；须 nvcc（CUDA devel 镜像，开关 `USE_CUDA_IMAGE=1`） |
| 全注意力后端（对照） | FA3（默认）：冷 52.52/54.94 秒；Triton：冷 252.0/252.41 秒（慢约 4.8 倍） | 第 5 轮 |
| chunked prefill 步长 | 65536（`--max-num-batched-tokens 65536`） | 第 1–3 轮：8192→65536 只快 1.5%，分片不是瓶颈；262144 初始化失败；副作用：KV 容量 2,955,530→2,592,357 token |
| GDN prefill 后端 | 默认（auto→Hopper 上 FlashInfer JIT） | Profiling 轮：GDN 全部内核仅约 1.4 秒（3%），不是瓶颈；triton 轮 53.04 秒无改善 |
| 编译 | 默认（不传 `--compilation-config`）；mode=3 经核实是空转 | 第 4 轮：`VLLM_USE_BREAKABLE_CUDAGRAPH=0` 后 mode 保持 3，但模型不支持 torch.compile，无 Inductor 活动；冷 53.64/55.76 秒无改善，**编译路线结案** |
| breakable CUDA 图 | 关（`VLLM_USE_BREAKABLE_CUDAGRAPH=0`） | 第 4 轮：vLLM 对 Qwen3.5 架构默认自动开，会把 mode 置回 NONE；设 0 退出该路径 |
| FP8 GEMM 内核 | CUTLASS（`cutlass_scaled_mm`） | 第 7 轮：DeepGEMM 不适用（只在 block 路候选表里）；第 8 轮：FlashInfer FP8 线性核两道门都拦（算力须 ≥100，H200 是 90；激活须 per-tensor，本模型是动态 per-token），**GEMM 侧无开关** |
| 显存占用比 | 0.90（`--gpu-memory-utilization 0.90`） | Modal 配置 |
| 并发（Modal 专属） | `@modal.concurrent(max_inputs=16)`，单容器 `max_containers=1` | EvalScope 轮：修好 Modal 默认串行化，吞吐 80→800–900 tok/s；**裸机不适用** |

## 5. 精度结果（代理集，非官方 190 题）

| 代理集 | FP8 | BF16 | 差值 |
|---|---|---|---|
| AIME 2025（30 题） | 56.67%（17/30） | 46.67%（14/30） | FP8 +3 题 |
| GPQA Diamond 前 50 | 70%（35/50） | 68%（34/50） | FP8 +1 题 |
| LiveCodeBench v6 前 40 | 42.5%（17/40） | 42.5%（17/40） | 持平，逐题完全相同 |
| BFCL v3 simple 前 50 | 94.00%（47/50） | 96.00%（48/50） | BF16 +1 题 |
| **合计 170 题** | **116** | **113** | **FP8 +3 题** |

结论（代理集）：FP8 高 1.77pp，无净损失；McNemar 检验 p≈0.55，差异不显著——
**测不到 FP8 量化的系统性精度损失**。附带：BF16 平均延迟比 FP8 高约 27%。
口径提醒：官方 190 题不公开，以上是自测代理集，不能等同官方复测分；
正式自测报告须用评审组统一下发的数据集与脚本重跑（见就绪检查 G2/G8）。
来源：EvalScope 轮 + BFCL 补跑 + BF16 对照 devlog。

## 6. 性能结果

| 指标 | Qwen3.8-27B-FP8（H200，FlashInfer） | 状态 |
|---|---|---|
| 冷 TTFT（272K 单请求） | 41.09 / 43.8 秒 | 已测；**仍差 15 秒目标约 2.8 倍** |
| 热 TTFT | 2.14 秒 | 已测，通过 |
| TPOT | 约 12.2 毫秒 | 已测，通过（门槛 75 毫秒） |
| AIPerf 并发阶梯 8/16/32/64/80 | **TBD** | 从没跑过；官方包到手后补 |
| 四档 SLO 场景（50K/0.70、100K/0.9、1K/8K、32K） | **TBD** | 只测过 272K 单请求；官方包到手后补 |
| Blackwell（RTX 6000D）冷 TTFT | **TBD** | H200 成绩不能直接搬；真机重测 |

## 7. TBD 清单（真机复测后重填）

| # | 待填项 | 依赖 |
|---|---|---|
| T1 | DeepSeek-V4.1-Flash 并行切分（TP/专家并行/PD 分离）、上下文、启动命令 | 比赛机 |
| T2 | 第二款模型整行（定下来后按 §3.1 格式补） | 用户拍板 + 比赛机 |
| T3 | Blackwell 上冷/热 TTFT、TPOT（Qwen、DeepSeek、第二款） | 比赛机 |
| T4 | AIPerf 并发阶梯与 Goodput（3 款模型） | 官方包 + 比赛机 |
| T5 | 四档 SLO 场景实测 | 官方包 + 比赛机 |
| T6 | 容器镜像（Dockerfile）或可复测服务地址 | 比赛机 |
| T7 | 正式自测报告（评审组模板 + 官方包重跑数） | 官方包 + 比赛机 |

## 8. 来源

- 规则：飞书《算力优化黑客松测试标准说明》（2026-11-23 重新导出，实质内容与 11-21 版一致）。
- 排练：`devlog/2026-11-21-modal-h100-冒烟.md`、`2026-11-21-modal-h200-272k.md`、
  `2026-11-21-modal-h200-冷ttft调优.md`、`2026-11-21-modal-h200-profiling.md`、
  `2026-11-22-modal-h100-evalscope.md`、`2026-11-22-modal-h100-bfcl.md`、
  `2026-11-22-modal-h200-冷ttft-mode3.md`、`2026-11-22-modal-h200-注意力后端对照.md`、
  `2026-11-22-modal-h100-bf16.md`、`2026-11-22-modal-h200-flashinfer-nvcc.md`、
  `2026-11-23-modal-h200-deepgemm.md`、`2026-11-23-modal-h200-flashinfer-linear.md`、
  `2026-11-23-submission-readiness.md`。
- 配置对照：`modal/modal_vllm_qwen27b.py`、`modal/README.md`、`serve-qwen27b.sh`、
  `setup-vllm.sh`（2026-11-23 补 flag 后）。
