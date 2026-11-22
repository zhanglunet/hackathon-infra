# Modal H200 冷 TTFT 调优第 4 轮：关 breakable cudagraph 再试 mode=3

- 日期：2026-11-22
- 部署：Modal H200 141GB，Qwen3.8-27B-FP8，vLLM 0.34.0，MAX_MODEL_LEN=272000（YaRN 外推）
- App：`ap-tfbyuabvewPI501F0fYEsu`，`https://zhanglunet--qwen27b-vllm-serve.us-east.modal.run`
- 部署创建 03:38:44 UTC，/health 200 于 03:41:32 UTC，停部署 03:44:47 UTC（已复核 stopped、0 tasks、/health 404）

## 结论先行

- **mode=3 的降级修好了，但编译还是没接通。** 关 breakable cudagraph 后，EngineCore 最终配置里 mode 保持 `VLLM_COMPILE (3)`（上轮被静默重置回 NONE），`use_breakable_cudagraph: False`。但引擎随后明确报：**模型不支持 torch.compile**，没产出 compiled submodule，CUDA 图回退到 FULL_DECODE_ONLY。
- **冷 TTFT 没快**：53.64 秒、55.76 秒（两次冷请求），对照基线 51.72 / 53.39 秒；热 TTFT 2.37 秒、TPOT 约 15–17 毫秒，均与基线持平。15 秒目标仍远。
- mode=3 这条路到此为止：问题不在 cudagraph 开关，而在模型本身不支持编译。下一轮应换方向（见 §6）。

## 本轮配置

- 代码改动（`modal/modal_vllm_qwen27b.py`）：新增 `VLLM_USE_BREAKABLE_CUDAGRAPH` 环境变量进 `_CONTAINER_ENV`，本地同名变量控制，默认 `"1"`（保持旧行为）。其余默认值未动。
- 部署命令：`MODAL_GPU=H200 MAX_MODEL_LEN=272000 MAX_NUM_BATCHED_TOKENS=65536 COMPILATION_MODE=3 VLLM_USE_BREAKABLE_CUDAGRAPH=0 modal deploy modal/modal_vllm_qwen27b.py`（max_containers=1 已在代码里）。
- 部署一次成功，无初始化失败重试。

## mode=3 是否接通（日志证据）

三条关键日志，全部来自 `modal app logs qwen27b-vllm --tail 1000`：

1. APIServer 解析的配置（`startup.py:96`）：`compilation_config: {'mode': <CompilationMode.VLLM_COMPILE: 3>, ... 'use_breakable_cudagraph': None, ...}`——mode=3 如数传入。
2. EngineCore 最终配置（`core.py:153`，03:39:58）：`compilation_config={'mode': <CompilationMode.VLLM_COMPILE: 3>, ..., 'use_breakable_cudagraph': False, 'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>, 'compile_ranges_endpoints': [65536], ...}`——**mode 仍是 VLLM_COMPILE (3)，上轮的静默降级没有再发生**；`use_breakable_cudagraph: False` 证明 `VLLM_USE_BREAKABLE_CUDAGRAPH=0` 生效。
3. 但随即两条警告堵死了编译：
   - `(EngineCore pid=32) WARNING 11-22 03:40:03 [vllm.py:3737] 'torch.compile' is turned on, but the model /models/Qwen3.8-27B-FP8 does not support it. Please open an issue on GitHub if you want it to be supported.`
   - `(EngineCore pid=32) WARNING 11-22 03:41:04 [compilation.py:1636] Cudagraph mode FULL_AND_PIECEWISE requires piecewise capture, but the loaded model provides neither a compiled submodule nor breakable CUDA graphs. Overriding to FULL_DECODE_ONLY.`
   - 结果：CUDA 图以 FULL 模式捕获 7 张（上轮 PIECEWISE 11 张），全程无 Inductor 编译活动日志（无 Compiling/torch.compile 相关行）。

## 延迟测量

提示：《白鲸》正文，271,500 token（templated），max_tokens=64，temperature 0。脚本：`probe/run_probe.py`（冷+热连发）与 `probe/cold_only.py`（换开头不同的 `prompt_272k_v2.txt` 再测一次冷）。

| 请求 | TTFT（秒） | TPOT（毫秒） | 输出 token | 备注 |
|---|---|---|---|---|
| 冷 1（run_probe） | 53.64 | 15.5 | 59 | 首请求，完整 prefill |
| 热（run_probe） | 2.37 | 21.5 | 8 | 前缀缓存命中；token 少，TPOT 噪声大 |
| 冷 2（cold_only，v2 提示） | 55.76 | 17.1 | 36 | 换开头避免命中缓存，确认冷值 |

对照基线（2026-11-21）：冷 51.72 / 53.39 秒、热 2.36 秒、TPOT 14.7 毫秒。**冷 TTFT 两次都在基线区间内（53.64 / 55.76 秒），无改善**；热 TTFT 与 TPOT 持平。

附记：两次请求前后 `/metrics` 的 `vllm:kv_cache_usage_perc` 均读 0.0（疑似指标口径问题，照实记录）；热请求 2.37 秒本身证明前缀缓存命中，未受此影响。

## 引擎占用（本轮，与 8192 步长基线对比）

- 权重 28.42 GiB；KV 缓存总容量 2,592,357 token（基线 2,955,530），单请求最大并发 9.53x（基线 10.87x）；可用 KV 86.55 GiB（基线 95.41 GiB）；峰值激活 9.94 GiB（基线 1.73 GiB）。
- 原因：`MAX_NUM_BATCHED_TOKENS=65536` 拉高单步激活显存，vLLM 少分了 KV 块。单 272K 请求只占 KV 约 10.5%，仍够用；但若要多并发，应回到 8192 步长换 KV 余量。
- 引擎初始化（profile + 建 KV + warmup）67.74 秒。

## 费用与墙钟

- H200 墙钟约 6 分钟（部署 03:38:44 → 停 03:44:47 UTC），按 H200 $4.54/时估算约 **$0.46**，预算 $2.30 内。部署已停并复核。
- 全程无 ModelScope 联网，未动 BF16 对照。

## 下一步建议（未执行，待定）

1. **mode=3 / mode=1 编译路线可结案**：`vllm.py:3737` 的警告针对 torch.compile 本身，mode=1（STOCK_TORCH_COMPILE）大概率撞同一堵墙，不值得再烧一轮验证。如需彻底确认，可只查 vLLM 上游 Qwen3.5 是否在支持编译的模型名单里，不用上 GPU。
2. **回攻真正的瓶颈**：profiling 结论仍是 FA3 全注意力 prefill 约 65% + FP8 GEMM 约 25%。候选：全注意力后端对照（FlashInfer/Triton attention vs FA3，需先核实 vLLM 0.34.0 里 Qwen3.5 可用后端列表）。
3. **核实 SLO 口径**：P90 提示长度取决于组织方长度分布；272K/15 秒是最严读法，真实门槛可能略低，长度分布未拿到前算不准。
4. 注意 `MAX_NUM_BATCHED_TOKENS=65536` 的副作用（KV 容量 2.96M → 2.59M token）：后续轮次若无明确收益，改回 8192。
