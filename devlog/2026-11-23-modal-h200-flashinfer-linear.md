# Modal H200 冷 TTFT 调优第 8 轮：FlashInfer FP8 线性核为什么没被选中

- 日期：2026-11-23
- 部署：无（本轮只做源码核实，未上 GPU）
- 模型：Qwen3.8-27B-FP8（稠密），vLLM 0.34.0，Modal H200 141GB（未动用）
- 费用：**$0**；余额仍 $9.61（2026-11-22 16:23 核）

## 结论先行

- **FlashInfer FP8 线性核没被选中，有两个独立原因，任意一个都足以排除它：**
  1. `is_supported` 要求算力 ≥100，H200 是 SM90（算力 90）——这条在 H200 上就不通过。
  2. `can_implement` 要求激活与权重都是 per-tensor 尺度；本模型权重是 per-tensor，但激活是动态 per-token——这条在任何硬件上都不通过，包括比赛用的 Blackwell（算力 120）。
- **因此 `--linear-backend flashinfer_cutlass` 也救不了**：它只过滤候选表，同一对检查照样跑；没有用户可调的强制开关。
- CUTLASS 是本模型在 vLLM 0.34.0 内唯一可用的 per-tensor FP8 稠密内核，在 H200 和比赛 Blackwell 机上都是它。**GEMM 侧彻底没招了**（第 7 轮 DeepGEMM 结论不变）。

## 代码核实：证据链

源码来源：pip 轮子 `vllm-0.34.0-cp38-abi3-manylinux_2_28_x86_64.whl`（2026-11-16 构建），即 Modal 部署实际跑的版本；按需解压单个文件读（`/tmp` 是 512M tmpfs，整包解压会满）。注：第 7 轮行号引自 GitHub raw，本轮按轮子实测行号为准，逻辑一致；另，第 7 轮简报里给的路径前缀 `layers/quantization/utils/kernels/linear/` 不准，0.34.0 轮子里实际是 `model_executor/kernels/linear/`（第 7 轮 devlog 正文用的已是正确前缀，结论不受影响）。

### 1. 模型量化口径：per-tensor 权重 + 动态 per-token 激活（第 7 轮已核实）

`Qwen/Qwen3.8-27B-FP8` 的 `config.json` 里 `quantization_config` 只有三项：

```json
{ "activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8" }
```

没有 `weight_block_size` 字段。稠密 `Fp8LinearMethod`（`vllm/model_executor/layers/quantization/fp8.py:276-284`），非 block 路径：

- `weight_quant_key = kFp8StaticTensorSym`（per-tensor 权重尺度）；
- `activation_scheme == "dynamic"` → `activation_quant_key = kFp8DynamicTokenSym`（动态 per-token 激活尺度；两条分支都落在这里）。

### 2. 尺度定义：per-token 不是 per-tensor

`vllm/model_executor/layers/quantization/utils/quant_utils.py`：

- `GroupShape.is_per_tensor()`（:126-127）：当且仅当 `row == -1 and col == -1`；
- `kFp8StaticTensorSym`（:198）：`GroupShape.PER_TENSOR`，即 (-1, -1) → `is_per_tensor()` 为 True；
- `kFp8DynamicTokenSym`（:210）：`GroupShape.PER_TOKEN`，即 (1, -1) → `is_per_tensor()` 为 False（`is_per_token()` 为 True）。

### 3. 候选表：FlashInfer 排第 1 位，CUTLASS 第 2 位

`vllm/model_executor/kernels/linear/__init__.py:459-471`，`_POSSIBLE_FP8_KERNELS[CUDA]` 按顺序是：

```python
FlashInferFP8ScaledMMLinearKernel,
CutlassFP8ScaledMMLinearKernel,
B12xTensorFP8ScaledMMLinearKernel,
PerTensorTorchFP8ScaledMMLinearKernel,
ChannelWiseTorchFP8ScaledMMLinearKernel,
MarlinFP8ScaledMMLinearKernel,
HummingFP8ScaledMMLinearKernel,
```

### 4. 选择逻辑：按表顺序取第一个双通过的

`vllm/model_executor/kernels/linear/__init__.py`：

- `is_supported_and_can_implement_kernel`（:651-673）：算力为空时取设备算力，`compute_capability = _cc[0] * 10 + _cc[1]`（H200 得 90）；先调 `kernel.is_supported(compute_capability)`，再调 `kernel.can_implement(config)`，任一不通过即排除；
- `choose_scaled_mm_linear_kernel`（:676-…）：按候选表顺序取第一个双通过的核；
- `init_fp8_linear_kernel`（:747-…）：非 block 路径走 `_POSSIBLE_FP8_KERNELS`；`module_name` 给出时记一条 `Selected %s for %s`（`fp8.py:351` 传入 `module_name=self.__class__.__name__`，即 `Fp8LinearMethod`）。

### 5. FlashInfer 的两道门（逐条核对，两道都拦）

`vllm/model_executor/kernels/linear/scaled_mm/flashinfer.py:39-65`，`FlashInferFP8ScaledMMLinearKernel`：

```python
@classmethod
def is_supported(cls, compute_capability=None):
    if not current_platform.is_cuda():
        return False, "requires CUDA."
    if not has_flashinfer():
        return False, "requires FlashInfer to be installed."
    if compute_capability is not None and compute_capability < 100:
        return False, "requires compute capability 100 and above."
    return True, None

@classmethod
def can_implement(cls, c):
    per_tensor_activation_scales = (
        c.activation_quant_key.scale.group_shape.is_per_tensor()
    )
    per_tensor_weight_scales = c.weight_quant_key.scale.group_shape.is_per_tensor()
    if not (per_tensor_activation_scales and per_tensor_weight_scales):
        return False, "requires per tensor activation and weight scales."
    return True, None
```

对本模型逐条核对：

| 门 | 条件 | 本模型 | 结果 |
|---|---|---|---|
| `is_supported` | 算力 ≥100 | H200 算力 90 | 不通过（"requires compute capability 100 and above."） |
| `is_supported` | FlashInfer 已安装 | 第 6 轮注意力后端已跑通 | 通过 |
| `can_implement` | 激活尺度 per-tensor | 动态 per-token | 不通过（"requires per tensor activation and weight scales."） |
| `can_implement` | 权重尺度 per-tensor | per-tensor | 通过 |

### 6. CUTLASS 为什么中选

`vllm/model_executor/kernels/linear/scaled_mm/cutlass.py:160-169`，`CutlassFP8ScaledMMLinearKernel`：

- `is_supported` 只检查 `cutlass_fp8_supported()`，与算力无关；
- `can_implement` 恒返回 `True`。

它是表里第一个双通过的核。Profiling 轮实测 GEMM 内核是 `cutlass_scaled_mm`，与此一致。

### 7. 为什么强制指定也没用

- `--linear-backend flashinfer_cutlass`：`_resolve_backend_kernels`（`__init__.py:418-449`）只对候选表做交集过滤（`flashinfer_cutlass` 映射集里含 `FlashInferFP8ScaledMMLinearKernel`）；筛完照样对每个候选跑同一对 `is_supported` / `can_implement` 检查，两道门一个都绕不过去。过滤结果为空时还带警告回落到自动选择。
- `force_kernel` 参数不对用户开放（只在 ROCm 的 batch-invariant 路径内部用），且即使传了也跑同一对检查。
- `VLLM_DISABLED_KERNELS` 只能禁用，不能强制。
- 改 `activation_scheme` 不是开关：那会改变量化语义，等于重新做检查点，需重做权重并重跑全部 4 集精度对照，比赛只剩 4 天，不推荐。

### 8. 旧日志佐证：取不到，不影响结论

引擎启动时会记 `Selected CutlassFP8ScaledMMLinearKernel for Fp8LinearMethod`，但 `modal app logs` 只回最近 100 行，内核选择发生在模型加载早期，已超出可取窗口——第 6 轮 App `ap-smGNMeauqSrWhxYDsM86PL` 的日志只剩最后 100 行，无内核选择行。结论不依赖日志，源码已足够。

### 9. 比赛机（Blackwell SM120，算力 120）也一样

第一道门（算力 ≥100）在 SM120 上能过，但第二道门（激活 per-token）与硬件无关，仍不过；CUTLASS 照样是第一个双通过的。排在后面的 B12x 等够不到（选择按顺序取第一个通过的）。**GEMM 内核在 H200 与比赛机上都是 CUTLASS。**

## 结论

- GEMM 侧彻底收口：本检查点（动态 per-token 激活 + per-tensor 权重）在 vLLM 0.34.0 里只能走 CUTLASS，H200 与 Blackwell 皆然；FlashInfer 线性核与 DeepGEMM 两条路都已证伪（第 7 轮 + 本轮）。
- 唯一理论上的改法是把检查点重做成静态 per-tensor 激活（或 block FP8）再重跑全部精度对照——比赛只剩 4 天，不推荐。
- 冷 TTFT 现状：FlashInfer 注意力后端 41.09/43.8 秒是最好成绩，仍差 15 秒目标约 2.8 倍；剩余缺口只能看 P90 口径或接受现实。
