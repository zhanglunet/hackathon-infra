# Modal H200 冷 TTFT 调优第 7 轮：DeepGEMM 代码核实——不适用于本模型，停手

- 日期：2026-11-23
- 部署：无（本轮只做源码核实，未上 GPU）
- 模型：Qwen3.8-27B-FP8（稠密），vLLM 0.34.0，Modal H200 141GB（未动用）
- 费用：**$0**；余额仍 $9.61（2026-11-22 16:23 核）

## 结论先行

- **DeepGEMM 不适用于本模型，一分钱 GPU 都没烧。** 第 6 轮日志里的 `DeepGEMM kernels are disabled: DeepGEMM's JIT needs nvcc 12.9 or newer, found 12.8` 只是“DeepGEMM 不可用”的警告，不等于“本模型在用它”——换 13.0 镜像消掉警告也改变不了任何 GEMM 内核选择。
- **GEMM 侧（Profiling 轮约 13 秒，占冷请求 GPU 时间约 25%）在 vLLM 0.34.0 内已无可直接开的开关。** 本模型稠密线性层跑的是 CUTLASS `cutlass_scaled_mm`，这是 per-tensor FP8 在 H200 上唯一可用的在树内核。
- 冷 TTFT 调优线就此收口：后端对照已穷尽（第 5/6 轮）、编译路线已结案（第 4 轮）、GEMM 无开关（本轮）。剩余缺口只能看 P90 口径或接受现实，见 §5。

## 代码核实：证据链（vLLM 0.34.0，tag v0.34.0）

源码经 GitHub raw 逐文件读（本地 `pip download --no-binary :all: vllm==0.34.0` 因元数据准备子进程报错失败，改走线上源码；以下行号均为 tag v0.34.0）。

### 1. 模型量化口径：per-tensor FP8，没有 block 尺寸

`Qwen/Qwen3.8-27B-FP8` 的 `config.json` 里 `quantization_config` 只有三项：

```json
{ "activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8" }
```

**没有 `weight_block_size` 字段。** 因此 `Fp8Config.from_config`（`vllm/model_executor/layers/quantization/fp8.py:292-326`）读到的 `weight_block_size` 是 `None`，`Fp8LinearMethod.block_quant = False`（`fp8.py:491`）。

### 2. 激活量化键：per-token，不是 per-group

`Fp8LinearMethod.__init__`（`fp8.py:496-533`），非 block 路径：

- `weight_quant_key = kFp8StaticTensorSym`（per-tensor 权重尺度）；
- 动态激活 + `cutlass_fp8_supported()` 为真 → `activation_quant_key = kFp8DynamicTokenSym`（per-token 激活尺度）。

per-token 不是 per-group——这决定了下面走哪条候选表。

### 3. 内核选择：per-tensor 候选表里根本没有 DeepGEMM

`init_fp8_linear_kernel`（`vllm/model_executor/kernels/linear/__init__.py:1476`）按 `activation_quant_key.scale.group_shape.is_per_group()` 分叉：

- per-group（block）→ `_POSSIBLE_FP8_BLOCK_KERNELS`；
- 否则（本模型）→ `_POSSIBLE_FP8_KERNELS`。

CUDA 的 `_POSSIBLE_FP8_KERNELS`（`__init__.py:898-914`，按优先级排序）全文是：

```python
FlashInferFP8ScaledMMLinearKernel,
CutlassFP8ScaledMMLinearKernel,
B12xTensorFP8ScaledMMLinearKernel,
PerTensorTorchFP8ScaledMMLinearKernel,
ChannelWiseTorchFP8ScaledMMLinearKernel,
MarlinFP8ScaledMMLinearKernel,
HummingFP8ScaledMMLinearKernel,
```

**没有 DeepGEMM 内核。** DeepGEMM 只出现在 `_POSSIBLE_FP8_BLOCK_KERNELS`（`__init__.py:972-990`，`DeepGemmFp8BlockScaledMMKernel` 排第 2 位）和 `"deep_gemm"` 的 `--linear-backend` 映射里（`__init__.py:678-684`，`DeepGemmFp8BlockScaledMMKernel` + `DeepGemmMxfp8BmmLinearKernel`）。`--linear-backend deep_gemm` 也救不了：后端过滤是对 per-tensor 候选表做交集，交集为空，无核可选。

### 4. DeepGEMM 内核自身的门：本模型两项都不满足

`DeepGemmFp8BlockScaledMMKernel.can_implement`（`vllm/model_executor/kernels/linear/scaled_mm/deep_gemm.py:101-155`）：

- 要求 `act_quant_desc.group_shape == GroupShape(1, 128)`——“Supports only dynamic per token group activation quantization with group_shape=(1,128)”；
- `process_weights_after_loading`（同文件 `:164`）直接断言 `layer.weight_block_size is not None`。

本模型激活是 per-token、权重无 block 尺寸，两项全不满足。它继承的是 `Fp8BlockScaledMMLinearKernel`，从根上就是 block 路的核。

### 5. `Fp8LinearMethod.use_deep_gemm` 是死属性

`fp8.py:480-486`：

```python
if self.quant_config.use_deep_gemm is not None:
    self.use_deep_gemm = self.quant_config.use_deep_gemm
else:
    self.use_deep_gemm = is_deep_gemm_supported()
```

这个属性在 `fp8.py` 全文只出现四处（`fp8.py:252` 定义、`480/482/486` 赋值），**从未被读取**。0.34.0 里稠密线性路的 DeepGEMM 开关是死的，真正的选择只走 `init_fp8_linear_kernel`（`fp8.py:661` 调用处）。

### 6. 第 6 轮那句警告的真正含义

警告出自 `is_deep_gemm_supported()`（`vllm/utils/deep_gemm.py:274-326`）：Hopper/Blackwell 架构 + `VLLM_USE_DEEP_GEMM` + `deep_gemm` 包在位 + nvcc 存在 + **nvcc >= 12.9**。第 6 轮镜像 nvcc 12.8，卡在最后一关，所以报了那句警告。

但这只影响“DeepGEMM 是否可用”，不影响“本模型是否走 DeepGEMM”——后者由上面第 3 条的候选表决定，与 nvcc 版本无关。换 13.0 镜像后 `is_deep_gemm_supported()` 会返回 True，稠密线性仍走 per-tensor 表，GEMM 内核仍是 CUTLASS。

### 7. 旁证：Profiling 轮的实测与源码链一致

Profiling 轮 torch profiler 实测 GEMM 内核是 `cutlass_scaled_mm`（1280 次调用，约 13 秒）——正是 per-tensor 路选出的 `CutlassFP8ScaledMMLinearKernel`。实测与源码选择链互相印证。

## GEMM 侧还剩什么

- **vLLM 0.34.0 内，CUTLASS 是本模型在 H200 上唯一可用的 per-tensor FP8 稠密内核。** 候选表里排在它前面的 `FlashInferFP8ScaledMMLinearKernel` 未被选中（`is_supported`/`can_implement` 未通过，原因未核实）——这是唯一未核实的在树内候选；若要追，需另起一轮读引擎日志的失败原因（或以 `--linear-backend flashinfer_cutlass` 强制后看报错），且 FlashInfer 的 FP8 线性核未必比 CUTLASS 快，期望要低。`B12xTensorFP8ScaledMMLinearKernel` 排在 Cutlass 之后同样未被选中；Triton FP8 不在 CUDA per-tensor 表里；Marlin 是 weight-only 路，不适用 W8A8 动态激活。
- **结构性改法**：把权重重做成 block FP8（`weight_block_size` 128×128）才能进 DeepGEMM 的 block 路——要重做 27B 权重量化、重跑全部 4 集精度对照，且 DeepGEMM block 核在 Hopper 上的收益未核实；比赛只剩 4 天，不推荐。
- **MoE 路与本模型无关**：模型是稠密 64 层，无 MoE 层；DeepGEMM 的 grouped 内核（MoE 专家层）用不上。`fp8_gemm_nt` 那个稠密包装器（`vllm/utils/deep_gemm.py:1115`）只被 block 核调用，per-tensor 路碰不到它。
- **结论**：GEMM 侧约 13 秒在 vLLM 0.34.0 内动不了；冷 TTFT 的剩余缺口（41 秒 vs 15 秒目标）须换别的口径。

## 费用

- 本轮 $0（未部署、未跑 GPU）。余额 $9.61 不变（2026-11-22 16:23 核）。

## 下一步建议

1. **冷 TTFT 调优线收口**：后端穷尽（第 5/6 轮）、编译结案（第 4 轮）、GEMM 无开关（本轮）。不再为 GEMM 换镜像烧钱。
2. **P90 口径**：真实门槛取决于组织方长度分布，未拿到；272K/15 秒是最严读法。
3. **比赛机复测**：报名 11-22 24:00 已截止，机器分配后在 Blackwell 真机上复测 FlashInfer 配置。
4. 若非要再挖 GEMM，唯一小尾巴是查 `FlashInferFP8ScaledMMLinearKernel` 未被选中的原因（须上 GPU 读日志，一轮约 $0.60），期望低。
