# MoE 调优脚本（已打补丁，离线可用）

## 这个文件是什么

`benchmark_moe.py`，vLLM v0.34.0 官方源码里的 Triton MoE 调优基准脚本
（`benchmarks/kernels/benchmark_moe.py`），2026-11-25 从
`https://raw.githubusercontent.com/vllm-project/vllm/v0.34.0/benchmarks/kernels/benchmark_moe.py`
取回，sha256：

```
5ef0b600f672702eb6a8971631ab2d29111a21e7496b6052ec8ed574e3156187
```

## 打了什么补丁

原脚本的 `get_model_params` 不认识 `Glm5NextForCausalLM` /
`DeepseekV41ForCausalLM`，会错读 `num_local_experts` 报错
（"has no MoE experts"）。补丁把这四个架构名加进读 `n_routed_experts`
的分支（文件第 877 行附近）：

- `Glm5NextForCausalLM`
- `Glm5NextForConditionalGeneration`
- `DeepseekV41ForCausalLM`
- `DeepseekV41ForConditionalGeneration`

核实：`grep -c Glm5NextForCausalLM benchmark_moe.py` = 1，
`grep -c DeepseekV41ForCausalLM benchmark_moe.py` = 1。

## 怎么用

`scripts/tune_moe_sm120.sh` 会优先用这里的 bundled 副本，
不联网；删掉它才会重新从 GitHub 下载（再自动打补丁）。

```bash
# 只准备（不跑调优）
bash scripts/tune_moe_sm120.sh --prep-only

# 为 GLM-5.3-Flash 生成调优表
MODEL_PATH=/data/models/GLM-5.3-Flash TP_SIZE=8 SAVE_DIR=./tuned_configs \
  bash scripts/tune_moe_sm120.sh

# 启动时加载调优表
VLLM_TUNED_CONFIG_FOLDER=./tuned_configs bash serve-glm.sh
```

背景见 Issue #1 调研：vLLM 0.34.0 内置 Triton MoE 调优表没有
RTX 6000D 条目，会回落默认慢配置；SM120 用不了 DeepGEMM MegaMoE，
须在比赛机上自调这张表。
