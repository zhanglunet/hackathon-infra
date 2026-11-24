# 算力优化黑客松自测报告（模板骨架）

> 状态：**骨架（2026-11-24 建，\$0）**。所有带 `TBD` 的数据格子须在 4 台比赛机（`4 × 8×RTX 6000D-84G`）上用**评审组统一下发的官方评测包（190 题精度集 + AIPerf 压测脚本）**实跑后填入，**不得用早期代理集或假服务数字冒充正式自测结果**。
>
> 最终提交须对齐评审组下发模板与最终锁定 SLO。

---

## 1. 参赛模型与 32 卡分配预案

| 角色 | 模型 | 权重精度 | 拟定卡数分配（4 台 × 8 卡 = 32 卡） | 部署拓扑（初案，压测后定稿） | 服务入口 |
|---|---|---|---|---|---|
| 必选 | `DeepSeek-V4.1-Flash-1017` | 官方 MXFP4/MXFP8（~511 GB） | 节点 1（或节点 1+2：2 套 TP=8 经前缀粘滞路由） | 8 卡 TP + Engram CPU Offload（对照 DEP8） | `http://<node>:8000` |
| 自选 1 | `Qwen3.8-27B-FP8` | 官方 FP8（~30.9 GB） | 节点 3（8 卡：8×单卡副本 + `prefix_router.py` 或 `DP_SIZE=8`） | 272K（YaRN）+ FP8 KV + FlashInfer + 前缀粘滞路由 | `http://<node>:8000` |
| 自选 2 | `GLM-5.3-Flash` | 官方 FP8（~328 GB） | 节点 4（8 卡 TP=8，对照 DEP8） | 272K（原生 1M）+ FP8 KV + `glm47` parser + SM120 MoE 调优表 | `http://<node>:8000` |

> 注：如果评测逐模型分时段验收（而非三模型同时常驻），单模型可独占全部 4 台机（32 卡）做多节点副本 + 前缀粘滞路由冲极限 Goodput；按组织方复测规则最终定案。

---

## 2. 精度测试结果（官方 190 题，损失门限 ≤ 2%）

| 模型 | AIME 2025 (30) | GPQA Diamond (50) | LiveCodeBench Lite (40) | BFCL v3 (50) | 综合 (190) | 相对官方原版损失 | 截断率 | 结论 |
|---|---|---|---|---|---|---|---|---|
| `DeepSeek-V4.1-Flash-1017`（基线） | TBD | TBD | TBD | TBD | TBD | — | TBD | TBD |
| `DeepSeek-V4.1-Flash-1017`（优化后） | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `Qwen3.8-27B`（官方 BF16 基线） | TBD | TBD | TBD | TBD | TBD | — | TBD | TBD |
| `Qwen3.8-27B-FP8`（272K + YaRN） | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `GLM-5.3-Flash`（基线） | TBD | TBD | TBD | TBD | TBD | — | TBD | TBD |
| `GLM-5.3-Flash`（优化后） | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

---

## 3. 四档 SLO 性能摸高与极限 Goodput（AIPerf）

> SLO 底线（以评审组最终锁定公示为准）：
> - **智能体（agent）**：50K/1K，缓存命中 0.70，P90 TTFT ≤ 15s，P90 TPOT ≤ 75ms，阶梯 8/16/32/64/80
> - **编程（coding）**：100K/2K，缓存命中 0.90，P90 TTFT ≤ 15s，P90 TPOT ≤ 75ms，阶梯 8/16/32/64/80
> - **长文本生成（longgen）**：1K/8K，缓存命中 0，P90 TTFT ≤ 10s，P90 TPOT ≤ 75ms，阶梯 8/16/32/64/80
> - **高压稳定（stress）**：32K/1K，缓存命中 0，P90 TTFT ≤ 20s，P90 TPOT ≤ 75ms，阶梯 8/16/32/64

### 3.1 DeepSeek-V4.1-Flash-1017

| 场景 | 摸高并发 | P90 TTFT (s) | P90 TPOT (ms) | 错误率 | Goodput (req/s) | 输出吞吐 (tok/s) | 实测缓存命中 |
|---|---|---|---|---|---|---|---|
| 智能体（agent） | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 编程（coding） | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 长文本（longgen） | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 高压（stress） | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 3.2 Qwen3.8-27B-FP8

| 场景 | 摸高并发 | P90 TTFT (s) | P90 TPOT (ms) | 错误率 | Goodput (req/s) | 输出吞吐 (tok/s) | 实测缓存命中 |
|---|---|---|---|---|---|---|---|
| 智能体（agent） | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 编程（coding） | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 长文本（longgen） | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 高压（stress） | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 3.3 GLM-5.3-Flash

| 场景 | 摸高并发 | P90 TTFT (s) | P90 TPOT (ms) | 错误率 | Goodput (req/s) | 输出吞吐 (tok/s) | 实测缓存命中 |
|---|---|---|---|---|---|---|---|
| 智能体（agent） | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 编程（coding） | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 长文本（longgen） | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 高压（stress） | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

---

## 4. 关键硬件与软件环境核对（真机填写）

- 显卡型号（`torch.cuda.get_device_name(0)`）：`TBD`
- 驱动 / CUDA / nvcc / torch / vLLM：`TBD`（目标：CUDA 13.0 + torch 2.14.0+cu130 + vLLM 0.34.0）
- 关键优化摘要：详见 [ENVIRONMENT.md](../ENVIRONMENT.md)

## 5. 提交三件套核对清单

- [ ] 自测报告：3 模型官方 190 题精度 + 四档 AIPerf 压测全部填实、无 `TBD`
- [ ] 推理服务接入信息：3 模型服务地址或容器镜像（`Dockerfile` 真机构建验证通过）+ 启动脚本，全部通过 `smoke-test.sh`、272K 上下文、原生 `tools`、`/metrics`
- [ ] 环境与优化说明表：`ENVIRONMENT.md` 中 `T1–T7` 全部清零
