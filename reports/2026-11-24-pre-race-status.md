# 赛前筹备进度同步（2026-11-24）

> 正式比赛：**2026-11-27（周五）14:30**，剩余约 3 天。
> 当前代码基线：`main @ 5eec61f`（全部不依赖真机的 \$0 准备工作已收口）。

---

## 1. 参赛三款模型已拍板

| 角色 | 模型 | 权重规模（FP8/官方） | 关键启动配置（核对自 vLLM 0.34.0 官方 recipe） | 状态 |
|---|---|---|---|---|
| **必选** | `DeepSeek-V4.1-Flash-1017` | ~511 GB / 48 分片（含 183 GiB Engram） | `TP=8`、`MAX_MODEL_LEN=272000`（原生 1M）、`--tokenizer-mode deepseek_v41`、`deepseek_v41` tool/reasoning parser、默认 `ENGRAM_CPU_OFFLOAD=true` | 脚本就绪，待比赛机 |
| **自选 1** | `Qwen3.8-27B-FP8` | ~30.9 GB / 66 分片 | `MAX_MODEL_LEN=272000`（YaRN）、FP8 KV、FlashInfer、`qwen3_coder` / `qwen3` parser；8 卡机测多副本 + 前缀粘滞路由 / `DP_SIZE=8` | Modal 已排练，待 Issue #6 精度复核与真机压测 |
| **自选 2（已定）** | `GLM-5.3-Flash` | ~328 GB（原生 FP8） | `TP=8`、`MAX_MODEL_LEN=272000`（原生 1M）、FP8 KV（Blackwell）、`glm47` tool/reasoning parser、SM120 MoE 调优表 | 脚本就绪，待比赛机 |

---

## 2. 已落地的 \$0 赛前成果（全在 `main`）

1. **三模型下载与启动脚本齐备**：
   - `download-model.sh` / `serve-qwen27b.sh`（PR #3、#8、#10）：修完 P1 全部脚本 bug，补齐 `qwen3_coder` + `qwen3` parser、`DP_SIZE`、`--enable-prompt-tokens-details`、vLLM 缓存透传。
   - `download-deepseek.sh` / `serve-deepseek.sh`（PR #8、#10）：按官方 recipe 配齐 `deepseek_v41` tokenizer/parser、Engram CPU Offload、DSpark 推测解码、DEP8 开关。
   - `download-glm.sh` / `serve-glm.sh`（PR #12）：按官方 recipe 配齐 `glm47` parser、FP8 KV、MTP 与 `--moe-backend` 开关，按 `index.json` 动态核对分片数。
2. **四档 SLO 并发压测框架就绪（`loadtest/`，PR #9，xiaoyaxy）**：
   - 覆盖规则四档场景（agent 50K/0.70、coding 100K/0.90、longgen 1K/8K、stress 32K）× 并发阶梯（8/16/32/64/80），支持 AIPerf 与内置引擎交叉核对、自动判定 P90 摸高与 Goodput、`/metrics` 核对真实缓存命中率；本地 `selftest.sh` 已通过。
3. **多副本前缀粘滞路由就绪（`router/prefix_router.py`，PR #12）**：
   - 纯标准库 HTTP/SSE 路由器，按前 512 字符 SHA-256 一致性哈希把同前缀固定到同一后端副本并聚合 `/metrics`；本地双假服务实测 **agent 0.65 / coding 0.87** 命中率零损失、`smoke-test.sh` 4/4 通过。
4. **SM120（RTX 6000D）MoE 调优脚本就绪（`scripts/tune_moe_sm120.sh`，PR #12）**：
   - 响应 Issue #1 源码调研结论（SM120 不可用 MegaMoE，且 vLLM 0.34 缺 6000D MoE 预调优表）：自动打印 `torch.cuda.get_device_name(0)`、拉取 `v0.34.0` `benchmark_moe.py` 并为 `Glm5Next*` / `DeepseekV41*` 打补丁，一键生成 `VLLM_TUNED_CONFIG_FOLDER`。
5. **标准化容器镜像与交件骨架就绪**：
   - `Dockerfile`（CUDA 13.0 devel + Python 3.12 + vLLM 0.34.0 + FlashInfer，对齐 `torch 2.14.0+cu130` 并预配 `/cache/vllm` 缓存目录）。
   - `reports/self-test-report-template.md`（三模型 190 题精度表、四档 SLO 表、32 卡分配预案与交件清单）。
   - `eval/run_modal_p0_eval.sh` + `eval/check_truncation.py`（零空转 Modal P0 评测脚本与截断率/McNemar 分析器）。

---

## 3. 当前 3 个关键阻塞项

| 阻塞项 | 说明 | 负责人 / 动作 |
|---|---|---|
| **Issue #6（Modal GPU 接活）** | 跑 Qwen3.8-27B 的 P0-3（native BFCL）+ P0-1/P0-2（272K + YaRN + 32K FP8 vs BF16 精度对照）；Modal 余额约 \$9.54，全套预计 \$3.1–\$5.3 | Modal 机器拉最新 `main` 按 Issue #6 执行，跑完回传 |
| **比赛机到位（4 × 8×RTX 6000D）** | 预计需 2–3 天完成三模型下载、调优与全量压测，最晚须 11-25 到手 | 盯组织方群分配通知 |
| **官方评测包与规则锁定** | 官方 190 题固化切片、AIPerf 压测脚本、SLO 最终锁定值、提交模板与截止时间 | 收到后第一时间更新 `loadtest/scenarios.py` 与报告模板 |

---

## 4. 比赛机到手后的执行顺序（11-24 ~ 11-27）

1. **T+0h（拿到机器立刻并行开下权重）**：节点 1 下 DeepSeek（511 GB）、节点 3 下 Qwen（31 GB）、节点 4 下 GLM（328 GB）；同时跑 `scripts/tune_moe_sm120.sh --prep-only` 记录 RTX 6000D 设备名。
2. **优先打通必选模型 DeepSeek-V4.1-Flash**：跑通 `serve-deepseek.sh` + `smoke-test.sh` + 272K + 原生 `tools`，用 `loadtest/` 跑四档并发 8 摸底，再对照 `ENGRAM_CPU_OFFLOAD` / DEP8 / DSpark 与 SM120 MoE 调优表。
3. **验证 Qwen3.8-27B 与 GLM-5.3-Flash**：Qwen 在 8 卡机上对比「8×单卡副本 + `prefix_router.py`」与 `DP_SIZE=8` / TP 的 Goodput；GLM 跑 MoE 调优表与 `--moe-backend` 对照。
4. **官方评测包重跑与交件收口**：按官方包跑完三模型 190 题精度与四档 AIPerf 阶梯，填实 `reports/` 正式自测报告与 `ENVIRONMENT.md`（清零 `T1–T7`），确认容器镜像或常驻服务接入地址。
