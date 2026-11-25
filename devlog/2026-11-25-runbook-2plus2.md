# devlog：runbook 改成 2+2 双线版（2026-11-25）

## 背景

肖娅 2026-11-25 14:37 在本队群上传 R2 作战计划 v3.1（`ai-team-round2-battle-plan.html`，
2+2 赛马 Day 1–2、Day 3–4 优胜合流 4 台整群交卷、Gate 0–3）。我全文读完后对照
runbook 与清单页，列出 8 项调整清单，用户点选“按清单改，runbook 和清单页一起更新”。
本提交只做 runbook 与两个脚本；清单页 v3 另走 artifact.edit + 飞书重传。

## 改了什么

### DAY1-RUNBOOK.md（12,234 → 21,473 字节，全文重写）

1. **时间表改成 2+2**：09:00–11:00 四台共同做（首检、环境、找权重 + RoCE 绑定），
   11:00 分线——vLLM 线 Node 1-2（GLM 冒烟 + MoE 调优表并行、午饭时 DeepSeek
   后台启动），SGLang 线 Node 3-4（DeepSeek 272K 边界摸底，chunk 1024 起逐档上探）；
   13:00–15:00 分线继续；15:00 两线对表过 SGLang A/B 门初判；
   17:00–18:00 router + 压测链路（vLLM 线主导）；夜间长任务分线挂。
2. **Gate 1 改 21:00 裁决**：15:00 只做初判，终判在 21:00 两线对数时拍
   （≤20 秒继续攻 PD/CP，>30 秒或 OOM 止损回流）。
3. **§0.5 11/29 提前进场条件**：默认没有调试日（宁宇飞 11-25 11:06 已向组织方确认，
   后续批次没有）；只有组织方因前轮机器提前释放、单独联系组长协商提前开赛时，
   才按 Day-0 清单做（验 SSH、发镜像、RoCE 绑定 + 满速验证、评测容器冒烟、
   NFS 核对）；没有提前进场时，Day-0 的活全部并进 11-30 上午。
4. **§1.5 赛前预打包双镜像（11/25–28）**：vLLM 镜像（仓库 Dockerfile +
   rdma-core）与 SGLang 镜像（`sglang-patches/Dockerfile` + 修复 1–6 +
   Mooncake/NIXL 依赖），有 docker 的机器上 build + `docker save` 成包，
   11-30 上机 `docker load` 分发；赛前没条件打的，Day-1 仍按 15 分钟规则现 build。
   注：`tuned_configs/` 的 device_name 是 RTX PRO 6000，比赛卡是 6000D，
   须改名或重调（见 `sglang-patches/README.md`）。
5. **§9 Day 2–4 大纲与 Gate 0–3**：Gate 2（12-1 21:00 封 `safe-baseline-v1`）、
   Gate 3（12-2 21:00 冻 `rc-final-v1`）、Day 4（12-3 15:00 前写共享目录、
   17:30 硬封盘、tag `r2-submission-final`）；Qwen Day-2 对比
   （32xTP1 + prefix_router 对 16xTP2 + 路由）；每日 09:30（5 分钟）+
   21:00（15 分钟）对数。
6. **两条铁律写进 §8**：交卷必须 4 台整群重跑（2+2 的数只做选型）；
   同口径对跑（统一官方评测容器 + 随机防缓存标记，合数不合代码）。
   2 小时止损（任一线超 2 小时未解的框架级 crash 立刻切回 safe-baseline-v1）
   写进 §4 决策门汇总。
7. **RoCE 绑定步**（10:30–11:00）：`ibv_devinfo` / `rdma link` 筛出 8 个
   ACTIVE 口，写死 `NCCL_IB_HCA`（核对 `NCCL_IB_GID_INDEX`、
   `NCCL_SOCKET_IFNAME`），容器加 `--network host --ipc host
   --device /dev/infiniband --ulimit memlock=-1`，跑 `NCCL all_reduce_perf`
   验满速；故障手册加“跨机通信超时或静默回落 TCP”条目。
8. **GLM 冒烟加一项**：SM120 下 FP8 KV 对 BF16 KV 的正确性对照
   （vLLM 配方说支持、SGLang 社区说不可用，以真机为准，两条线都要测）。
9. **§2 关键数字两处更新**：
   - 跨机 PD/CP 书面已放行：测试标准说明原文“各队可自由选择单机多实例、
     多机分布式并行或 PD 分离等部署架构”，成绩按 4 台机器总有效产能。
   - 190 题待核：官方题库总数写 190，四集 30+50+40+50 合计 170，
     差 20 道，是组织方文档自身不一致（已导出原文核实），已问组织方，
     等回复；以评审组下发的固化切片为准。

### scripts/day1-check.sh（第 7 节）

- 加 ACTIVE RoCE 口列出（只读）：有 `rdma` 用 `rdma link show` 筛
  `state ACTIVE`，没有用 `ibv_devinfo`，两个工具都没有时提示手工筛。
  现场把这 8 个口抄进 `NCCL_IB_HCA`。

### serve-qwen27b.sh（第 115 行）

- 报错提示改掉：找不到 `config.json` 时不再说“请先运行 bash download-model.sh”，
  改为“请确认 MODEL_DIR 指向 NFS 只读权重根目录（比赛权重已预置，不要自己下载）”。
  第 150 行的 BF16 提示（`download-model.sh --bf16`）是 Modal 对照轮的旧文案，
  与比赛无关但不误导 Day-1 操作，本次未动。

## 核实经过

- **190/170**：2026-11-25 15:48 用 lark-cli 把《算力优化黑客松测试标准说明》
  （wiki `ICrCwffvviNWkOkvVOecWtGZnHC`）导出成 markdown 核对：原文 §二.2
  写“共 190 题”，下面四集 30+50+40+50 合计 170，差 20 道——组织方文档自身
  不一致，不是咱们也不是肖娅算错。导出文件存
  `~/workspace/goals/ai-infra/hidden_files/test-standard-20261125.md`。
- **PD/CP**：同一份原文 §一.1 明确允许 PD 分离等部署架构，§二.3 写成绩按
  “4 台机器总有效产能”算。群里再确认是锦上添花，不是前置条件。
- **serve 脚本 MODEL_DIR**：`serve-qwen27b.sh` 本就认 `MODEL_DIR` 环境变量；
  `serve-deepseek.sh` / `serve-glm.sh` 在 main 上（本地工作目录没有），
  按 11-25 环境对照结论“serve 只认 MODEL_DIR”。肖娅说的“改造 serve 脚本
  支持 MODEL_DIR 直连 NFS”在实质上已满足，不用改。
- **脚本验证**：`bash -n` 两脚本均过；本机试跑 `day1-check.sh` 第 7 节，
  输出符合预期（本机无 RDMA：WARN + “rdma link 未见 ACTIVE 口”，
  `rdma` 命令本身存在）。未上 GPU，$0。

## 未动

- 清单页 v3（另走 artifact.edit，改完重新导出 + 飞书重传）。
- runbook 的其余内容（首检、docker/venv 15 分钟规则、故障手册其余条目、
  交付物清单大部、Day-1 别碰五件事）原样保留。
