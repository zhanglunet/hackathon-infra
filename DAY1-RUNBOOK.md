# Day-1 Runbook：R2 首日 11-30（周一）——2+2 双线版

> 适用：AI 赋能小队（T09），R2 轮次 11-30 9:00 至 12-3 18:00。
> 时间一律按北京时间（CST）写。组长宁宇飞拿机器账号，机器开赛才发。
> **没有调试日**：11-30 9:00 直接开赛，Day-1 必须一次把环境立起来。
> **分线**：宁宇飞 11-25 14:18 已拍板，vLLM 与 SGLang 两条线**按空间分、
> 一边两台同步跑**（vLLM 线 Node 1-2，SGLang 线 Node 3-4）。
> 规则允许 24 小时跑，夜间可挂长任务；但**工作日 GPU 空闲超 8 小时
> 会被收回算力并扣分**，卡住时先切到另一件 GPU 活，别让卡空转。

## 0. Day-1 目标（收工时必须达成的三条）

1. 四台机全部验收通过（网络、GPU、磁盘、NFS 权重、docker）。
2. 至少一个模型在真机上点火成功（优先 GLM，DeepSeek 随后）。
3. SGLang A/B 门有结论（Gate 1，21:00 两线对数裁决：投跨机 CP，
   还是改走 CP + 前缀缓存）。

达不成第 3 条也行，但第 1、2 条是底线；第 3 条的数第二天还能补。

## 0.5 11/29 提前进场？（只有一种情况才有 Day-0）

- **默认没有调试日**：宁宇飞 11-25 11:06 已向组织方确认，
  只有第一批有提前一天熟悉环境的调试日，后续批次没有。
  R2 是 11-30 9:00 直接开赛、机器到时才发。
- **例外**：如果组织方因前轮机器提前释放、单独联系组长
  协商提前开赛（组织方口径：未沟通不强改），拿到机器的当天
  就按下面 Day-0 清单做，不要等 11-30：
  - 全员第一时间验 SSH 连通（办公网、安全区、出差外网都验；
    R1 安全队网络不通是前车之鉴）。
  - 双镜像分发到 4 台（见 §1.5），落本地 NVMe（`/data/nvme1n1`）。
  - RoCE 绑定（见 §3 的 10:30–11:00 步），`NCCL_IB_HCA` 写死，
    跑 `NCCL all_reduce_perf` 验满速。
  - 评测容器冒烟：官方评测容器能访问推理端点、输出落 home 目录。
  - NFS 权重核对：DeepSeek 48 分片、GLM index.json、Qwen 66 分片。
- **没有提前进场时**，Day-0 的活全部并进 11-30 上午（§3 的
  09:00–11:00），Day-1 的 DS 赛马从午后开始，不要按
  “11/29 已做完环境”的假设排时间。

## 1. 赛前已备好的东西（不用再找）

| 东西 | 位置 | 说明 |
|---|---|---|
| 首检脚本 | `scripts/day1-check.sh` | 9 节只读检查，见第 3 步；含 RoCE ACTIVE 口列出 |
| MoE 调优脚本 + 已打补丁的副本 | `scripts/tune_moe_sm120.sh`、`scripts/moe_tuner/` | 离线可用，不用再从 GitHub 拉 |
| SGLang SM120 补丁（25 个文件） | `sglang-patches/` | 修复 1–6、启动参考、验证脚本；**未应用、未上机** |
| SGLang 镜像构建 | `sglang-patches/Dockerfile` | v0.5.23-cu130 + 修复 1–6，见 §1.5 |
| vLLM 镜像构建 | `Dockerfile`（仓库根目录） | vLLM 0.34.0 + nvcc + FlashInfer，见 §1.5 |
| vLLM 启动脚本 | `serve-deepseek.sh`、`serve-glm.sh`、`serve-qwen27b.sh` | 均未在真机验证过；都认 `MODEL_DIR` 环境变量 |
| 前缀路由器 | `router/prefix_router.py` | 纯标准库，本地实测命中率 agent 0.65 / coding 0.87 |
| 压测框架 | `loadtest/` | 四档场景已按规则写好；只在假服务上自测过 |
| 本 runbook | `DAY1-RUNBOOK.md` | 本文件 |

## 1.5 赛前预打包双镜像（11/25–28，现在就能做）

- 目标：11-30 上机直接 `docker load` 分发，不用当天现 build，
  把 Day-1 上午的 docker 风险提前消掉。
- vLLM 镜像：仓库 `Dockerfile`（CUDA 13.0 devel + vLLM 0.34.0 +
  nvcc + FlashInfer），另加 `rdma-core` / `libibverbs` /
  `ibverbs-utils`（跨机 PD/CP 要用，见 §3 RoCE 绑定）。
- SGLang 镜像：`sglang-patches/Dockerfile`（`lmsysorg/sglang:v0.5.23-cu130`
  + 修复 1–6 + `--fp8-gemm-backend flashinfer_cutlass`），另加
  Mooncake / NIXL RoCE 传输依赖；`tuned_configs/` 的 device_name
  是 `NVIDIA_RTX_PRO_6000_Blackwell_Max-Q_Workstation_Edition`，
  比赛卡是 RTX 6000D，设备名不同不会自动匹配，须改名或重调
  （见 `sglang-patches/README.md`）。
- 做法：在有 docker 的机器上 `docker build` 后 `docker save` 成包，
  记下镜像 digest；11-30 上机后 `docker load` 分发到 4 台
  （组织方口径：镜像多机分发自理）。
- 赛前没条件打的：Day-1 仍按 §3 的 15 分钟规则现 build，
  15 分钟拉不到基础镜像就切 venv，不要死磕。

## 2. 关键数字速查

- 机器：4 台，每台 8x RTX 6000D 84GB；x86；内存约 1TB（组织方口径
  "190C1T"）；数据盘 `/data/nvme1n1`；Ubuntu 24.04.4 LTS。
- 互联：12x400G RoCE，只用 8 个（11-25 已确认）；**必须显式绑定
  能用的 8 个口**，见 §3 的 RoCE 绑定步。
- 权重（NFS 只读共享存储预置，挂载点未知，现场找）：
  DeepSeek-V4.1-Flash-1017 约 511GB、48 分片；
  GLM-5.3-Flash 约 328GB（index.json 核对）；
  Qwen3.8-27B-FP8 约 31GB、66 分片。
- 规则：4 台一起、一个模型一个模型轮流测，组织方折算单台值；
  MiniMax 只能用一台（我们不选它，不用管）。
  测试标准说明原文明确允许 PD 分离（“各队可自由选择单机多实例、
  多机分布式并行或 PD 分离等部署架构”，成绩按 4 台机器总有效产能），
  **跨机 PD/CP 书面已放行**。
- SLO：TTFT P90 15 秒（272K 单请求按最严读法），TPOT 75 毫秒；
  精度相对官方权重损失不超 2%。
  **待核**：官方题库总数写 190，四集 30+50+40+50 合计 170，
  差 20 道，是组织方文档自身不一致（2026-11-25 已导出原文核实），
  已问组织方，等回复；自测仍按 190 题口径写，实际以评审组下发的
  固化切片为准。官方原版精度值也还没到（武振华 11-24 已问，未回）。
- 超时：DeepSeek 首次启动设 `VLLM_ENGINE_READY_TIMEOUT_S=3600`；
  SGLang 首次启动约 7 分钟。
- vLLM 线最好成绩：Qwen 冷 TTFT 41.09 秒（FlashInfer，H200），
  离 15 秒差约 2.8 倍；DeepGEMM 与 FlashInfer 线性核已代码核实
  不适用，**不要在 Day-1 重开 vLLM 调优轮**。

## 3. 时间表（11-30，CST）——上午四台共同做，下午分线

### 09:00–09:30 拿机器 + 首检（四台共同）

1. 组长（宁宇飞）收组织方私发的服务器地址与账号，转给队内。
2. 每台机 SSH 登录，先跑首检（四台都跑，约 5 分钟一台）：
   `bash scripts/day1-check.sh`
   首检查：系统、CPU/内存、GPU（8 张、单卡 84GB）、磁盘、网络
   （6 个端点）、docker、RoCE、权重、工具链。
3. **R1 教训先验**：网络连通性排在 GPU/磁盘前面验
   （R1 安全队 09:17 就报网络连不上；docker.io 不通是已知坑）。
4. 首检有 FAIL 先记下来，进第 5 节故障手册；不要跳过首检直接起服务。

### 09:30–10:30 环境：docker 或 venv，二选一（四台共同）

1. 先看 docker：`docker version`、`docker info` 是否配了国内镜像源。
   docker.io 不通时配国内源再 build（`Dockerfile` 注释里有
   PyPI 清华源的 build-arg 写法）。
2. **赛前已预打包（§1.5）**：直接 `docker load` 分发到 4 台，
   跳过 build 这步。
   **没预打包**：试 `docker build -t hackathon-vllm:0.34.0 .`。
   **15 分钟还拉不到基础镜像就停，切 venv 路线**，不要死磕：
   `bash setup-vllm.sh`（装 vLLM 0.34.0 + flashinfer + nvcc；
   nvcc 装不上也不报错，回落 FA3 还能跑，只是慢）。
3. 四台机都要有环境：第一台 build 出镜像后，用 `docker save` /
   `docker load` 分发到其余三台（组织方口径：镜像多机分发自理）；
   venv 路线则每台各跑一遍 `setup-vllm.sh`。
4. 记下每台机的环境结论（docker 通 / venv 通 / 哪个源能用），
   写进当日 devlog。

### 10:30–11:00 找权重 + RoCE 绑定（四台共同）

1. `mount -t nfs`、`df -h`、`ls /data/nvme1n1`，找到 NFS 挂载点。
2. 核对三模型：DeepSeek 48 分片、GLM 有 index.json、Qwen 66 分片
   （首检第 8 节已自动核，现场再人工看一眼）。
3. 定 `MODEL_DIR`（例如 `MODEL_DIR=/data/xxx`），之后所有
   `serve-*.sh` 都用它。**不要跑 `download-*.sh`**——权重已预置，
   下 511GB 纯属浪费时间。
4. 找不到权重或分片不对：先问组长/组织方，不要自己下权重。
5. **RoCE 绑定（必做，跨机 PD/CP 的前提）**：
   12 个 400G RoCE 口只有 8 个可用，不显式过滤，NCCL / Mooncake /
   NIXL 极易探到不可用口导致跨机通信超时卡死，或静默回落到慢速 TCP。
   用 `ibv_devinfo` / `rdma link` 筛出 4 台各自 8 个 ACTIVE/UP 的
   RoCE 设备名，统一写进环境变量 `NCCL_IB_HCA`（并核对
   `NCCL_IB_GID_INDEX` 与 `NCCL_SOCKET_IFNAME`）；容器启动加
   `--network host --ipc host --device /dev/infiniband
   --ulimit memlock=-1`；跑 `NCCL all_reduce_perf` 验满速。
   首检第 7 节已列出 ACTIVE 口，现场照抄。

### 11:00 分线：vLLM 线 Node 1-2，SGLang 线 Node 3-4

**vLLM 线（Node 1-2）：GLM 冒烟 + DeepSeek 启动**

1. GLM 冒烟（第二款模型，第一次点火）：
   `MODEL_DIR=<上一步定的> bash serve-glm.sh`
   先看能不能起来、/health 与 /v1/models 是否正常；
   再用 `smoke-test.sh` 跑一个短请求。
   GLM 注意点：`--tool-call-parser glm47`（不能用 `glm`）、
   Blackwell 上 `--kv-cache-dtype fp8` 可用、原生 1M 上下文、
   272000 不用 YaRN。
   **冒烟加一项**：SM120 下 FP8 KV 对 BF16 KV 的正确性对照
   （vLLM 配方说 Blackwell 支持 FP8 KV，SGLang 社区说不可用，
   以真机实测为准；两条线都要测，数要对得上）。
2. 同时（另一台机或后台）跑 MoE 调优表生成：
   `MODEL_PATH=<GLM 路径> TP_SIZE=8 SAVE_DIR=./tuned_configs bash scripts/tune_moe_sm120.sh`
   调优要跑一阵子，挂后台；生成后启动时加
   `VLLM_TUNED_CONFIG_FOLDER=./tuned_configs`。
3. GLM 起不来：先看日志里是不是 parser、KV dtype、显存三类问题；
   不要改模型配置，先按 serve-glm.sh 注释里的核实项对。
4. 午饭时 DeepSeek 后台启动（511GB 加载 + JIT，慢）：
   `MODEL_DIR=<定的> bash serve-deepseek.sh`
   挂后台跑（日志写 `logs/`），timeout 已设 3600 秒。
   注意点：`--tokenizer-mode deepseek_v41` 与
   `--tool-call-parser deepseek_v41` 是必传项；
   默认 Engram CPU offload（每卡约 36.6GiB 权重驻留）。

**SGLang 线（Node 3-4）：DeepSeek 272K 边界摸底**

1. 装 SGLang v0.5.23（镜像 `lmsysorg/sglang:v0.5.23-cu130`，
   确认构建含 #45261；或 pip 装同版本），上 `sglang-patches/` 里
   的修复 1–6（说明见 `sglang-patches/README.md`）。
   先用 `tests/test_topk_v2_sm120.py` 验修复 5。
2. 单机（8 卡）跑 DeepSeek 272K 冷/热 TTFT、TPOT：
   从 chunk=1024 起逐档上探（1024 / 2048 / 4096 / 8192），
   与 `--mem-fraction-static` 联调，冷 TTFT P90 至少测 3 次。
   84GB 卡在 265K–272K 容易爆 indexer float32 OOM，
   摸到边界就记下来，不要硬顶。
3. 精度代理集（eval/ 四集，4096 口径先跑通链路即可），
   重点验 SWA bounded replay 开/关的精度差。

### 13:00–15:00 分线继续

- **vLLM 线**：DeepSeek 冒烟 + 272K 初测。先 /health、/v1/models，
  再短请求冒烟；用 `probe/cold_only.py`（仓库已有，H200 上跑过的
  同一探针），先测冷 1、冷 2、热 1 三个数。记数：冷 TTFT、热 TTFT、
  TPOT、KV 容量、显存占用，写进当日 devlog；同步更新
  ENVIRONMENT.md §6（性能 TBD 的第一行实测数）。
  DeepSeek 起不来或 272K OOM：先降 `MAX_NUM_SEQS`、再看
  Engram offload 是否生效；不要在 Day-1 改权重或重做量化。
- **SGLang 线**：继续 272K 边界摸底 + 四档 C=8/16 摸底
  （同一官方评测容器 / `loadtest/`，与 vLLM 线同口径），
  30 题精度快筛；Node 4 按需准备跨机 PD/CP 或 SGLang GLM 冒烟。

### 15:00 两线对表：SGLang A/B 门初判

1. 两线 15 分钟对数（同机实测表）：SGLang 单机 272K 冷 TTFT、
   OOM 状态、RoCE 是否满速。
2. **初判（11-24 已定，照此执行）**：
   - 单机 272K 冷 TTFT ≤20 秒：立刻投跨机 PD/CP，往 15 秒压。
   - 单机 >30 秒或 OOM：不在单机上继续调，改走 CP + 前缀缓存，
     思路回流 vLLM 线，Qwen 留在 vLLM 上只做通用优化。
   - 20–30 秒之间：先跑一遍跨机 CP 看能压多少，再定。
3. 初判不是终判：终判在 21:00 Gate 1（见 §4）。

### 15:00–17:00 分线按初判走

- **vLLM 线**：GLM 四档 C=8/16 摸底 + Qwen 上线准备
  （YaRN 272K + FlashInfer，测 RTX 6000D 真实冷/热 TTFT，
  对照 Modal 41.09 秒）；优化 `prefix_router.py` 并发吞吐
  与 /metrics 聚合，为 Day-2 扩容到 4 台做准备。
- **SGLang 线**：按初判走——投 CP 的搭跨机 PD/CP（1P1D，
  与 KV layout 一致性）；止损的精调单机大 chunk / KV 池 /
  Radix 前缀缓存上限，或转攻 GLM 侧翼对照。

### 17:00–18:00 router 验证 + 压测链路跑通（vLLM 线主导）

1. router：起 2 个 vLLM 副本（单机内先），
   `python3 router/prefix_router.py --backends http://127.0.0.1:8001,http://127.0.0.1:8002`
   验证前缀粘滞命中率与 /metrics 聚合。
2. 压测：先用内置引擎跑通 `bash loadtest/selftest.sh`
   （假服务，1–2 分钟，只验链路）；再看 AIPerf 能不能装
   （`pip install aiperf`），能装就用 AIPerf 引擎跑一档真数，
   装不上就先用内置引擎顶着，第二天再解决。
3. 跨机 router（4 台机各一个 TP8 副本 + 1 个入口）是 Day-2 的活，
   Day-1 只做到单机验证。

### 18:00 之后：夜间长任务（分线挂）

1. 18:00 是当天赛程结束时间，但规则允许 24 小时跑：
   挂上夜间任务再走。
   - vLLM 线 Node 1：官方评测容器跑 DeepSeek 完整 190 题精度；
     Node 2：挂跑 `scripts/tune_moe_sm120.sh` 生成 RTX 6000D 的
     DeepSeek 与 GLM MoE 调优表（耗时数小时，正好填满夜间）。
   - SGLang 线 Node 3-4：挂跑 SGLang 线 DeepSeek 完整四档阶梯
     （8/16/32/64）+ 稳定性 Soak 压测，同步跑 190 题全量精度，
     验证 bounded replay 安全余量。
2. 挂之前确认：任务是后台跑、日志有人看、GPU 不是空闲状态
   （空闲 8 小时规则只算工作日，夜间跑任务正好避开）。
3. 当日 devlog 当晚写完（验收结果、三个模型的冒烟结论、
   SGLang A/B 门结论、ENVIRONMENT.md 更新了哪几行）。

### 21:00 Gate 1：DeepSeek 初赛门（两线对数裁决）

- SGLang 84GB 下 272K 不 OOM 且冷 TTFT ≤20 秒 → SGLang 线
  Day 2 深入攻跨机 PD/CP 与高并发，vLLM 线保留为稳态对照。
- SGLang 直接 OOM 或冷 TTFT >30 秒 → 立刻止损不死磕单机，
  把 flashinfer_cutlass / 大 chunk 思路沉淀回 vLLM 线或转攻
  GLM / PD 分离。
- 每日对数节奏：09:30（5 分钟，对夜跑数）+ 21:00（15 分钟，
  过 Gate 与排夜班队列）。

## 4. 决策门汇总（贴在墙上）

| 时间点 | 决策 | 判据 |
|---|---|---|
| 09:45 | docker 还是 venv | docker build 15 分钟拉不到镜像就切 venv（预打包好的直接 load） |
| 10:30 | 分线 | vLLM 线 Node 1-2，SGLang 线 Node 3-4；严禁同机混跑 |
| 15:00 | SGLang A/B 门初判 | 单机 272K ≤20s 投跨机 CP；>30s 或 OOM 改 CP + 前缀缓存 |
| 17:00 | 压测引擎 | AIPerf 装得上就用它，装不上先用内置引擎 |
| 21:00 | Gate 1 终判 | 见 §3 末尾：≤20s 继续攻 PD/CP，>30s 或 OOM 止损 |
| 全天 | GPU 空闲 | 任何一台机空闲别超 8 小时（工作日规则），卡住就换活 |
| 全天 | 2 小时止损 | 任一线遇到超 2 小时未解的框架级 crash，立刻切回 safe-baseline-v1 |

## 5. 故障手册（R1 已踩过的坑 + 我们的预案）

- **网络连不上机器**：R1 安全队 09:17 就遇到。先确认 SSH、内外网
  互通，再看 GPU；找组织方（崔行兵）或组长协调。
- **docker.io 不通**：R1 10:26 已确认。改国内源；基础镜像
  `nvidia/cuda:13.0.0-devel-ubuntu24.04` 也要走源；再不行切 venv。
- **推理环境是不是自己配**：是。组织方 10:34 确认：含系统参数
  调优，全部自理。
- **权重找不到**：先 `mount`/`df` 找 NFS，不要自己下载；
  目录名与组织方公布的不一致时以现场实测为准，记下来。
- **RoCE 跨机通信超时或静默回落 TCP**：12 个口只有 8 个可用，
  没绑 `NCCL_IB_HCA` 最容易踩。先 `rdma link` / `ibv_devinfo`
  确认 ACTIVE 口，再核对 `NCCL_IB_HCA`、`NCCL_IB_GID_INDEX`、
  `NCCL_SOCKET_IFNAME` 三个变量；容器须加 `--device /dev/infiniband`。
- **SGLang 启动报 #44445**：candidate indexer 要 DeepGEMM paged
  sparse MQA logits；用 `sglang-patches/adapter/sitecustomize.py`
  的修复 3（`DS41_SM120_INDEXER_FIX=1`）。
- **SGLang prefill 只有约 294 tok/s**：修复 1 没上；打
  `runtime/flash_mla_sm120.patch` 再起。
- **DeepSeek 启动超时**：511GB 加载 + JIT 本来就慢，
  timeout 是 3600 秒；先看日志走到哪一步，不要反复重启
  （重启一次又要重加载）。
- **搞不完**：组织方口径是可申请往后延半天；轮次也可私下互换，
  须提前 1 天告知。先找组长商量，不要自己硬扛。
- **共享目录 / 提交物模板 / 压测脚本**：组织方说赛前统一提供，
  截至 11-25 还没发；Day-1 留意群里有没有发，有就按新的来。

## 6. Day-1 交付物（收工前打勾）

- [ ] 四台机首检结果（`day1-check.sh` 输出存档）
- [ ] 环境结论（docker/venv、哪个源能用、镜像分发到几台）
- [ ] MODEL_DIR 定了，三模型分片数核对过
- [ ] RoCE 绑定结论（8 个 ACTIVE 口、`NCCL_IB_HCA`、满速验证）
- [ ] GLM 冒烟结论（起来没有、短请求通不通、FP8 KV 对 BF16 KV）
- [ ] DeepSeek 冒烟 + 272K 冷/热 TTFT 初测数（两线各一份）
- [ ] SGLang A/B 门结论（Gate 1，21:00 裁决：投 CP / 改 CP + 前缀缓存）
- [ ] router 单机验证 + 压测链路跑通（内置或 AIPerf）
- [ ] 当日 devlog（`devlog/2026-11-30-day1.md`）+ ENVIRONMENT.md 更新

## 7. 分工建议（组长定）

- 组长（宁宇飞）：拿账号、对外协调（组织方、轮次、延期）、
  SGLang A/B 门与 Gate 1–3 拍板。
- vLLM 线（Node 1-2）：环境 + 权重 + GLM/DeepSeek 冒烟 +
  router + 压测链路；SGLang 线（Node 3-4）：SGLang 部署 +
  DeepSeek 272K 边界 + 跨机 PD/CP；两线并行，别串行等。
- 群里同步：当天关键结论贴本队群（只贴结论与数字，不贴账号密码）。

## 8. 不做的事（Day-1 别碰）+ 两条铁律

- 不要重开 vLLM 冷 TTFT 调优（第 4–8 轮已结案：后端穷尽、
  编译结案、GEMM 无开关）。
- 不要跑 BF16 32K 对照（用户已定先不跑；余额约 $5.27，红线 $4）。
- 不要自己下载权重（511GB + 328GB，NFS 已预置）。
- 不要在官方包没到之前写正式自测报告（只有代理集数）。
- 不要把账号密码贴群里、贴 devlog 里、贴仓库里。
- **铁律 1：交卷必须 4 台整群重跑。** 2+2 赛马数只做选型，
  写入报告的最终数据必须 4 台跑出。
- **铁律 2：同口径对跑。** 两线统一官方评测容器 + 随机防缓存标记，
  确保数据直接可比；合数不合代码，不强行合并两套代码，
  每模型按实测分数定胜者接管 4 台。

## 9. Day 2–4 大纲与 Gate 0–3（总览，细表以当日为准）

- **Gate 0 · 11/29 调试日（仅当提前进场）**：SSH 全通、32 卡 84G /
  1TB RAM / NVMe 就绪、双镜像分发完、官方评测容器通测、RoCE
  8 个 ACTIVE 口锁定并验满速。默认没有这一天（见 §0.5）。
- **Day 2 · 12/1（周二）**：vLLM 线拿下 Qwen + GLM 真机四档达标
  并合入 SM120 MoE 调优表；SGLang 线攻坚 DeepSeek 跨机 PD/CP
  与 GLM 侧翼对照。Qwen 对比：32xTP1 + prefix_router 对
  16xTP2 + 路由（TP2 可把长 prefill 近乎减半）。
  21:00 **Gate 2**：三模型全部封存及格版 `safe-baseline-v1`，
  按 2+2 同口径榜单逐个模型拍板 Day 3 由哪条线接管 4 台。
- **Day 3 · 12/2（周三）**：停止 2+2 拆分，全面合兵为官方
  “4 台整群服务单个模型”形态，由各模型胜出路线主导 4 台
  极限摸高（DeepSeek → Qwen → GLM 轮流跑全阶梯 8/16/32/64/80）。
  21:00 **Gate 3**：三模型最终 4 台配置冻结为 `rc-final-v1`，
  深夜全自动出正式成绩；12/3 严禁再引未经 190 题精度回归的新改动。
- **Day 4 · 12/3（周四）**：09:00–12:00 正式数据全收口
  （GLM 4 台出数 + 终版精度终验）；12:00–15:00 镜像固化 +
  共享目录 + 冷启演练，填齐《正式自测报告》与 ENVIRONMENT.md
  （T1–T7 清零）；15:00–17:30 双线交叉终审 + 在线表改状态 +
  提交三件套，代码打 tag `r2-submission-final`；17:30 硬封盘，
  18:00 收官清盘。
