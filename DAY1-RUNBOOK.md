# Day-1 Runbook：R2 首日 11-30（周一）

> 适用：AI 赋能小队（T09），R2 轮次 11-30 9:00 至 12-3 18:00。
> 时间一律按北京时间（CST）写。组长宁宇飞拿机器账号，机器开赛才发。
> **没有调试日**：11-30 9:00 直接开赛，Day-1 必须一次把环境立起来。
> 规则允许 24 小时跑，夜间可挂长任务；但**工作日 GPU 空闲超 8 小时
> 会被收回算力并扣分**，卡住时先切到另一件 GPU 活，别让卡空转。

## 0. Day-1 目标（收工时必须达成的三条）

1. 四台机全部验收通过（网络、GPU、磁盘、NFS 权重、docker）。
2. 至少一个模型在真机上点火成功（优先 GLM，DeepSeek 随后）。
3. SGLang A/B 门有结论（投跨机 CP，还是改走 CP + 前缀缓存）。

达不成第 3 条也行，但第 1、2 条是底线；第 3 条的数第二天还能补。

## 1. 赛前已备好的东西（不用再找）

| 东西 | 位置 | 说明 |
|---|---|---|
| 首检脚本 | `scripts/day1-check.sh` | 9 节只读检查，见第 3 步 |
| MoE 调优脚本 + 已打补丁的副本 | `scripts/tune_moe_sm120.sh`、`scripts/moe_tuner/` | 离线可用，不用再从 GitHub 拉 |
| SGLang SM120 补丁（25 个文件） | `sglang-patches/` | 修复 1–6、启动参考、验证脚本；**未应用、未上机** |
| vLLM 启动脚本 | `serve-deepseek.sh`、`serve-glm.sh`、`serve-qwen27b.sh` | 均未在真机验证过 |
| 前缀路由器 | `router/prefix_router.py` | 纯标准库，本地实测命中率 agent 0.65 / coding 0.87 |
| 压测框架 | `loadtest/` | 四档场景已按规则写好；只在假服务上自测过 |
| 本 runbook | `DAY1-RUNBOOK.md` | 本文件 |

## 2. 关键数字速查

- 机器：4 台，每台 8x RTX 6000D 84GB；x86；内存约 1TB（组织方口径
  "190C1T"）；数据盘 `/data/nvme1n1`；Ubuntu 24.04.4 LTS。
- 互联：12x400G RoCE，只用 8 个（11-25 已确认）。
- 权重（NFS 只读共享存储预置，挂载点未知，现场找）：
  DeepSeek-V4.1-Flash-1017 约 511GB、48 分片；
  GLM-5.3-Flash 约 328GB（index.json 核对）；
  Qwen3.8-27B-FP8 约 31GB、66 分片。
- 规则：4 台一起、一个模型一个模型轮流测，组织方折算单台值；
  MiniMax 只能用一台（我们不选它，不用管）。
- SLO：TTFT P90 15 秒（272K 单请求按最严读法），TPOT 75 毫秒；
  精度相对官方权重损失不超 2%（官方 190 题与原版精度值**还没到**）。
- 超时：DeepSeek 首次启动设 `VLLM_ENGINE_READY_TIMEOUT_S=3600`；
  SGLang 首次启动约 7 分钟。
- vLLM 线最好成绩：Qwen 冷 TTFT 41.09 秒（FlashInfer，H200），
  离 15 秒差约 2.8 倍；DeepGEMM 与 FlashInfer 线性核已代码核实
  不适用，**不要在 Day-1 重开 vLLM 调优轮**。

## 3. 时间表（11-30，CST）

### 09:00–09:30 拿机器 + 首检

1. 组长（宁宇飞）收组织方私发的服务器地址与账号，转给队内。
2. 每台机 SSH 登录，先跑首检（四台都跑，约 5 分钟一台）：
   `bash scripts/day1-check.sh`
   首检查：系统、CPU/内存、GPU（8 张、单卡 84GB）、磁盘、网络
   （6 个端点）、docker、RoCE、权重、工具链。
3. **R1 教训先验**：网络连通性排在 GPU/磁盘前面验
   （R1 安全队 09:17 就报网络连不上；docker.io 不通是已知坑）。
4. 首检有 FAIL 先记下来，进第 9 节故障手册；不要跳过首检直接起服务。

### 09:30–10:30 环境：docker 或 venv，二选一

1. 先看 docker：`docker version`、`docker info` 是否配了国内镜像源。
   docker.io 不通时配国内源再 build（`Dockerfile` 注释里有
   PyPI 清华源的 build-arg 写法）。
2. 试 `docker build -t hackathon-vllm:0.34.0 .`。
   **15 分钟还拉不到基础镜像就停，切 venv 路线**，不要死磕：
   `bash setup-vllm.sh`（装 vLLM 0.34.0 + flashinfer + nvcc；
   nvcc 装不上也不报错，回落 FA3 还能跑，只是慢）。
3. 四台机都要有环境：第一台 build 出镜像后，用 `docker save` /
   `docker load` 分发到其余三台（组织方口径：镜像多机分发自理）；
   venv 路线则每台各跑一遍 `setup-vllm.sh`。
4. 记下每台机的环境结论（docker 通 / venv 通 / 哪个源能用），
   写进当日 devlog。

### 10:30–11:00 找权重，定 MODEL_DIR

1. `mount -t nfs`、`df -h`、`ls /data/nvme1n1`，找到 NFS 挂载点。
2. 核对三模型：DeepSeek 48 分片、GLM 有 index.json、Qwen 66 分片
   （首检第 8 节已自动核，现场再人工看一眼）。
3. 定 `MODEL_DIR`（例如 `MODEL_DIR=/data/xxx`），之后所有
   `serve-*.sh` 都用它。**不要跑 `download-*.sh`**——权重已预置，
   下 511GB 纯属浪费时间。
4. 找不到权重或分片不对：先问组长/组织方，不要自己下权重。

### 11:00–12:30 GLM 冒烟 + MoE 调优表（并行）

1. GLM 冒烟（第二款模型，第一次点火）：
   `MODEL_DIR=<上一步定的> bash serve-glm.sh`
   先看能不能起来、/health 与 /v1/models 是否正常；
   再用 `smoke-test.sh` 跑一个短请求。
   GLM 注意点：`--tool-call-parser glm47`（不能用 `glm`）、
   Blackwell 上 `--kv-cache-dtype fp8` 可用、原生 1M 上下文、
   272000 不用 YaRN。
2. 同时（另一台机或后台）跑 MoE 调优表生成：
   `MODEL_PATH=<GLM 路径> TP_SIZE=8 SAVE_DIR=./tuned_configs bash scripts/tune_moe_sm120.sh`
   调优要跑一阵子，挂后台；生成后启动时加
   `VLLM_TUNED_CONFIG_FOLDER=./tuned_configs`。
3. GLM 起不来：先看日志里是不是 parser、KV dtype、显存三类问题；
   不要改模型配置，先按 serve-glm.sh 注释里的核实项对。

### 12:30–13:00 午饭 + DeepSeek 后台启动

1. DeepSeek 启动（511GB 加载 + JIT，慢）：
   `MODEL_DIR=<定的> bash serve-deepseek.sh`
   挂后台跑（日志写 `logs/`），timeout 已设 3600 秒。
   注意点：`--tokenizer-mode deepseek_v41` 与
   `--tool-call-parser deepseek_v41` 是必传项；
   默认 Engram CPU offload（每卡约 36.6GiB 权重驻留）。
2. 午饭时让它加载，13:00 回来看。

### 13:00–15:00 DeepSeek 冒烟 + 272K 初测

1. 先 /health、/v1/models，再短请求冒烟。
2. 272K 冷/热 TTFT 初测：用 `probe/cold_only.py`（仓库已有，
   H200 上跑过的同一套探针），先测冷 1、冷 2、热 1 三个数。
3. 记数：冷 TTFT、热 TTFT、TPOT、KV 容量、显存占用，写进当日 devlog；
   同步更新 ENVIRONMENT.md §6（性能 TBD 的第一行实测数）。
4. DeepSeek 起不来或 272K OOM：先降 `MAX_NUM_SEQS`、再看
   Engram offload 是否生效；不要在 Day-1 改权重或重做量化。

### 15:00–17:00 SGLang A/B 门（Day-1 最关键的决策点）

1. 装 SGLang v0.5.23（镜像 `lmsysorg/sglang:v0.5.23-cu130`，
   确认构建含 #45261；或 pip 装同版本），上 `sglang-patches/` 里
   的修复 1–6（说明见 `sglang-patches/README.md`）。
   先用 `tests/test_topk_v2_sm120.py` 验修复 5。
2. 单机（8 卡）跑 DeepSeek 272K 冷/热 TTFT、TPOT、精度代理集
   （eval/ 四集，4096 口径先跑通链路即可）。
3. **判据（11-24 已定，照此执行）**：
   - 单机 272K 冷 TTFT ≤20 秒：立刻投跨机 PD/CP，往 15 秒压。
   - 单机 >30 秒或 OOM：不在单机上继续调，改走 CP + 前缀缓存，
     Qwen 留在 vLLM 上只做通用优化。
   - 20–30 秒之间：先跑一遍跨机 CP 看能压多少，再定。
4. 跨机 CP 的前提已确认（RoCE 400G x8 可用）；Day-1 能搭通
   就搭，搭不通记下来第二天继续，不要通宵死磕。

### 17:00–18:00 router 验证 + 压测链路跑通

1. router：起 2 个 vLLM 副本（单机内先），
   `python3 router/prefix_router.py --backends http://127.0.0.1:8001,http://127.0.0.1:8002`
   验证前缀粘滞命中率与 /metrics 聚合。
2. 压测：先用内置引擎跑通 `bash loadtest/selftest.sh`
   （假服务，1–2 分钟，只验链路）；再看 AIPerf 能不能装
   （`pip install aiperf`），能装就用 AIPerf 引擎跑一档真数，
   装不上就先用内置引擎顶着，第二天再解决。
3. 跨机 router（4 台机各一个 TP8 副本 + 1 个入口）是 Day-2 的活，
   Day-1 只做到单机验证。

### 18:00 之后：夜间长任务

1. 18:00 是当天赛程结束时间，但规则允许 24 小时跑：
   挂上夜间任务再走——精度代理集重跑、压测阶梯、SGLang soak。
2. 挂之前确认：任务是后台跑、日志有人看、GPU 不是空闲状态
   （空闲 8 小时规则只算工作日，夜间跑任务正好避开）。
3. 当日 devlog 当晚写完（验收结果、三个模型的冒烟结论、
   SGLang A/B 门结论、ENVIRONMENT.md 更新了哪几行）。

## 4. 决策门汇总（贴在墙上）

| 时间点 | 决策 | 判据 |
|---|---|---|
| 09:45 | docker 还是 venv | docker build 15 分钟拉不到镜像就切 venv |
| 15:00–17:00 | SGLang 投不投 | 单机 272K ≤20s 投跨机 CP；>30s 或 OOM 改 CP + 前缀缓存 |
| 17:00 | 压测引擎 | AIPerf 装得上就用它，装不上先用内置引擎 |
| 全天 | GPU 空闲 | 任何一台机空闲别超 8 小时（工作日规则），卡住就换活 |

## 5. 故障手册（R1 已踩过的坑 + 我们的预案）

- **网络连不上机器**：R1 安全队 09:17 就遇到。先确认 SSH、内外网
  互通，再看 GPU；找组织方（崔行兵）或组长协调。
- **docker.io 不通**：R1 10:26 已确认。改国内源；基础镜像
  `nvidia/cuda:13.0.0-devel-ubuntu24.04` 也要走源；再不行切 venv。
- **推理环境是不是自己配**：是。组织方 10:34 确认：含系统参数
  调优，全部自理。
- **权重找不到**：先 `mount`/`df` 找 NFS，不要自己下载；
  目录名与组织方公布的不一致时以现场实测为准，记下来。
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
- [ ] GLM 冒烟结论（起来没有、短请求通不通）
- [ ] DeepSeek 冒烟 + 272K 冷/热 TTFT 初测数
- [ ] SGLang A/B 门结论（投 CP / 改 CP + 前缀缓存 / 待定）
- [ ] router 单机验证 + 压测链路跑通（内置或 AIPerf）
- [ ] 当日 devlog（`devlog/2026-11-30-day1.md`）+ ENVIRONMENT.md 更新

## 7. 分工建议（组长定）

- 组长（宁宇飞）：拿账号、对外协调（组织方、轮次、延期）、
  SGLang A/B 门拍板。
- 建议一人跟环境 + 权重（首检、docker/venv、MODEL_DIR），
  一人跟模型冒烟（GLM、DeepSeek），一人跟 SGLang + router +
  压测；三路并行，别串行等。
- 群里同步：当天关键结论贴本队群（只贴结论与数字，不贴账号密码）。

## 8. 不做的事（Day-1 别碰）

- 不要重开 vLLM 冷 TTFT 调优（第 4–8 轮已结案：后端穷尽、
  编译结案、GEMM 无开关）。
- 不要跑 BF16 32K 对照（用户已定先不跑；余额约 $5.27，红线 $4）。
- 不要自己下载权重（511GB + 328GB，NFS 已预置）。
- 不要在官方包没到之前写正式自测报告（只有代理集数）。
- 不要把账号密码贴群里、贴 devlog 里、贴仓库里。
