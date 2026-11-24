# loadtest：四档场景 SLO 压测

负责人：xiaoyaxy。按比赛规则「二.3 性能标准」的四档场景和并发阶梯压测我们的推理服务，
判定 P90 SLO 是否过线，找出每档的「摸高」并发和 Goodput。

> 口径：这里测出的数是**自测**，最终以评审组复测为准。数字必须注明在哪台机器、什么部署方式下测的。
> Modal 上的数只算方法验证，不能当比赛机成绩。

## 为什么需要它

1. **计分就是看这个。** 规则是：四档场景各有一条 P90 底线 SLO，不过线直接不及格；
   过线后按「4 台机器在并发阶梯下的最高有效吞吐（Goodput 摸高）」排名。
   在这之前，仓库里没有任何一个按这套口径测出来的数。
2. **部署方案要靠它选。** 一台 8 卡机是起 1 个 TP=8 的大服务，还是 8 个单卡副本（`DP_SIZE=8`），
   `MAX_NUM_SEQS` 设 32 还是 80，chunked prefill 步长多大……这些没有标准答案，
   只能按比赛场景各压一遍，看谁过线、谁的 Goodput 高。
3. **交件要用。** 交件第 1 项「自测报告」要有性能部分，就是这里输出的 `report.md`。
4. **改完要回归。** 比赛前几天会不停改参数，每改一次就重跑相关档位，确认没有改坏。

评审组会用 NVIDIA AIPerf 测，所以这里默认也用 AIPerf 发请求，我们只在外面加了一层：
造数据（含缓存命中）→ 预热 → 按阶梯逐级跑 → 读 `/metrics` 算实际命中率 → 判定 → 出报告。

## 四档场景（`scenarios.py`，照抄规则）

| 场景 | 输入 / 输出 / 命中 | 并发阶梯 | 底线 SLO（P90） |
|---|---|---|---|
| agent 智能体 | 50K / 1K / 0.70（混合长度，最长 256K） | 8 / 16 / 32 / 64 / 80 | TTFT ≤ 15s，TPOT ≤ 75ms |
| coding 编程 | 100K / 2K / 0.9（混合长度，最长 256K） | 8 / 16 / 32 / 64 / 80 | TTFT ≤ 15s，TPOT ≤ 75ms |
| longgen 长文本生成 | 1K / 8K / 0 | 8 / 16 / 32 / 64 / 80 | TTFT ≤ 10s，TPOT ≤ 75ms |
| stress 高压稳定 | 32K / 1K / 0（定长） | 8 / 16 / 32 / 64 | TTFT ≤ 20s，TPOT ≤ 75ms |

规则注明 SLO 是「初始参考」，开赛前评审组会实测后锁定公示。公示值不同就只改 `scenarios.py`。

## 文件

| 文件 | 作用 |
|---|---|
| `scenarios.py` | 四档场景定义（长度、命中率、阶梯、SLO） |
| `make_datasets.py` | 生成压测请求：每场景每级并发一个 JSONL，外加预热文件 |
| `run_ladder.py` | 主入口：预热 → 压测 → 抓 `/metrics` → 判定 → 出报告 |
| `aiperf_engine.py` | 调 AIPerf 跑一级并发，并把它的输出换成统一字段 |
| `builtin_client.py` | 内置客户端（只用标准库）：AIPerf 装不上时的备胎，也用来和 AIPerf 互相核对 |
| `report.py` | 结果目录 → `report.md`；给多个目录时并排对比摸高 |
| `mock_server.py` | 假推理服务：没有 GPU 时调通整条链路（只用标准库） |
| `selftest.sh` | 一键自检：假服务 + 缩小版数据 + 每档并发 8，1～2 分钟，$0 |
| `common.py` | 共用工具：`/metrics` 解析、分位数、SLO 判定 |

## 数据是怎么造的

- **长度**：智能体和编程档按对数正态分布抽样、截断到 256K，均值对准 50K / 100K；
  长文本和高压档定长。给 `--tokenizer`（模型目录）时按真实 token 切；
  不给时按 3.6 字符/token 估算，报告里会注明「估算」。
- **缓存命中**：每级并发 C 造 C 个会话（想成 C 个同时干活的智能体或 C 个代码仓库）。
  同一会话的请求共享开头「命中率 × 请求长度」个 token，后面是各自独有的内容。
  正式压测前先把各会话的共享前缀发一遍（预热，不计成绩），所以理论命中率 ≈ 0.70 / 0.90。
  **实际命中率**用压测前后两次 `/metrics` 的增量算（`vllm:prefix_cache_hits_total ÷ vllm:prefix_cache_queries_total`），
  写在报告里。实际值明显低于目标，通常说明 KV 缓存装不下这么多会话、被挤掉了。这本身就是要调的东西。
- **文本**：本机 Python 标准库的源码和注释，是自然的代码 + 英文，不是随机乱码。
  规则明确禁止长文本档用乱码测速度，因为乱码会让投机解码的接受率失真。
- **输出长度**：默认带 `ignore_eos: true`，让每个请求都输出到设定长度（1K / 2K / 8K）。
- 不同并发级别用不同的会话，上一级留在缓存里的内容帮不了下一级。

和官方数据的差别（**未核实**，等评测包）：官方智能体档是「多轮编程轨迹」，命中来自多轮对话复用历史；
我们用「共享前缀 + 预热」近似。官方长文本档是 SPEED-Bench 固化 60 条。评测包下发后，
把官方数据转成同样的 JSONL（每行 `session_id` / `text` / `output_tokens`）放进 `data/<场景>/c<并发>.jsonl`，
其余流程不变；如果官方脚本本身就能一键跑完整阶梯，以官方脚本为准，本目录只保留批量对比和 `/metrics` 核对。

## 怎么用

### 0. 准备环境

- 只用内置引擎：Python 3.9+ 即可，不用装任何包。
- 用 AIPerf（默认，和评审组一致）：需要 Python 3.10+。

```bash
pip install aiperf
```

- 想按真实 token 切长度：

```bash
pip install tokenizers
```

先自检一下工具链（不用 GPU）：

```bash
bash loadtest/selftest.sh
```

```bash
ENGINE=aiperf bash loadtest/selftest.sh
```

### 1. 生成数据（一次即可，估算模式约 1 秒；给 tokenizer 时要先编码语料，慢一些。全量约 500 MB）

```bash
python3 loadtest/make_datasets.py --out loadtest/data --tokenizer /models/Qwen3.8-27B-FP8
```

### 2. 起服务（另一个终端），然后跑阶梯

```bash
python3 loadtest/run_ladder.py --url http://127.0.0.1:8000 --data loadtest/data --out loadtest/results/qwen-tp1 --tokenizer /models/Qwen3.8-27B-FP8 --label "Qwen FP8 单卡 TP=1，比赛机 A 卡 0"
```

跑完看 `loadtest/results/qwen-tp1/report.md`：最上面是每档的摸高并发和 Goodput，下面是每一级的明细。

### 3. 对比几种部署方案

```bash
python3 loadtest/report.py loadtest/results/qwen-tp1 loadtest/results/qwen-dp8 loadtest/results/qwen-tp2-dp4
```

## 什么情况怎么用

| 情况 | 做法 |
|---|---|
| **没有 GPU，想先调通脚本**（现在） | 起假服务：`python3 loadtest/mock_server.py --port 8000 --speedup 10`；数据加 `--scale 0.02` 缩小；跑阶梯加 `--engine builtin`。全流程约 1–2 分钟，$0。假服务的数字没有意义 |
| **比赛机刚到，先摸底** | 先只跑每档最低一级：`--concurrency 8`。看能不能过线、实测命中率对不对，再决定往上测 |
| **选部署方案（TP / DP / 副本数）** | 每种方案起一次服务，用同一份数据跑同样的档位，`--out` 用不同目录，最后 `report.py` 并排对比 |
| **调某个参数** | 只跑受它影响最大的档位，比如 chunked prefill 步长主要影响 agent / coding 的 TTFT：`--scenarios agent,coding` |
| **某档在某级不过线，想看再往上会怎样** | 加 `--no-stop-on-fail`（默认不过线就停，省时间） |
| **想看冷缓存下的表现** | 加 `--no-warmup` |
| **多个副本、前面没有负载均衡** | `--url http://h1:8000,http://h2:8000 --engine builtin`：同一会话固定发到同一个地址，保住缓存命中。**注意**：真要交件时，评审组只连一个地址，前面的负载均衡也必须按会话/前缀粘滞，否则命中率会掉 |
| **AIPerf 和内置客户端的数对不上** | 用同一份数据各跑一次同一级，对比 P90 TTFT / TPOT；差得多先查 tokenizer 和 `ignore_eos` 是否生效 |
| **官方评测包到了** | 见上一节「和官方数据的差别」：换数据、按官方 SLO 改 `scenarios.py`，以官方脚本结果为准 |

## 已经踩过、已经处理的坑

- **重跑同一份数据会虚高**：服务的前缀缓存里还留着上一次的内容，命中率和 TTFT 都会好看得不真实
  （本地实测：高压档目标命中 0，重跑时实测 0.94）。现在每次运行都给请求开头加一个随机标记，轮内共享前缀照常命中，跨轮不命中。
- **结果目录混用**：`--out` 指向已有结果的目录会被拒绝，要么换目录，要么加 `--force` 清空。
- **AIPerf 的 `session_id`**：自定义数据带 `session_id` 时，AIPerf 会把同一会话的多行拼成多轮对话、带上前面的回复，
  输入长度就不对了。所以转给 AIPerf 的数据只有 `text` 和 `output_length`（`aiperf_engine.py`）。
- 两个引擎对同一份数据的结果在本地假服务上相差约 2% 以内（P90 TTFT、P90 TPOT、Goodput）。

## 关于 AIPerf 的几个发现（0.14.0，读 `--help` 和源码，未在真机验证）

- 内置 SPEED-Bench 数据类型（`--custom-dataset-type speed-bench-throughput-1-k` 等）。规则里长文本档用的就是 SPEED-Bench，
  评审组很可能这样跑。**未核实**，评测包到手后对照。
- 内置智能体轨迹回放（`weka-trace`、`tracelab`、`--scenario agentx`）。官方「多轮编程轨迹」可能是这类格式。**未核实**。
- `--reset-kv-cache` 会在每轮前 POST `/reset_prefix_cache` 清缓存（vLLM 要开 `VLLM_SERVER_DEV_MODE=1` 才有这个接口）。
- vLLM 加 `--enable-prompt-tokens-details` 后，AIPerf 能按请求统计缓存命中 token。建议启动脚本加上，和 `/metrics` 互相核对。
- 多个 `--url` 时 AIPerf 只支持轮询（round-robin），不按会话粘滞，多副本时会打掉命中率；所以多副本直连时用 `--engine builtin`。

## 时间和费用估算

压测时间主要花在 prefill 上。粗略估算：一级并发要处理的 token ≈ 请求数 × 输入长度 × (1 − 命中率)。
默认每级请求数 = max(24, 3 × 并发)，比如 coding 档并发 80：240 × 100K × 0.1 ≈ 2,400 万 token 要 prefill，
还要加上预热的 80 × 约 9 万 token。真实耗时取决于机器的 prefill 速度，比赛机上第一次跑时记下来，
写进 devlog，后面再估。全部四档、全部级别跑一遍可能要数小时，所以**平时只跑需要的档位和级别**。

在 Modal 上跑要先按 CLAUDE.md 估算费用并告知，跑完立刻停部署。

## 判定口径

- 一级并发**过线** = P90 TTFT ≤ SLO 且 P90 TPOT ≤ SLO 且错误率 ≤ `--max-error-rate`（默认 0）。
- TTFT = 发出请求到收到第一个 token（思考内容 `reasoning` 也算 token）。
- TPOT = (最后一个 token 时刻 − 第一个 token 时刻) ÷ (输出 token 数 − 1)，按请求算，再取 P90。
- Goodput = 同时满足 TTFT、TPOT 两条 SLO 的请求数 ÷ 压测墙钟时间（req/s）。
- **摸高** = 从低到高连续过线的最高一级。
- 评审组的确切计分公式（四档、三模型怎么合成总分）规则里没写，**未核实**。
