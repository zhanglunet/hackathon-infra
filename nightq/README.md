# nightq：夜跑队列

晚上把要对比的配置排成一张清单挂上去，早上看一张表。纯标准库，只要 `python3` 和 `bash`。

## 为什么需要它

比赛机只有 4 天，DeepSeek 换一次配置要重新加载 511 GB 权重。白天人盯着跑只能试几组，
夜里 10 多个小时没人守着。`loadtest/run_ladder.py` 只管压一个已经起好的服务，
「停掉旧的、起新的、等就绪、出错了换下一个」这几步以前要人做。nightq 把这几步自动化：

- 一个任务 = 起服务 → 等就绪 → 按顺序跑几条命令 → 停服务、确认端口放开。
- 任何一步出问题（起不来、中途挂了、某步超时），记下原因，停掉服务，**接着跑下一个任务**。
- 每个任务结束就重写一次 `summary.md`，里面有每个任务的结果、用时、四档摸高和失败日志的最后几行。

## 怎么用

都在仓库根目录执行。

```bash
# 1. 写队列文件（照 nightq/deepseek-vllm.ini 改），先检查一遍，不执行
python3 nightq/nightq.py nightq/deepseek-vllm.ini --dry-run

# 2. 挂后台跑；--stop-after 表示 8:30 以后不再开新任务，把机器留给白天
nohup python3 nightq/nightq.py nightq/deepseek-vllm.ini --out nightq/runs/1130-node1 --stop-after 08:30 &

# 3. 中途或早上看结果
cat nightq/runs/1130-node1/summary.md
tail -f nightq/runs/1130-node1/nightq.log
```

没有 GPU 想先试：`python3 nightq/nightq.py nightq/example-mock.ini --out nightq/runs/demo`（假服务，约 15 秒）。
改了 nightq 之后跑 `bash nightq/selftest.sh`（约 40 秒，27 项检查）。

| 情况 | 做法 |
|---|---|
| 跑到一半断了（机器重启、被打断） | 同一条命令再跑一次：成功的跳过，被打断的那个重跑 |
| 想把失败的再跑一遍 | 加 `--retry-failed`；旧结果目录改名成 `任务名.attempt1` 留着 |
| 只跑其中几个 | `--only 任务名1,任务名2` |
| 想提前停 | `kill <nightq 的 pid>`（或 Ctrl-C）：它会先把当前服务停掉再退出 |
| 提示「已经有一个夜跑队列在跑」 | 同一台机默认只让一个队列跑。确实要并行（各用各的卡、各用各的端口）就加 `--lock 另一个文件` |

## 队列文件怎么写

INI 格式。每个 `[小节]` 是一个任务，按文件里的顺序跑。`[DEFAULT]` 里写的每个任务都继承，
所以做配置对照时，起服务命令和步骤写在 `[DEFAULT]`，每个任务只写不一样的环境变量：

```ini
[DEFAULT]
start = bash serve-deepseek.sh
ready_url = http://127.0.0.1:8000/health
env =
    MODEL_DIR=/data/models
    LOG_DIR=$NIGHTQ_OUT/logs
steps =
    BASE_URL=http://127.0.0.1:8000 MODEL=deepseek-v4.1-flash bash smoke-test.sh
    python3 loadtest/run_ladder.py --url http://127.0.0.1:8000 --data loadtest/data --out $NIGHTQ_OUT/ladder --engine builtin

[ds-tp8-baseline]

[ds-dep8]
env =
    TP_SIZE=1
    DP_SIZE=8
    ENABLE_EP=1
```

几条容易踩的规则：

- **注释必须独占一行**（`#` 开头）。写在命令后面的 `#` 会被当成命令的一部分。
- 多行的值（`steps`、`env`、`snapshot`）换行后要缩进，一行一条；一条命令不能折成两行。
- 任务里的 `env` 是并到 `[DEFAULT]` 的 `env` 上面，别的键是整个覆盖。
- 不需要服务的任务（比如生成 MoE 调优表），如果 `[DEFAULT]` 里有 `start`，要写一行 `start =` 把它清空。
- 键名写错会在开跑前报错，不会悄悄漏跑。所以**挂上去之前一定先 `--dry-run`**。
- 命令里可以用 `$NIGHTQ_OUT`（本任务的结果目录）、`$NIGHTQ_JOB`（任务名）、`$NIGHTQ_QUEUE_OUT`（整个队列的结果目录）。
  `env` 里的值不经过 shell，只认这三个变量。

全部的键：

| 键 | 默认值 | 作用 |
|---|---|---|
| `start` | 空 | 起服务的命令；空 = 这个任务不需要服务 |
| `ready_url` | `http://127.0.0.1:8000/health` | 返回 200 就算服务起好了 |
| `ready_timeout` | 3600 | 等就绪最多多少秒 |
| `steps` | 空 | 多行，一行一条命令，按顺序跑；每步的输出在 `step-N.log` |
| `step_timeout` | 14400 | 每一步最多多少秒，超了就杀掉 |
| `on_step_fail` | `stop` | `stop` = 某步失败后面的不跑；`continue` = 接着跑后面的 |
| `env` | 空 | 多行 `KEY=VALUE`，起服务和每一步都带上 |
| `snapshot` | 空 | 多行命令，就绪后和收尾前各跑一遍存档（比如 `nvidia-smi`），失败不影响任务 |
| `stop` | 空 | 停服务的命令；不写就只杀 `start` 起的那组进程 |
| `detached` | `false` | `true` = `start` 起完就返回（如 `docker run -d`），这时必须写 `stop` |
| `enabled` | `true` | `false` = 这个任务先不跑 |
| `cwd` | 启动 nightq 时的目录 | 命令的工作目录 |
| `stop_grace` | 60 | 发 TERM 后等多少秒再强杀 |
| `down_timeout` | 120 | 停服务后等端口放开最多多少秒 |
| `settle` | 5 | 端口放开后再等多少秒（让显存释放） |
| `health_interval` / `health_fail_limit` | 30 / 6 | 跑步骤期间每隔几秒探一次 `ready_url`，连续几次不通就判服务挂了；`health_fail_limit = 0` 关掉 |
| `ready_interval` | 5 | 等就绪时每隔几秒探一次 |

## 结果怎么看

```
nightq/runs/1130-node1/
  summary.md        ← 早上看这个
  nightq.log        ← 队列自己的流水
  queue.ini         ← 当时用的队列文件留底
  ds-tp8-baseline/
    job.json        ← 结果、原因、用时、命令、环境变量、仓库版本
    server.log      ← 服务的输出
    step-1.log …    ← 每一步的输出
    snapshot-ready.log / snapshot-end.log
    ladder/         ← 步骤自己写进 $NIGHTQ_OUT 的东西
```

| 结果 | 意思 | 先看哪里 |
|---|---|---|
| ✅ 成功 | 服务起来了，每一步都返回 0 | — |
| ❌ 服务没起来 | 就绪前进程退出，或等到超时还没返回 200 | `server.log` |
| ❌ 服务中途挂了 | 跑步骤时服务进程退出，或探活连续不通（多半是 OOM） | `server.log` 末尾、`snapshot-ready.log` 里的显存 |
| ❌ 步骤失败 / 步骤超时 | 某一步返回非 0，或超过 `step_timeout` | 对应的 `step-N.log` |
| ❌ 端口被占 | 开跑前端口上已经有服务。nightq 不会去动它 | 找出是谁起的，停掉后 `--retry-failed` |
| ⏹ 被打断 | nightq 收到了 TERM / Ctrl-C | 再跑一次就续上 |

「成功」只表示命令都返回 0，**不表示过了 SLO**。过没过线看 `summary.md` 的摸高一栏和各任务的 `ladder/report.md`。

## 用 docker 起服务

两件事要注意（**都没在真机上验证过**，上机第一个任务先用短的试一下）：

1. 给容器起名字，并写 `stop = docker rm -f 名字`。只杀 docker 客户端进程，容器不一定会停。
2. 队列文件 `env` 里的变量只到 `docker run` 这条命令为止，要进容器得在命令里加 `-e 变量名`
   （不带值，表示把外面的值传进去；外面没设就不传）。

```ini
[DEFAULT]
start = docker run --rm --name nq-ds --gpus all --ipc=host --net=host -v /data/models:/models:ro -v $NIGHTQ_OUT/logs:/logs -e MODEL_DIR=/models -e LOG_DIR=/logs -e TP_SIZE -e DP_SIZE -e ENABLE_EP -e ENGRAM_CPU_OFFLOAD -e KV_CACHE_DTYPE -e MAX_NUM_BATCHED_TOKENS -e DSPARK_TOKENS hackathon-vllm:0.34.0 bash serve-deepseek.sh
stop = docker rm -f nq-ds
```

## 排夜班时的几条建议

- 最想要的数排最前面。一个任务可能一两个小时，后面的不一定轮得到。
- 每个任务第一步放 `smoke-test.sh`：服务不对劲就早点失败，别压了一夜才发现。
- 步骤里把结果都写进 `$NIGHTQ_OUT`，汇总表才能自动带上摸高。
- 队列提前跑完后卡会空着。工作日空闲超 8 小时会被收回算力，所以最后放一个长任务（比如稳定性长压）垫底。
- 第二天把 `summary.md` 里的表抄进当日 devlog；`nightq/runs/` 不进仓库。

## 已验证和没验证的

- 已验证（2026-11-26，本机 Mac、Python 3.9.6、假服务，$0）：`bash nightq/selftest.sh` 27 项全过，
  覆盖上表每一种结果，以及续跑、重跑、被打断、同机互斥、写错键名；
  还用假的 `vllm` 和 `nvidia-smi` 串过真的 `serve-deepseek.sh` + `smoke-test.sh`。
- **没验证**：真 vLLM / SGLang、docker 方式、Linux 和 Python 3.12（比赛机的环境）、
  `deepseek-vllm.ini` 里各个开关在真机上是否可用。上机第一晚之前，先用一个短任务白天试一遍。
