# devlog：压测加共同 system prompt 选项 + 路由器对照脚本（2026-11-26）

## 背景

issue #14 里肖娅自留两件事：① 给 `make_datasets.py` 加「所有请求带同一段共同 system prompt」选项；
② router 改完后用假服务验证。② 已在 2026-11-26 做完并回帖到 #14（结论：两个 P0 修好，
默认 `--epsilon 0.25` 让命中率掉约 10 个点）。这次补 ①，并把验证用的临时脚本整理成 `loadtest/router_check.py`，
上机后用真 vLLM 再跑一次对照（假服务的 SSE 不是 chunked 编码，真 vLLM 是）。

## 改了什么（只动 loadtest/）

- `make_datasets.py`：新增 `--shared-system-tokens N`（默认 0，按 `--scale` 缩放）和
  `--shared-system-role system|user`（默认 system）。共同内容用单独的随机源生成，**不加选项时生成的 JSONL
  与改动前逐字节相同**（已 diff 核实，只有 meta.json 多两个字段）。
- `builtin_client.py`：记录里有 `system` 字段时，发成 `[system, user]` 两条消息。
- `run_ladder.py`：每次运行的随机标记有 system 时加在 system 最前面（它才是整个请求的开头）。
- `aiperf_engine.py`：有 system 时写 `system_prompt.txt`，传 AIPerf 0.14.0 的 `--system-prompt-file`
  （源码 `dataset/composer/base.py::_inject_custom_system_prompt` 对自定义数据集同样生效）。
  用抓包代理核实：8 个预热 + 24 个 AIPerf 请求全部是 `[system, user]`，system 开头是同一个运行标记。
- 新增 `router_check.py`：同一份数据直连各副本（客户端按会话固定）和经路由器各跑一遍（都用 builtin 引擎），
  出 `router_check.md`：每级命中率 / P90 TTFT / P90 TPOT / 错误 / Goodput 并排，加各副本请求占比
  （读 `/router/stats` 前后差）和故障转移次数；命中率差 > 0.03 或最忙副本 > 2 倍均分时给警告。
  `--mock N` 本地自起假服务，不给 `--router` 时自起 `prefix_router.py`（`--router-args` 透传）。
- `README.md`：数据一节、用法 2.5 节、「什么情况怎么用」表各加一条。

## 验证（本地 Mac，4 个假服务，scale 0.02，并发 32，$0）

| 设置 | agent 命中 直连 → 路由 | coding 命中 直连 → 路由 |
|---|---|---|
| 旧版 router（709718a），共同前缀在 system | 0.75 → 0.67（13 个请求出错） | 0.89 → 0.81（9 个出错） |
| 当前 router（92ddd53）默认 | 0.75 → 0.66 | 0.89 → 0.77 |
| 当前 router `--epsilon 1` | 0.75 → 0.75 | 0.89 → 0.89 |
| 共同前缀在 user 消息，默认 | 0.75 → 0.49 | 0.89 → 0.52 |
| 共同前缀在 user 消息，`--prefix-chars 2048` | 0.75 → 0.65 | 0.89 → 0.78 |
| 共同前缀在 user 消息，`--epsilon 1 --prefix-chars 2048` | 0.75 → 0.75 | 0.89 → 0.87 |

旧版 router 出错样例：`URLError: [Errno 22] Invalid argument`、`Connection reset by peer`（未深究，旧版已被替换）。
旧版没有 `/router/stats`，所以报告里没有负载占比，脚本会注明。
`bash loadtest/selftest.sh` 仍通过（四档并发 8 都有结果，命中 0.65 / 0.87）。

## 未核实 / 待上机

- 真 vLLM（chunked SSE）经路由器的表现：上机后用 `router_check.py --backends ... --router ...` 跑一次。
- 官方数据的共同前缀放在哪、多长：决定 `--prefix-chars` 取多少（必须比共同前缀的字符数长）。
