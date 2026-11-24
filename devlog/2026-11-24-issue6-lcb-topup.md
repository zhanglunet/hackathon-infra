# issue #6 LCB 补跑：13 道缺题（Modal H100，FP8 32K）

- 时间：2026-11-24 12:44:34–13:03:33 UTC；部署 `ap-iAz3g6rrp6gx5MsNtERSLV`（H100）
- 配置：FP8 权重 + FP8 KV、`MAX_MODEL_LEN=272000`（YaRN）、`MAX_TOKENS=32768`、`TEMPERATURE=0`、`TOP_P=0.95`、`EVAL_BATCH_SIZE=32`、`--tool-call-parser qwen3_coder --reasoning-parser qwen3`（与上一轮 FP8 全套完全同口径，只变题集）
- 背景：上一轮 FP8 全套 LCB 只跑了 27/40（26/27 通过），缺 13 题（index 6, 9, 14–18, 21, 24, 25, 31, 34, 38）。本次单独补跑这 13 题，跑完 FP8 全套四集才完整。
- 前置说明：上一轮补跑尝试在动手前停下了——核对 EvalScope LCB 数据加载器代码时遇到执行风险检查没通过，当时未起 GPU、未花钱、未改文件。**用户明确拍板“继续用 EvalScope 跑”后执行本次补跑。**

## 方法：过滤数据集

- 脚本 `eval/run_modal_p0_eval.sh` 的 LCB 分支只能按顺序跑前 N 题，没有按 index 选子集的选项。做法：用 pyarrow 从 `eval/data/lcb_v6_flat/test.parquet` 按原顺序取出 13 行，建成 `eval/data/lcb_v6_topup13/test.parquet`，EvalScope `--limit 13` 跑该过滤集：
  ```python
  import pyarrow.parquet as pq
  t = pq.read_table('eval/data/lcb_v6_flat/test.parquet')
  missing = [6, 9, 14, 15, 16, 17, 18, 21, 24, 25, 31, 34, 38]
  pq.write_table(t.take(missing), 'eval/data/lcb_v6_topup13/test.parquet')
  ```
  （过滤集未随本提交推送——parquet 是二进制文件，推送脚本只走 UTF-8 文本；按上面三行即可精确重建。13 行 question_id 已逐项核对，与原表一致。）
- 口径依据：已实证 prediction 里的 `index` N = parquet 第 N 行（0-based；上一轮 27/27 按题目内容逐项比对一致）。合并时按 `question_id` 对回原 index，不按过滤集内的新 index。
- 部署方式：手动部署（脚本 fp8 模式的 trap 会在脚本退出时停机，不适合“部署后手动跑 EvalScope”；部署命令的环境变量与脚本内完全一致）。`/health` 第 3 次检查 200，`/v1/models` 确认 served 名 `qwen3.8-27b`。

## 结果：补跑 13 题 0/13

| 过滤集 index | 原 index | question_id | 输出 token | 结果 |
|---|---|---|---|---|
| 0 | 6 | abc388_g | 32,768（顶格截断） | 错 |
| 1 | 9 | abc388_f | 32,768（顶格截断） | 错 |
| 2 | 14 | abc389_g | 32,768（顶格截断） | 错 |
| 3 | 15 | abc389_f | 32,768（顶格截断） | 错 |
| 4 | 16 | abc389_e | 32,768（顶格截断） | 错 |
| 5 | 17 | abc390_e | 32,768（顶格截断） | 错 |
| 6 | 18 | abc390_g | 32,768（顶格截断） | 错 |
| 7 | 21 | abc390_d | 32,768（顶格截断） | 错 |
| 8 | 24 | abc391_d | 32,768（顶格截断） | 错 |
| 9 | 25 | abc391_f | 32,768（顶格截断） | 错 |
| 10 | 31 | abc392_g | 32,768（顶格截断） | 错 |
| 11 | 34 | abc392_f | 14,474（自然结束） | 错 |
| 12 | 38 | abc393_e | 32,768（顶格截断） | 错 |

- 补跑 13 题全军覆没（0/13）。12 道顶到 32,768 上限被截断（vLLM `/metrics` 佐证：本部署 `finish_reason` length 12、stop 1）；唯一自然结束的是 abc392_f（14,474 token），仍答错。
- 评测耗时：EvalScope 701 秒（约 11.7 分钟），平均输出 31,361 token/题（上一轮 27 题平均仅 6,804 token——这 13 道就是上轮最慢的那批）。
- 合并：上一轮 27 题 26 通过 + 本轮 13 题 0 通过 = **LCB 26/40 = 65.0%**（4096 口径 17/40 = 42.5%，+22.5pp）。本集截断率 12/40 = 30.0%（上一轮 27 题 0 截断）。失败题 index：6, 9, 14, 15, 16, 17, 18, 21, 24, 25, 31, 33, 34, 38（共 14 道）。

## 四集 32K 口径完整对照（FP8，temp 0）

| 集 | 32K | 4096 | 截断率 4096 → 32K |
|---|---|---|---|
| AIME 2025（30 题） | 26/30 = 86.7% | 17/30 = 56.7% | 70% → 10.0%（3/30） |
| GPQA Diamond 前 50 | 44/50 = 88.0% | 35/50 = 70.0% | 38–48% → 6.0%（3/50） |
| LiveCodeBench v6 前 40 | 26/40 = 65.0% | 17/40 = 42.5% | 55–57.5% → 30.0%（12/40） |
| BFCL v3 simple 前 50（native） | 49/50 = 98.00% | 47/50（qwen_agent） | — |
| **合计 170 题** | **145** | **116** | — |

## 结论

- FP8 全套 32K 四集已完整：**145/170 = 85.3%**（4096 口径 116/170 = 68.2%）。
- LCB 是短板：难题长推理顶格截断仍是主因（12/13 顶到 32,768）。32K 对这批题仍不够；是否值得试 64K 口径待定（另涉 KV 容量与成本）。
- **「FP8 vs BF16 亏没亏」仍未回答**：4096 口径的对照（116 vs 113）是在截断主导下测的，须等 BF16 32K 对照（issue #6 下一步）。

## 费用与停机

- 本部署账单 **$1.19**（H100 约 19 分钟：部署 12:44:34 → 停机 13:03:33 UTC），实付 **$0.00**（免费额度抵扣）。低于本次 $2.4 上限。本月累计约 $24.73，免费额度剩约 $5.27，仍高于 $4 余量红线。
- 停机复核：手动 `modal app stop --yes qwen27b-vllm` 后 `/health` 404，`modal app list` 确认 stopped、0 tasks。
- 环境：`/usr/local/lib/python3.12/dist-packages/hatch-user-site.pth` 在位，`import modal` 一次通过；部署前 `modal app list` 无残留；本地 5 个关键文件 blob sha 与 main（c826ecac）一致。
- `ENVIRONMENT.md` §5 已用 32K 完整四集数更新（新增 §5.1）。

## 待办

- BF16 32K 对照（issue #6 下一步），待拍板。
- LCB 32K 仍截断 30%：是否试 64K 口径，待定。
