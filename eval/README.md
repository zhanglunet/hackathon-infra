# 精度自测：测试集说明

## 官方 190 题不在这里

比赛的 190 题固定题库由组织方持有，不公开；复测以组织方那套为准。
本目录**不含官方题**，也不冒充官方题。

## 题库构成（官方口径）

| 评测 | 官方题数 | 公开数据集 | HuggingFace ID（2026-11-20 核对存在） |
|---|---|---|---|
| AIME 2025（数学） | 30 | AIME 2025 全集 30 题 | `MathArena/aime_2025` |
| GPQA Diamond（知识问答） | 50 | GPQA Diamond 198 题（自测抽前 50 题） | `Idavidrein/gpqa`（本地数据 `eval/data/gpqa_diamond.jsonl`） |
| LiveCodeBench Lite（编程） | 40 | Lite 六个版本 `test.jsonl`~`test6.jsonl` | `livecodebench/code_generation_lite` |
| BFCL v3（工具调用） | 50 | BFCL v3 25 个分类 JSON | `gorilla-llm/Berkeley-Function-Calling-Leaderboard`（取 `BFCL_v3_*.json`） |

官方 190 题是上述公开集的固定子集，具体抽了哪 50/40 题未披露；
自测用全集或自行抽题，结果只能做参考，不能等同于复测成绩。

## 自测办法

1. 下数据：`python3 eval/fetch-benchmarks.py`（需先装
   `pip install datasets huggingface_hub`），落到 `eval/data/`。
2. 起服务：按 README §3 起 vLLM（272K、FP8、prefix caching）。
3. 跑精度：用 EvalScope 对本地 `/v1/chat/completions` 跑上述四集，
   与官方权重基线对比，确认损失 ≤2%；BFCL 按官方评测仓库的办法跑。
4. 跑压测：用 AIPerf 按并发阶梯 8/16/32/64/80 压，核 TTFT/TPOT SLO
   （见 `EVALUATION.md`）。

## 注意

- 自测通过 ≠ 复测通过；复测为准。
- 公开集的题量比官方大（GPQA 主集 448 题 vs 官方 50 题），
  跑全集更费时间，赶时间可先抽与官方同题数的小样本做冒烟。
- BFCL 评测需要真实执行函数调用环境，按官方脚本要求准备。
- 数据集 ID、配置若在实例上报错，以 HuggingFace 页面为准核对；
  不要凭记忆改 ID。
