# 2026-11-22 Modal H100 EvalScope 精度自测（FP8 Qwen3.8-27B）

- 部署：Modal H100，vLLM 0.34.0，`Qwen/Qwen3.8-27B-FP8`，served 名 `qwen3.8-27b`，`max_model_len` 262144（原生上限，未开 YaRN），prefix caching 开，`max_containers=1`；权重在 Volume `qwen27b-weights`（30.9 GB，已就绪，不用重下）
- 部署地址（已停）：`https://zhanglunet--qwen27b-vllm-serve.us-east.modal.run`
- App：`ap-5LB5rDSPAjyxXOr4rYonAD`（第一轮部署）；重部署后同名新 App（见“并发修复”一节）
- 任务：测 FP8 的 Qwen3.8-27B 在 190 题代理题库上的得分，验相对官方权重的精度损失是否 ≤2%
- 时间线（JST）：11-21 23:49:29 部署创建 → 23:52:19 /health 200 → 01:05:33 AIME 全量出分 → 01:06:04 重部署（并发修复）→ 01:09:20 新容器就绪 → 01:13:03 GPQA 全量出分 → 01:17:15 LCB 全量出分 → 01:18:46 部署已停（`modal app stop --yes`，复核 0 tasks，/health 404）

## 结论先行

| 代理集 | 题数 | FP8 得分 | 小样本验证 |
|---|---|---|---|
| AIME 2025（`aime25`） | 30 | **17/30 = 56.67%** | 3/3 |
| GPQA Diamond（`general_mcq`，本地 diamond 文件前 50） | 50 | **35/50 = 70%** | 3/3 |
| LiveCodeBench（`live_code_bench`，v6 前 40，pass@1） | 40 | **17/40 = 42.5%** | 2/3 |
| BFCL v3 | 50（计划） | **未跑**（见“BFCL 未跑原因”） | 未验证 |

**对照口径（重要，如实写）：**

- 官方 190 题组织方持有、不公开，本次是**代理题自测**，分数不能直接等同官方复测分。
- Qwen3.8-27B 官方权重（BF16）在上述代理集上的分数**没有公开可比数**，记为“**无公开基线对照**”。因此**不能下“精度损失 ≤2%”的结论**。
- 要下这个结论，缺的是：同一代理集上 BF16 权重的对照分——需另起一次 BF16 部署跑同一代理集（H100 80GB 装得下 56GB BF16 权重），或官方公布同题分数。本轮预算内未做 BF16 对照（见费用一节）。
- 另注意口径差异：GPQA 用的是 diamond（198 题最难子集）的前 50，难度高于 main-448 平均；LCB 用的是 v6（最新发布）的前 40；AIME 是完整 30 题（AIME I+II 2025）。

## 数据集取回情况

取数工具：`eval/fetch-benchmarks.py` 只过 `py_compile`，未真跑过；本轮真跑时在 numpy/pyarrow 冲突上失败（见问题 1），后改走 curl 直连 HF（已验证不弹授权卡）+ EvalScope 内置集。

| 集 | 来源 | 取回结果 | 备注 |
|---|---|---|---|
| AIME 2025 | `MathArena/aime_2025`，`data/train-00000-of-00001.parquet`（14KB） | 30 行，`eval/data/aime_2025.jsonl` | 字段 problem / answer（int）/ problem_type；已进仓库 |
| GPQA | 官方仓 `Idavidrein/gpqa` 是 gated（`gated: auto`，需登录接受条款），VM 上无 HF token，取不到 | 改走公开镜像 `fingertap/GPQA-Diamond` 的 `test/gpqa_diamond.parquet`（72KB），198 行 | 原文件只有 `question`/`answer` 两列（题干内含 A–D 选项、answer 为字母）；已解析成 `eval/data/gpqa_diamond.jsonl`（198 行，题干 + A/B/C/D + answer，198/198 解析成功，跨行选项用“从后往前找 A<B<C<D 标记”处理）；已进仓库 |
| LiveCodeBench Lite | `livecodebench/code_generation_lite`，`test.jsonl`~`test6.jsonl` | 6 个文件共 **866 题**（211+111+101+101+167+175），`eval/data/livecodebench/`，约 3.3GB | “Lite 六文件共 866 题”与“官方 Lite 40 题”不是同一口径——40 题是官方从中抽的子集；本轮 LCB 跑的是 v6（见下），6 个 jsonl 未进仓库（体积） |
| LiveCodeBench v6（实际跑的） | `marianna13/livecodebench_code_generation_lite_parquet` 的 `v6/test-00000-of-00001.parquet`（124MB，curl 52 秒下完） | 175 行，`eval/data/lcb_v6_flat/test.parquet`（本地，未进仓库） | EvalScope 默认子集 `release_latest` 是 9 个分片共约 4.2GB，下不动（见问题 5）；改 v6 单分片 |
| BFCL v3 | `gorilla-llm/Berkeley-Function-Calling-Leaderboard`，25 个 `BFCL_v3_*.json` | 25 文件共 5226 行，`eval/data/bfcl_v3/`（约 11MB），已进仓库 | 格式为 JSONL（每行一条），字段 id / question / function / ground_truth；但 EvalScope 的 bfcl_v3 adapter 要的是 ModelScope 加工版 schema（turns/tools/functions/initial_config 等），本地原始文件**不能直接喂**（见“BFCL 未跑原因”） |

## 小样本验证（全量前，每集 2–3 题）

- 先用 1 道 AIME 题直接调端点探过：响应 `content` 里**内含 think 块**（`<think>...</think>` 后接最终答案），无独立的 `reasoning_content` 字段；答案 `\boxed{70}` 与 gold 一致。EvalScope 的 `\boxed{}` 提取（math-verify 规则判分）与 general_mcq 的字母提取在小样本里均工作正常。
- AIME 3/3（100%），GPQA（general_mcq 本地文件）3/3（100%），LCB 2/3（66.7%）。三集 harness 均调通后才跑全量。
- 生成参数（全量与小样本一致）：`temperature: 0`，`max_tokens: 4096`；`--eval-batch-size 16`（AIME 全量第一轮时服务端串行，实际未生效，见“并发修复”）。

## 全量结果（FP8，temperature 0，max_tokens 4096）

### AIME 2025：17/30 = 56.67%

- EvalScope `aime25`（HF `math-ai/aime25`），0-shot，提示词要求 `\boxed{}`，math-verify 规则判分（`integer_only=True`）。
- 报告：`eval/outputs/20261121_164127/reports/qwen3.8-27b/aime25.json`；预测与评审同目录。
- 平均输入 254 token、平均输出 3477 token；30/30 成功、0 报错。
- 注意：AIME 全量跑在**串行**服务端上（重部署前），报告里的平均延迟 508.4 秒含排队等待，不是单题生成时间；单题墙钟约 44 秒。GPQA/LCB 的延迟是在并发服务端上测的，两者延迟数字不可直接比较。

### GPQA Diamond：35/50 = 70%

- EvalScope `general_mcq` + 本地文件 `--dataset-args '{"general_mcq": {"dataset_id": ".../eval/data/gpqa_diamond.jsonl"}}'`，取前 50 题，0-shot，字母答案提取。
- 报告：`eval/outputs/20261121_170953/reports/qwen3.8-27b/general_mcq.json`；用时 3 分 07 秒。
- 平均输入 284 token、平均输出 2592 token、平均延迟 47.8 秒；50/50 成功、0 报错。
- 注意：EvalScope 内置的 `gpqa_diamond` 走 ModelScope（`AI-ModelScope/gpqa_diamond`），本次因 ModelScope 联网确认卡未获批改走本地文件；题源是同一 diamond 198 题，但抽样与选项洗牌与内置集不完全一致，分数只在“diamond 前 50”口径内可比。

### LiveCodeBench v6：17/40 = 42.5%（pass@1）

- EvalScope `live_code_bench` + 本地文件 `--dataset-args '{"live_code_bench": {"dataset_id": ".../eval/data/lcb_v6_flat", "subset_list": ["default"]}}'`，取 v6 前 40 题，0-shot，本地执行测试用例判分（pass@1）。
- 报告：`eval/outputs/20261121_171331/reports/qwen3.8-27b/live_code_bench.json`；用时 3 分 41 秒。
- 平均输入 638 token、平均输出 2701 token、平均延迟 50.0 秒；40/40 成功、0 报错。

### BFCL v3：未跑

- 原因有二，缺一不可：
  1. EvalScope 的 `bfcl_v3` adapter 没有 HF 镜像（`dataset_id='AI-ModelScope/bfcl_v3'`，无 `hf_dataset_id`），只能走 ModelScope；ModelScope CDN（`cdn-lfs-cn-1.modelscope.cn`）的联网确认卡自 00:15 JST 起挂起未获批，下载走不通。
  2. 本地已取回的 25 个原始 JSON 文件与 adapter 要的加工版 schema 不匹配（原始字段只有 id/question/function/ground_truth；adapter 要 turns/tools/functions/initial_config/missed_functions/multi_turn 等），且 `datasets.load_dataset` 对本地目录不支持按子集名（`simple` 等 17 个子集）解析配置，拼成本地目录的两次尝试均报 `BuilderConfig ... not found`。
- 解堵二选一（下次做）：① 用户点批 ModelScope 联网确认卡，走 EvalScope 内置 `bfcl_v3`（需 `bfcl_eval`，已装好）；② 用 `bfcl_eval` CLI（`bfcl generate` + `bfcl evaluate`，`--skip-server-setup` 指已有端点）直接跑本地原始文件——CLI 本身已能跑（补装了缺失的 `soundfile` 依赖），但模型配置接线（把 `qwen3.8-27b` 指到本地 OpenAI 兼容端点）还没做，本轮没时间试。

## 并发修复（本轮最大发现，影响所有后续 Modal 评测）

- 现象：全量 AIME 第一轮（`--eval-batch-size 16`）跑得极慢，vLLM 引擎日志恒为 `Running: 1 reqs, Waiting: 0 reqs`，生成吞吐约 80 tok/s；4 路并行 curl 直接请求全部在 120 秒超时。结论：请求在服务端被**串行化**了，`--eval-batch-size` 再大也没用。
- 根因：Modal 的 `@app.function` 默认每个容器一次只处理 1 个输入；`max_containers=1` 时多余请求只排队，不扩容。之前冒烟/272K 探测都是单请求，从未暴露。
- 修复：`modal/modal_vllm_qwen27b.py` 的 serve 函数加 `@modal.concurrent(max_inputs=16)`（注意：`allow_concurrent_inputs=16` 参数在 modal 1.6.4 已废弃，直接写会报 `DeprecationError`，第一版重部署因此失败，改装饰器后才部署成功）。
- 验证：重部署后 4 路并行 curl 全部 5.6–6.3 秒同时返回；引擎日志 `Running: 16 reqs`，生成吞吐 800–900 tok/s（约 10 倍）。GPQA 50 题 3 分 07 秒、LCB 40 题 3 分 41 秒跑完（串行估算要 37 分钟和约 30 分钟）。
- 代价：重部署约 3 分钟模型重载；第一轮 AIME 是串行跑完的（24 分钟），GPQA/LCB 是并发跑的。以后 Modal 评测默认带这个装饰器。

## 墙钟时间与费用（估算）

- H100 部署总墙钟：2026-11-21 23:49:29 JST 创建 → 2026-11-22 01:18:46 JST 停，共 **89.3 分钟**。
- 费用估算（按 H100 约 $3.95/时）：89.3 分钟 ≈ **$5.87**。本轮开始前剩余额度 $19.91（账单口径，00:06 核过）→ 本轮后剩约 **$14.03（估算，账单页未复核）**。本轮总预算约 $5–8，实际约 $5.87，在预算内。
- 明细（均为墙钟，含部署等待与初始化）：
  - 取数与环境折腾约 40 分钟（期间部署已起，GPU 空转——最大浪费项；教训：下次先备好数据再起部署，或部署后立即跑小样本）。
  - AIME 全量（串行）约 24 分钟；重部署约 4 分钟；GPQA 全量 3 分 07 秒；LCB 全量 3 分 41 秒。
- 注意：账单口径与用量明细口径此前有约 $0.13–0.16 的差异（见 H100 冒烟篇），上述为估算，以账单页复核为准。

## 遇到的问题（按时间序）

1. **numpy 被 bfcl 依赖降级**：`evalscope[bfcl]` 安装时顺带装了 numpy 1.26.4，把 numpy 2.5.5 顶掉，`fetch-benchmarks.py` 报 `pyarrow requires NumPy 2.0 or newer`。修法：装完 bfcl 后重装 `numpy>=2`（2.5.5），复核 `import evalscope/datasets/bfcl_eval` 全通。
2. **系统 scipy 与 numpy 2 不兼容**：`datasets` 加载时 `scipy.io` 报 `numpy.core.multiarray failed to import`。修法：`pip install --user --force-reinstall --no-deps scipy`（1.18.1）。
3. **EvalScope 参数名**：`--outputs` 不存在，输出目录用 `--work-dir`；`--datasets` 多集用空格分隔（逗号会被当成一个名字）；`--dataset-hub` 取值是小写 `huggingface`/`modelscope`/`local`（`HuggingFace` 报 `Unsupported dataset hub`）。
4. **GPQA 官方仓 gated**：`Idavidrein/gpqa` 需登录接受条款，VM 无 HF token；改公开镜像 `fingertap/GPQA-Diamond`（parquet，198 行，question/answer 两列），解析成 general_mcq 格式。
5. **LCB 默认子集太大**：EvalScope 默认 `release_latest` 是 9 分片约 4.2GB，约 1MB/s 下不动（已下 805MB 后放弃）；改 v6 单分片 124MB（curl 52 秒）。另：`datasets.load_dataset` 对本地目录不支持 `{子集}/{分片}` 的配置写法（两次尝试均 `BuilderConfig ... not found`），最终用平铺 `test.parquet` + `subset_list: ["default"]` 走通。
6. **ModelScope 联网确认卡**：gpqa 内置集与 bfcl_v3 都要走 ModelScope，`modelscope.cn` / `cdn-lfs-cn-1.modelscope.cn` 的确认卡挂起约 1 小时未获批（用户未点）；GPQA 改本地文件绕过，BFCL 未绕过（见上）。
7. **VM 运行时重启**：约 00:38 JST，运行时重启杀掉第一轮 AIME+GPQA 全量跑（只跑了 37 秒，无结果）；Modal 侧部署未受影响（App 仍 deployed、1 task），但残留的孤儿请求在服务端又跑了几分钟（/health 一度因 10 秒超时报 000，60 秒超时即 200——教训：健康检查超时别设 10 秒）。
8. **bfcl CLI 缺依赖**：`bfcl` 启动报 `No module named 'soundfile'`（qwen_agent 的传递依赖），补装后 `--help` 正常；模型接线未做（见 BFCL 一节）。
9. **AIME 延迟数字口径**：AIME 报告的平均延迟 508.4 秒含串行排队，不代表单题生成时间；GPQA/LCB 的延迟是并发下的，不可横向比较。平均输出 token 数（AIME 3477 / GPQA 2592 / LCB 2701）才是可比的成本口径。

## 复现命令（均在仓库根目录，部署已停，需重部署后跑）

```bash
# AIME 30 题（HF hub）
evalscope eval --model qwen3.8-27b --eval-type openai_api \
  --api-url https://zhanglunet--qwen27b-vllm-serve.us-east.modal.run/v1 --api-key EMPTY \
  --datasets aime25 --dataset-hub huggingface --limit 30 --eval-batch-size 16 \
  --generation-config '{"max_tokens": 4096, "temperature": 0}' --work-dir ./eval/outputs

# GPQA diamond 前 50（本地文件）
evalscope eval --model qwen3.8-27b --eval-type openai_api \
  --api-url https://zhanglunet--qwen27b-vllm-serve.us-east.modal.run/v1 --api-key EMPTY \
  --datasets general_mcq \
  --dataset-args '{"general_mcq": {"dataset_id": "/home/hatch/workspace/hackathon-infra/eval/data/gpqa_diamond.jsonl"}}' \
  --limit 50 --eval-batch-size 16 \
  --generation-config '{"max_tokens": 4096, "temperature": 0}' --work-dir ./eval/outputs

# LCB v6 前 40（本地 parquet，需先按 devlog 所述取回 test.parquet 到 eval/data/lcb_v6_flat/）
evalscope eval --model qwen3.8-27b --eval-type openai_api \
  --api-url https://zhanglunet--qwen27b-vllm-serve.us-east.modal.run/v1 --api-key EMPTY \
  --datasets live_code_bench \
  --dataset-args '{"live_code_bench": {"dataset_id": "/home/hatch/workspace/hackathon-infra/eval/data/lcb_v6_flat", "subset_list": ["default"]}}' \
  --limit 40 --eval-batch-size 16 \
  --generation-config '{"max_tokens": 4096, "temperature": 0}' --work-dir ./eval/outputs
```

环境前置（本 VM 已装）：evalscope 1.15.0、datasets 4.8.4、numpy 2.5.5、pandas 3.1.1、scipy 1.18.1、bfcl-eval 2025.10.27.1；`NO_PROXY=localhost,127.0.0.1` 覆盖掉裸 IPv6 的代理配置再跑（否则 httpx 报 `Invalid port: ':1]'`）。

## 本次进仓库的文件

- `devlog/2026-11-22-modal-h100-evalscope.md`（本篇）+ `devlog/README.md` 索引
- `modal/modal_vllm_qwen27b.py`（`@modal.concurrent(max_inputs=16)` 并发修复）
- `eval/data/aime_2025.jsonl`、`eval/data/gpqa_diamond.jsonl`、`eval/data/bfcl_v3/*.json`（25 个）
- 三轮全量输出的报告/预测/评审/任务配置：`eval/outputs/20261121_164127/`（AIME）、`eval/outputs/20261121_170953/`（GPQA）、`eval/outputs/20261121_171331/`（LCB）
- 未进仓库（体积原因，本地留存、可复现）：`eval/data/livecodebench/`（3.3GB，6 个 jsonl）、`eval/data/lcb_v6_flat/test.parquet`（118MB，取回命令：`curl -L -o eval/data/lcb_v6_flat/test.parquet https://huggingface.co/datasets/marianna13/livecodebench_code_generation_lite_parquet/resolve/main/v6/test-00000-of-00001.parquet`）、`eval/data/gpqa_diamond/train.parquet`（原始镜像文件，72KB；解析脚本见取数过程）

## 待办

- BFCL v3：按“BFCL v3：未跑”一节二选一解堵后补跑（50 题）。
- BF16 对照：要下“损失 ≤2%”的结论，需另起 BF16 部署跑同一代理集（AIME 30 + GPQA 前 50 + LCB v6 前 40）作对照。
- 费用：到 Modal 账单页复核本轮实际扣费与剩余额度（估算剩约 $14.03）。
- 部署纪律补充：以后 Modal 评测先备好数据再起部署；`@modal.concurrent` 已成默认配置；健康检查超时用 60 秒。
