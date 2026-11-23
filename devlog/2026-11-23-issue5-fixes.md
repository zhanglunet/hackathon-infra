# 2026-11-23 issue #5 三项代码修复

- Issue：#5《赛前全仓评审：修复部署与评测链路，补齐三模型验收》（肖娅 2026-11-23 08:27 开），五项里只做三项代码修复；另两项（DeepSeek 启动链路、官方精度对照 + 四档性能数据）被堵，本轮未动。
- 动手前先把本地工作目录同步到 main @ `73a15a5`（另一边今早合的 PR #3、#4）：17 个文件逐个按 Git Data API 的 blob sha 校验，与 main 完全一致。
- 费用：$0，未部署、未跑 GPU。

## 修复 1：download-model.sh 裸机下载入口

### 定位

- `df -BG --output=avail "$MODEL_DIR"` 在全新环境（没有 `models/`）会失败；在 `set -e + pipefail` 下，赋值语句拿到非零退出码，脚本在 `mkdir -p` 之前就退出了。
- 兜底分支 `df ... .` 查的是当前目录，和目标盘可能不是同一文件系统，数字会错（MODEL_DIR=/data/models 时尤其危险）。
- 另外两个同类问题：① 跳过下载只看 `model.safetensors.index.json` 是否存在——半成品也可能已有 index.json，跳过后校验必失败，还得整删重下；② ModelScope 失败回退 HF 前 `rm -rf "$TARGET_DIR"`，把半成品删了，逼用户重下几十 GB。

### 改动

- `mkdir -p "$MODEL_DIR"` 移到 `df` 之前，`df` 直接查 `$MODEL_DIR`。
- 新增 `weights_complete()`：index.json 存在 **且** 分片数 == 预期（FP8 66 / BF16 18）才算已有完整权重、跳过下载；否则进下载流程，往已有目录里续下。
- 删掉回退分支的 `rm -rf "$TARGET_DIR"`：半成品留在原地，`hf download` 会跳过已下完的分片续下。

### 验证

- `bash -n download-model.sh` 通过；grep 确认 `rm -rf "$TARGET_DIR"` 已无，`weights_complete` 在位。

## 修复 2：run_modal_p0_eval.sh 评测隔离

### 定位

- `DEPLOYED=1` 在 `modal deploy` 命令**之前**就置位：部署失败时 trap 照样去停共享应用，可能杀掉别人的部署。
- `wait_health` 只看 `/health` 200：上一轮崩了没停干净的旧实例也能返回 200，测到的会是旧实例。
- BFCL 垫片直接改 site-packages 里的数据文件：并发轮次互相覆盖；被 `kill -9` 时 trap 跑不到，垫片残留（50 行的 v3 数据冒充 400 行的 v4），下轮会拿 v3 数据当 v4 跑。

### 改动

- `DEPLOYED` 只在 `modal deploy` 返回 0 后才置 1；部署失败时明确报错、不停共享应用。
- 部署前新增 `ensure_no_stale_deploy()`：查 `modal app list`，残留的同名部署先停掉再部署本轮（本脚本按“同一时间只跑一轮”使用）。
- `wait_health` 在 `/health` 200 后再验 `/v1/models` 里有 served 名 `qwen3.8-27b`，否则报错退出。
- `install_bfcl_shim` 改为隔离环境：把已装的 `bfcl_eval` 整个包拷到本轮临时目录，垫片只装进副本，再把副本目录插到 `PYTHONPATH` 最前面；已装的包原样不动，退出时只删临时目录。装前加守卫：已装的数据文件须为 400 行（原版 v4），否则报错并给出恢复命令（`pip install --force-reinstall --no-deps bfcl-eval==2025.10.27.1`）。
- `cleanup` 不再恢复 site-packages 文件；停机复核 `/health` 非 404/000 时打印警告并给出手动停机命令。

### 取舍：没用 `modal deploy --name` 做每轮独立命名

`modal deploy` 有 `--name` 选项，但改名后 webhook URL 的派生规则未经实测验证，写错会让 #6 的 `wait_health` 对着死地址空等 10 分钟。固定名 + 残留预检 + 端点验名已覆盖顺序轮次的隔离需求；并发同跑仍共享应用名，是残留限制。

### 接口未变

`prep` / `fp8` / `bf16` 三档与全部环境变量（`MODAL_GPU`、`MAX_MODEL_LEN`、`MAX_TOKENS`、`RUN_DATASETS`、`RUN_EVALSCOPE`、`MODAL_GPU` 等）原样；#6 的执行命令不受影响。

### 验证

- `bash -n eval/run_modal_p0_eval.sh` 通过；grep 确认旧的 `SHIM_BACKUP_DIR` / `BFCL_PKG_DATA` 引用已清零。
- 离线抽测 `install_bfcl_shim`（只抽函数定义跑，不部署）：守卫通过（已装 400 行）；副本里两文件均为 50 行垫片；已装包前后都是 400 行、未动；`import bfcl_eval` 落在副本路径里；清理删临时目录。
- `bash eval/run_modal_p0_eval.sh prep` 实跑通过（$0）：依赖与数据全部就绪，LCB parquet 本地已有（123,690,418 字节），未重下。

## 修复 3：测量工具校准

### 3a. check_truncation.py：三态 + 稳定题目 ID + 逐集配对

#### 定位

- 无 `stop_reason` 时拿 `</think>` 缺失当截断证据：没有 predictions 文件的轮次会被整批误判成截断。
- 逐题对照按 `index` 配对：`messages[i].id` 是每轮随机生成的，不能跨轮；`index` 在同一数据集同一 limit 下碰巧稳定（实测三集同 index 同题：AIME 30/30、GPQA 50/50、LCB 40/40），但 limit 或数据集版本一变就错位。
- `compare` 只比每目录的 `rows[0]`：一个轮次目录里有三集时只对照第一集。

#### 改动

- 每题三态：截断（`stop_reason` 是 `max_tokens`/`length`）、完成（有 `stop_reason` 且不是截断）、未知（拿不到 `stop_reason`，如缺 predictions 文件）；`</think>` 计数只作诊断参考。表头新增“未知”列。
- 新增 `_stable_qid()`：user 角色消息内容 + `target` 的 sha256。LCB 的题目在第二条 user 消息（第一条是共享的 system 指令），所以拼全部 user 消息；user 消息内容在轮内无重复（AIME 30/30、LCB 40/40 唯一）。
- `compare` 按数据集名逐集配对，只在一边出现的数据集打印跳过；翻转题目列表显示 `index`（无 `index` 才显示稳定 ID 前 8 位）。

#### 验证（离线回归，用的就是仓库里已有的 FP8/BF16 全量报告）

- 单集：AIME FP8 17/30 vs BF16 14/30，截断 21/30（70.0%），未知 0，完成正确率 9/9（100.0%）；与 devlog 已记录数一致。
- 三集新旧配对逐项一致：AIME FP8 独对 3 / BF16 独对 0；GPQA 4 / 3；LCB 0 / 0。合计 FP8 独对 7、BF16 独对 3，p=0.3438。
- 口径提醒：BF16 devlog 里的 p≈0.55 是把 BFCL 的 1 题翻转也 pooled 进去（7 vs 4，170 题）；本工具只覆盖三集 EvalScope（120 题），p=0.3438。两个结论同样是“差异不显著”，旧结论不变。
- 无 predictions 的目录：30 题全标未知、截断 0（旧逻辑会全算截断）。
- 数据集不匹配的对照：打印跳过、exit 0。

### 3b. probe/run_probe.py + probe/cold_only.py：reasoning、流结束状态、非零退出

#### 定位

- 只数 `delta.content`：开 `--reasoning-parser qwen3` 后思考先以独立 `reasoning` 字段流出，只数 content 会漏掉整段思考、TTFT 也会被高估。
- 流没收到 `[DONE]`（连接断了）、或收到 `error` chunk，也当成功记数。
- `run_probe.py` 失败也返回 0，下游以为测完了（`cold_only.py` 原本就 exit 1）。

#### 改动

- `reasoning_content` / `reasoning` 与 `content` 都计 token：`tokens` 为两者之和，另分列 `tokens_content` / `tokens_reasoning`；TTFT 取首个任意 token。没开 reasoning-parser 时 reasoning 为空，`tokens` 就是原来的 content 计数。
- 捕获 `finish_reason`；`length` 时标 `truncated: true`（延迟数字仍有效，但答案不完整）。
- 流在 `[DONE]` 之前断开、或收到 `error` chunk，记 error；`run_probe.py` 任一请求失败则 exit 1（结果文件照常写）。

#### 验证（本地 fake SSE 服务器，四组，$0）

- reasoning + `length`：tokens 8（3 推理 + 5 内容）、`truncated: true`、`finish_reason: length`，exit 0（两个脚本都过）。
- 断流无 `[DONE]`：error `stream ended without [DONE]`，exit 1（两个脚本都过）。
- error chunk：error `stream error: ...`，exit 1（两个脚本都过）。
- 纯 content：tokens 4、reasoning 0、`finish_reason: stop`，exit 0，与旧行为一致。
- 测试覆盖写坏的 `probe_result.json` / `cold_probe_result.json` 已从备份恢复，sha256 与测试前一致（cold 41.09 秒记录完好）。

## 未动

- #5 另两项：三模型提交链路（DeepSeek 启动 + 第二款模型未定）、官方精度对照和四档性能数据（要官方包 + 真机）。
- #1（FlagOS）、#6（Modal GPU 跑分）未动。
- modal 部署代码的默认值（65536 步长、FlashInfer、breakable=0 等）一处未改。
- `download-model.sh` 仍只认 Qwen，DeepSeek 下载脚本没写。

## 文件清单

- `download-model.sh`（裸机下载入口：先建目录再 df、完整权重判定、HF 回退续下）
- `eval/run_modal_p0_eval.sh`（部署成功才置 DEPLOYED、残留预检、端点验名、BFCL 垫片隔离环境）
- `eval/check_truncation.py`（三态、稳定题目 ID、逐集配对）
- `probe/run_probe.py`、`probe/cold_only.py`（reasoning 计数、流结束状态、失败非零退出）
- `devlog/2026-11-23-issue5-fixes.md`（本篇）
