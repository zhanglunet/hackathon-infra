# 黑客松推理排练：部署脚本包

给「AI Infra 推理性能打榜」（2026-11-27 比赛）准备的部署物。
目标：先在租来的单卡 A100 80GB（或同档 80GB 卡）上把 Qwen3.8-27B 跑起来，
验证 vLLM 服务、272K 上下文、/metrics 指标；8 卡 DeepSeek 排练见 §6。

## 文件清单

| 文件 | 作用 |
|---|---|
| `setup-vllm.sh` | 检查 GPU/Python/磁盘，建虚拟环境，装 vLLM 0.34.0 + modelscope + huggingface_hub |
| `download-model.sh` | 下 Qwen3.8-27B 权重：ModelScope 优先，HF（hf-mirror）备用；校验分片数和大小 |
| `serve-qwen27b.sh` | 起 Qwen3.8-27B vLLM OpenAI 服务：272K 上下文、FP8、prefix caching、parser、DP/TP、/metrics，端口 8000 |
| `download-deepseek.sh` | 下 DeepSeek-V4.1-Flash-1017 权重（约 511 GB / 48 分片）：ModelScope 优先，HF 断点续下 |
| `serve-deepseek.sh` | 起 DeepSeek-V4.1-Flash-1017 8 卡 vLLM 服务（`deepseek_v41` tokenizer/parser、Engram CPU offload、272K） |
| `download-glm.sh` | 下 GLM-5.3-Flash 权重（约 328 GB，原生 FP8）：ModelScope 优先，HF 断点续下 |
| `serve-glm.sh` | 起 GLM-5.3-Flash 8 卡 vLLM 服务（`glm47` parser、FP8 KV、272K、SM120 MoE 调优表支持） |
| `router/prefix_router.py` | 多副本前缀粘滞路由器（纯标准库，按前缀一致性哈希保缓存命中，聚合 `/metrics`） |
| `scripts/tune_moe_sm120.sh` | RTX 6000D（SM120）Triton MoE 调优表一键生成脚本（自动补丁 GLM/DeepSeek 架构） |
| `loadtest/` | 四档 SLO 场景 × 并发阶梯压测框架（AIPerf + 内置引擎、假服务、自动摸高报告） |
| `nightq/` | 夜跑队列：按清单逐个配置起服务、等就绪、跑步骤、停服务，失败就换下一个，早上看 `summary.md` |
| `Dockerfile` | 标准化推理容器镜像草稿（CUDA 13.0 devel + Python 3.12 + vLLM 0.34.0 + FlashInfer） |
| `smoke-test.sh` | 冒烟测试：/health、/v1/models、流式出 token 测 TTFT/TPOT、/metrics |
| `README.md` | 本文件 |
| `EVALUATION.md` | 评测标准与要求提炼：模型、精度门槛、SLO、并发阶梯、提交物 |
| `eval/` | 精度自测：测试集构成说明 + 下载四项公开数据集的脚本（官方 190 题不公开，这里是自测代理） |
| `site/` | 报告站：build.py 渲染 Markdown 成静态站，deploy.py 部署到 Cloudflare Pages |
| `reports/` | 报告：每篇一个 Markdown，发布办法见 reports/README.md |
| `modal/` | Modal 部署：vLLM 服务代码 + 说明（A100 冒烟 / H200 测 272K） |
| `probe/` | 长上下文探测脚本：272K 提示构造 + 探测运行（探测文本本身未进仓库，可按脚本重建） |
| `devlog/` | 开发日志：每次部署验证一篇，索引见 devlog/README.md |

> 校验状态（2026-11-20）：5 个文件都过了 `bash -n` 语法检查；
> `modelscope download` 和 `hf download` 在本机实测过（各下一个 51 KB 的
> config.json，都成功）；vLLM 的全部启动参数逐项核对过 vLLM 0.34.0 官方
> CLI 参考（docs.vllm.ai）。唯一没在本机跑通的是 `vllm serve --help`
> （本机无 GPU、装 torch 时磁盘一度不够，见 §5）——参数本身以官方文档为准，
> 首次在实例上跑时如遇“不认识的参数”报错，按 §5 处理。

## 0. 租机（开始前先看）

- 2026-11-20 19:00 实查 AutoDL 行情：**A100 80GB 全站无货**。
  有货的同档卡：A800-80GB-NVLink（北京 B 区 6 张，¥7.98/时）、
  H800 80GB（西北 B 区 4 张，¥9.98/时）、H20 96GB（北京 A 区 2 张，¥9.98/时）。
- **推荐租 H800 80GB（西北 B 区，¥9.98/时）**：它是 Hopper 架构，支持 FP8，
  27B 跑 FP8 权重（约 28 GB）+ FP8 KV 缓存才能塞下 272K 上下文。
  **不要租 A800**：A800 是上代 Ampere 架构，没有 FP8 张量核，跑不了 FP8 权重；
  改跑 BF16 的话权重 56 GB 占掉大半显存，272K 测不了，而且与比赛的
  Blackwell 卡差距大。`serve-qwen27b.sh` 里有算力检查，A800 + FP8 会直接报错拦截。
  H20 96GB 也是 Hopper、支持 FP8，是备选。
- 账号须先实名认证 + 充值才能租机（新账号没有体验金，别指望白嫖额度）。
  计划工作量约 8~12 小时（环境+下模型 1~1.5h、冒烟 0.5h、272K 验证 1~2h、
  190 题精度自测 1~2h、并发压测 2~4h、调优 2~3h），建议先充 ¥150~200；
  按小时计费，用完记得释放实例。
- 租机配置建议：PyTorch 镜像（自带 CUDA 驱动和 Python）、系统盘 + 数据盘
  合计至少 150 GB 空闲（FP8 权重 31 GB + 环境 8 GB + 余量）；
  练 BF16 的话要 200 GB。
- 注意：第二模型如果选 GLM-5.3-Flash（328 GB），单卡跑不了，
  后面还得另租多卡节点；本包先只覆盖 Qwen3.8-27B 单卡排练。
- 实例到手后，把本目录整个拷过去（`scp -r` 或 AutoDL 的 JupyterLab 上传）。

## 1. 装环境

```bash
bash setup-vllm.sh
source .venv/bin/activate
```

预期输出：显卡型号和显存、`vllm 版本：0.34.0`、`torch 版本：2.14.0`。
首次安装要下载约 3 GB（torch），请耐心等；中途不要 Ctrl-C。

## 2. 下权重

```bash
bash download-model.sh          # 默认官方 FP8 版，约 31 GB
bash download-model.sh --bf16   # 只要在 FP8 出问题、或要做精度对照时才用
```

预期输出：`[完成] 权重就绪：./models/Qwen3.8-27B-FP8`，并打印
分片数（FP8 应为 66，BF16 应为 18）和总大小。ModelScope 下不动时脚本会
自动改走 HF（hf-mirror 镜像）；两个都失败时检查实例能不能上外网
（AutoDL 实例一般可以直接访问 modelscope.cn）。

## 3. 起服务

```bash
bash serve-qwen27b.sh
```

预期：先打印预检信息，然后 vLLM 加载权重（约 1~2 分钟），最后出现
`Application startup complete` 和 `Uvicorn running on http://0.0.0.0:8000`。
服务跑在前台，日志同时写到 `logs/vllm-serve-*.log`。

关键参数（都在 `serve-qwen27b.sh` 里，可用环境变量改）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `MAX_MODEL_LEN` | 272000 | 比赛要求 272K（超原生 262144 自动开 YaRN 外推，见下） |
| `KV_CACHE_DTYPE` | fp8 | KV 缓存用 FP8：262K 上下文只要约 8.6 GB（BF16 要 17.2 GB） |
| `GPU_MEM_UTIL` | 0.90 | 显存占用比 |
| `MAX_NUM_SEQS` | 32 | 最大并发序列数 |
| `--enable-prefix-caching` | 开 | 前缀缓存：比赛场景缓存命中 0.7/0.9 是主战场，必须开 |
| `--enable-per-request-metrics` | 开 | 每请求的耗时写进响应里，方便算 TTFT/TPOT |

**272K 上下文怎么来**：模型原生上限是 262144，比赛要 272000，
只差 3.8%。把 `MAX_MODEL_LEN=272000 bash serve-qwen27b.sh`，
脚本会自动加 YaRN 外推（factor≈1.04）。这是实验性路径：
先跑冒烟测试，再跑 EvalScope 精度对照，确认精度损失 ≤2% 再用；
如启动报错或精度不达标，退回 262144。

## 4. 冒烟测试（另开一个终端）

```bash
bash smoke-test.sh
```

预期：4 项全过——/health 通、模型列表里有 qwen3.8-27b、流式出 token、
/metrics 里有 `vllm:num_requests_running`。TTFT/TPOT 这里只测短提问
（正常 TTFT 应 <1 秒），**不代表长上下文能过 SLO**；长上下文压测
等组织方发评测包后用 AIPerf 跑。

## 5. 常见问题

| 现象 | 原因 / 办法 |
|---|---|
| `nvidia-smi` 找不到 | 实例没显卡或驱动没装好；AutoDL 换 PyTorch 镜像重开 |
| `pip install vllm` 很慢 / 超时 | 换清华源：`pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple` 后重跑；或直接重试 |
| `No space left on device` | /tmp 是小 tmpfs，别把虚拟环境建在 /tmp 下；本包默认建在当前目录的 `.venv` |
| 下载权重很慢 / 断掉 | ModelScope 失败会自动走 HF；两个都慢时检查实例出口带宽，或分时段重试；不要用 `git clone` 下权重 |
| 分片数 / 大小校验失败 | 下载不完整：删掉 `models/Qwen3.8-27B-FP8` 目录重下 |
| 起服务报 OOM | 先降级：`MAX_MODEL_LEN=32768 bash serve-qwen27b.sh`；再检查是不是误下了 BF16（55 GB）却按 FP8 的显存账算的 |
| `max_model_len 超出上下文` 报错 | 272000 需要 YaRN，脚本会自动加；如仍报错，把 MAX_MODEL_LEN 降回 262144 |
| `/metrics` 是空的 / 404 | vLLM 0.34 的 /metrics 默认就开，不用加参数；确认服务真的起好了再 curl |
| `vllm: error: unrecognized arguments` | vLLM 版本与脚本不符：`pip show vllm` 看版本，本包按 0.34.0 写的；新版本改了参数名时按 `vllm serve --help` 为准 |
| 生成速度明显慢 | 检查是不是误加了 `--enforce-eager`（只用于调试）；或并发太高，把 MAX_NUM_SEQS 降到 8 再测 |

## 6. 之后：8 卡跑 DeepSeek-V4.1-Flash（比赛必选模型）

已按 vLLM 官方 recipe（`recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash`）备好独立脚本（待 8×RTX 6000D 真机验证）：

1. **下载权重**：`MODEL_DIR=/data/models bash download-deepseek.sh`（默认 `deepseek-ai/DeepSeek-V4.1-Flash-1017`，约 511 GB / 48 分片，磁盘至少留 650 GB；支持断点续下，建议 `nohup` 后台跑）。
2. **启动服务**：`MODEL_DIR=/data/models bash serve-deepseek.sh`（默认 8 卡 TP、`MAX_MODEL_LEN=272000` 原生 1M 无需 YaRN、`--tokenizer-mode deepseek_v41`、`--tool-call-parser deepseek_v41`、`--reasoning-parser deepseek_v41`、`VLLM_ENGINE_READY_TIMEOUT_S=3600`）。
3. **显存与 Engram 账本**：权重总计约 476 GiB，其中 Engram 记忆表占 183.1 GiB；全局 KV 仅约 890 字节/token（272K 上下文 KV 仅 ~242 MB）。脚本默认开 `ENGRAM_CPU_OFFLOAD=true`（将 183 GiB 卸载到 pinned 主机内存，每卡显存驻留仅 ~36.6 GiB）；真机验证后可对照 `ENGRAM_CPU_OFFLOAD=false`、DEP（`TP_SIZE=1 DP_SIZE=8 ENABLE_EP=1`）、`LANGUAGE_MODEL_ONLY=1`（跳过 32 层 ViT）。
4. **可选加速**（每项均须重跑精度对照）：`DSPARK_TOKENS=5`（V4.1 已弃用 MTP，改用内置 DSpark 5-token 推测解码）、PD 分离。先保证「能跑 + 精度过关」，再抠 Goodput。

## 7. 下一步：评测（标准与测试集已在仓库里）

评测标准见 `EVALUATION.md`，测试集说明与下载见 `eval/`。

1. 下数据：`python3 eval/fetch-benchmarks.py`，落到 `eval/data/`。
2. 用 EvalScope 对本地服务跑四集精度，对照官方权重，确认损失 ≤2%。
3. 用 AIPerf 按并发阶梯 8/16/32/64/80 压测，核 TTFT/TPOT SLO。
4. 产出提交三件套：自测报告、服务地址或镜像 + 启动脚本、环境与优化说明表。

## 8. 报告站（仅队内可见）

报告站：https://hackathon-infra.pages.dev（Cloudflare Pages，Access 限队内邮箱）。
写报告见 `reports/README.md`；发布跑 `python3 site/deploy.py`。
