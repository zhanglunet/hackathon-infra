# Modal 部署：Qwen3.8-27B vLLM 服务

在 Modal 上跑 Qwen3.8-27B 的 OpenAI 兼容服务（vLLM 0.34.0，FP8，
prefix caching，`/v1/chat/completions` + `/metrics`），给 2026-11-27
比赛做排练。代码见 `modal_vllm_qwen27b.py`，参数照抄
`serve-qwen27b.sh` / `download-model.sh`（模型名、启动参数以那两个
文件为准）。

## 前置条件

- Modal 账号已就绪（2026-11-21 核实：GitHub zhanglunet 登录，Starter
  套餐，已绑卡解锁每月 $30 额度，用量上限 $42.50）。
- 本机装 Modal CLI 并登录（token 从 Modal 控制台取，**不要写进仓库**）：
  ```bash
  pip install modal
  modal setup
  ```

## 步骤

```bash
# 1. 下权重进 Modal Volume（约 31 GB FP8，只跑一次；不挂 GPU，只花 CPU 时间）
modal run modal/modal_vllm_qwen27b.py::download_weights

# 2a. 临时服务（开发调试，Ctrl-C 停；URL 会打印在终端里）
modal serve modal/modal_vllm_qwen27b.py

# 2b. 长期服务（固定 URL；用完在 Modal 控制台 Apps 页停掉）
modal deploy modal/modal_vllm_qwen27b.py

# 3. 冒烟（4 项：/health、/v1/models、流式 TTFT/TPOT、/metrics）
modal run modal/modal_vllm_qwen27b.py --url https://<部署地址>
```

## GPU / 模型开关（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `MODAL_GPU` | `A100-80GB` | 显卡型号。FP8 冒烟切 `H100`；测 272K 切 `H200` |
| `QUANT` | `fp8` | `fp8`（Qwen/Qwen3.8-27B-FP8，7 分片）或 `bf16`（Qwen/Qwen3.8-27B，18 分片） |
| `MAX_MODEL_LEN` | `262144` | 上下文上限。测 272K 时设 `272000`（自动加 YaRN 外推，见下） |
| `MAX_NUM_SEQS` | `32` | 最大并发序列数 |
| `GPU_MEM_UTIL` | `0.90` | 显存占用比 |

常用组合：

```bash
# A100 上 BF16 短上下文冒烟（A100 跑不了 FP8，见注意事项）
QUANT=bf16 MAX_MODEL_LEN=32768 modal serve modal/modal_vllm_qwen27b.py

# H100 上 FP8 冒烟（原生上下文）
MODAL_GPU=H100 modal deploy modal/modal_vllm_qwen27b.py

# H200 上 FP8 测 272K（YaRN 外推，实验性：先跑精度对照，见 serve-qwen27b.sh）
MODAL_GPU=H200 MAX_MODEL_LEN=272000 modal deploy modal/modal_vllm_qwen27b.py
```

## 费用估算（$30/月额度，按秒计费）

| 显卡 | 单价 | $30 能跑多久 |
|---|---|---|
| A100-80GB | $2.50/时 | 约 12 小时 |
| H100 | $3.95/时 | 约 7.6 小时 |
| H200 141GB | $4.54/时 | 约 6.6 小时 |

权重存在 Modal Volume 里（1 TiB 免费额度），不用每次重下。
容器超时 4 小时（`timeout=4h`），闲置 10 分钟后自动缩容。
用量上限 $42.50，超了会先提示再扣费——跑之前看一眼控制台用量页。

## 注意事项

- **A100 跑不了 FP8 权重**：FP8 需要 Hopper（算力 9.0+）或 Blackwell
  的张量核，A100 是 Ampere（8.0）。代码里有和 `serve-qwen27b.sh`
  一样的算力检查，A100 + FP8 会直接报错并告诉你怎么改（切 H100/H200，
  或 A100 上改 BF16）。不要绕过这个检查。
- **272K 是 YaRN 外推**：模型原生上限 262144，比赛要 272000，只差
  3.8%。`MAX_MODEL_LEN=272000` 时代码自动加 YaRN（factor≈1.04），
  这是实验性路径：先跑 EvalScope 精度对照，确认损失 ≤2% 再用；
  如启动报错或精度不达标，退回 262144。
- **显存账本**（见 serve-qwen27b.sh）：FP8 权重约 30.9 GB + FP8 KV
  约 8.6 GB ≈ 40 GB，80 GB 卡够用；H200（141 GB）测 272K 更宽裕。
- **首次部署要留时间**：镜像要装 vLLM + torch（约 8 GB），权重
  31 GB，第一次 `deploy` + `download_weights` 预留 30~60 分钟。
- **不要把 token 写进仓库**：`modal setup` 的 token 只放本机，
  仓库是公开的，任何密钥、token、密码都不许进提交和日志。
- 部署验证结果记在 `devlog/` 里，每次验证一篇（见 devlog/README.md）。
