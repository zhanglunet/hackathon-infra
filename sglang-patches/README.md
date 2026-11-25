# SGLang SM120 补丁（ellswoau，v0.5.23，2026-11-25 取回）

## 来源

上游仓库 `ellswoau/deepseek-v4.1-flash-1017-tp-4-rtx-pro-6000`
（MIT；`runtime/flash_mla_sm120.py` 衍生自 SGLang，Apache-2.0；
`NOTICE` / `LICENSE.sybil` / `LICENSE.sglang` 三份许可文件原样附在
本目录），2026-11-25 按上游 main 树逐文件取回，字节数与上游树
逐项核对一致。对应 SGLang 镜像 `lmsysorg/sglang:v0.5.23-cu130`
（sha256 `f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f`；
sgl-kernel 0.4.10、flashinfer 0.7.2、torch 2.14.1+cu130）。

背景与实测见 `~/workspace/research_notes/sglang-pr-issue-yanjiu-20261124-0312/report.md`
（SGLang 研究）：上游实测硬件是 4x RTX PRO 6000 96GB（SM120），
我们的比赛卡是 8x RTX 6000D 84GB。**本目录只是把补丁取回来，
Day-1 A/B 门过了才上真机试，未验证、未采用。**

## 6 个修复，一览

| # | 文件 | 作用 | 效果（上游实测） |
|---|---|---|---|
| 1 | `runtime/flash_mla_sm120.patch`（13 行）+ `runtime/flash_mla_sm120.py`（已打补丁的完整文件）+ `runtime/flash_mla_sm120.upstream-v0.5.23.py`（上游原文件，供对照） | V4.1 ratio-1/2 的 256/128-token 页拆成 64-token 页，FlashInfer SM120 sparse-MLA prefill 才能接 | prefill 约 294 tok/s -> 约 8K tok/s（否则回退 Triton） |
| 2 | `adapter/sitecustomize.py` | ratio-1/2 indexer 强制 DeepGEMM split planner 元数据 | SM120 上 FP4 DeepGEMM indexer 能跑 |
| 3 | `adapter/sitecustomize.py` | 守卫 `make_candidate_indexer`（sglang#44445）：DeepGEMM paged sparse MQA logits 不可用时走 dense-blocks 路，保留 dense FP4 indexer | v0.5.23 在 SM120 上能启动（不修直接报启动错） |
| 4 | `adapter/sitecustomize.py`（opt-in，`DS41_DSV4_SWA_HOST_MULT`，默认 11） | 只放大 HiCache SWA host pool | 48K 对话在 535 万 token 流量后仍可复用（未修 235 万即丢） |
| 5 | `adapter/sitecustomize.py`（opt-in，`DS41_SM120_PAGED_INDEXER`，默认 1） | SM120 上 decode 走 paged indexer（上游只在 SM90 开） | 1M 上下文 agent 吞吐 +12~60%，C=16 由 698 -> 1,209 tok/s |
| 6 | `tuned_configs/`（11 个 JSON） | Triton block-FP8 GEMM 调参（SM120 无官方 config） | 单流 decode +40~55%，冷 prefill +28% |

## 怎么用（Day-1 A/B 门过了之后）

```bash
# 1. 修复 1：把 patch 打到 SGLang 0.5.23 的同名文件上
#    （或直接用本目录已打补丁的 runtime/flash_mla_sm120.py 覆盖）
# 2. 修复 2-5：把 adapter/sitecustomize.py 放进 PYTHONPATH，
#    启动时设 DS41_SM120_INDEXER_FIX=1（修复 5 默认开，设
#    DS41_SM120_PAGED_INDEXER=0 可关；修复 4 设 DS41_DSV4_SWA_HOST_MULT=1 可关）
# 3. 修复 6：tuned_configs/ 交给 SGLang 的 tuned config 目录加载
# 4. 主机准备：host/ds41-thp-shmem.conf（透明大页 shmem 设 advise，
#    详见 host/README.md）
```

上游启动要点（`launch.sh` / `compose.yaml` / `.env.example` 原样附在
本目录，供抄启动参数）：TP4/EP4、`dsv4` attention、
`flashinfer_mxfp4` MoE、FP8 KV、`SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`、
chunked prefill 4096、DSpark block 5。首次启动约 7 分钟。

验证修复 5：`tests/test_topk_v2_sm120.py`（paged top-k v2 与
torch.topk 逐项比对）。

## 注意事项（上游原话 + 我们的核对）

- **状态 experimental，单台机器实测。** 1M 上下文 32/32 needles、
  4.9 小时 soak 零引擎错误，但只在一台 96GB 卡机器上测过。
- **tuned_configs 的 device_name 是
  `NVIDIA_RTX_PRO_6000_Blackwell_Max-Q_Workstation_Edition`，
  我们的卡是 RTX 6000D，设备名不同，SGLang 不会自动匹配这 11 个
  config。** 做法二选一：改名后试（有风险），或在比赛机上用
  SGLang 自带 tuner 重调（stock tuner 搜不到 block [32,32]，
  须按上游做法处理）。
- **主机内存要约 400GB**（Engram 189GiB 锁定 + HiCache 约 162GB）；
  组织方说的 "190C1T" 若真是 1TB 内存则够，否则 Engram host 表
  须缩容。首次启动约 7 分钟、后续约 3 分钟。
- **必须 pin v0.5.23 且确认构建含 #45261**，否则 SM120 启动报
  DeepGEMM `t.is_cuda()` assertion。
- 我们的 vLLM 线上，DeepGEMM 与 FlashInfer 线性核已代码核实不适用
  （第 7/8 轮）；SGLang 这条线是另一套内核路径，不冲突，但 Day-1
  仍须先跑单机 272K 真实数再决定投不投（判据见筹备分析报告）。
