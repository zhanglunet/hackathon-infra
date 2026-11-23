# 提交材料补齐：说明表初稿 + 启动脚本补 flag（2026-11-23）

就绪检查（`2026-11-23-submission-readiness.md`）的后续：两件现在就能做的事都做了——
编环境与优化说明表初稿（G1）、给启动脚本补已验证 flag（G7 代码部分）。
只改静态文件，未部署、未跑 GPU，费用 $0；所有改动都只在已验证值上做，
未验证的一律标 TBD / 待真机验证。

## 1. 新增：ENVIRONMENT.md（环境与优化说明表，初稿）

仓库根目录新建 `ENVIRONMENT.md`（9,076 字节），按规则 §五三要素组织：

- §1 总览（按模型）：Qwen 已验证；DeepSeek 与第二款模型整行 TBD。
- §2 引擎与软件环境：vLLM 0.34.0、torch 2.14.0+cu130、Python 3.12、nvcc 12.8.61、
  排练硬件 Modal H100/H200、比赛硬件 RTX 6000D（TBD）。
- §3 上下文与并行切分：Qwen 272000（YaRN factor 1.0376）、TP=1、KV 容量
  2,955,530 token（步长 65536 时 2,592,357）；DeepSeek 并行切分、第二款模型 TBD。
- §4 量化与缓存/加速策略：逐项写值 + 验证来源与效果（FP8 per-tensor、
  FlashInfer 最快、chunked prefill 65536、编译空转、GEMM 无开关等）。
- §5 精度结果（代理集）：FP8 116/170 vs BF16 113/170，McNemar p≈0.55，
  附官方 190 题不公开的口径提醒。
- §6 性能结果：冷 41.09/43.8 秒、热 2.14 秒、TPOT 约 12.2 毫秒；
  AIPerf 阶梯、四档 SLO 场景、Blackwell 冷 TTFT 全部 TBD。
- §7 TBD 清单 T1–T7（真机复测后重填）；§8 来源（规则 + 14 篇 devlog）。

未编造任何未实测数字：凡依赖比赛机的写 TBD 并标注“待真机验证”。

## 2. serve-qwen27b.sh：补 4 个已验证项

| # | 改动 | 已验证值（来源） | 核对 |
|---|---|---|---|
| 1 | `export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`（无条件） | Modal `_CONTAINER_ENV` 同款；2026-11-21 不设它时 vLLM 报 ValueError、容器循环重启约 11 分钟 | serve:100 / modal:112，在位 |
| 2 | `--max-num-batched-tokens`（环境变量可配，默认 65536） | 第 1–3 轮：65536（262144 初始化失败）；副作用 KV 容量 2,955,530→2,592,357 token | serve:93 + 命令行，在位 |
| 3 | `--attention-backend`（环境变量可配，默认 FLASHINFER；空则不传、走 FA3） | 第 6 轮最快：冷 41.09/43.8 秒，比 FA3 快约 21%；前提 flashinfer + nvcc，否则 warmup 报错（第 5 轮） | serve:94 + `ATTN_ARGS`，在位 |
| 4a | `--gdn-prefill-backend`（环境变量可配，默认不传） | 默认 auto；GDN 仅约 1.4 秒（3%），triton 轮无改善 | serve:95 + `GDN_ARGS`，在位 |
| 4b | `--compilation-config`（环境变量可配，默认不传） | 第 4 轮：mode=3 对本模型空转（不支持 torch.compile），传了也不提速；留开关只为以后换模型复用 | serve:96 + `COMP_ARGS`，在位 |
| 5 | `export VLLM_USE_BREAKABLE_CUDAGRAPH`（默认 0） | 第 4 轮：vLLM 默认自动开 breakable 图、把 mode 置回 NONE；设 0 退出该路径；第 4–6 轮均用 0 | serve:103，在位 |

注意两处默认值与 modal 代码默认值不同，都是故意的：serve 脚本 baked in 的是
**调优轮次实际用的值**（65536、breakable=0、FLASHINFER），而 modal 代码保持中性
默认值、靠环境变量覆盖（`MAX_NUM_BATCHED_TOKENS` 默认 8192、`VLLM_USE_BREAKABLE_CUDAGRAPH`
默认 "1"、`ATTENTION_BACKEND` 默认 ""）。调优轮次的环境变量传值与 serve 默认值一致。

另：文件头补了“已验证的调优项”说明段（每项括号里写来源与效果）、标志核对段补了
4 个新 flag、备选命令段加了 FlashInfer 起不来时回落 FA3 的应急命令
（`ATTENTION_BACKEND= bash serve-qwen27b.sh`）。

## 3. setup-vllm.sh：加 flashinfer 与 nvcc 安装（第 5/6 步）

- 步骤从 5 步改成 6 步（1/6–6/6），文件头“做什么”与注意事项同步更新；
  原来“本脚本不装 flashinfer”那句已删，改为第 5 步的说明。
- flashinfer：先 `python -c "import flashinfer"` 核对，缺了再 `pip install flashinfer-python`
  （vLLM 0.34.0 安装时通常会连带装，这里是显式核对）；装完再验一次 import。
- nvcc：先 `command -v nvcc` 核对，有则打印版本跳过；没有则走 apt 装
  `cuda-nvcc-12-8`（与 vLLM 0.34.0 pip 轮子的 CUDA 12.8 匹配），前提是 sudo 可用
  且 NVIDIA CUDA apt 源已配好；源没配好时只警告、不报错（FA3 后端不用 nvcc，
  serve 时设 `ATTENTION_BACKEND=""` 即可回落）；apt 失败也不让整个脚本死掉
  （`|| echo "[警告] ..."`），最后再验一次 `nvcc --version`。
- 校验步（6/6）新增 flashinfer import 与 nvcc 校验行。

未动 `download-model.sh`（DeepSeek 下载脚本是另一件活，等机器与模型定了再做）；
未动 modal 代码。

## 4. 静态验证（只做这些，没上 GPU）

- `bash -n serve-qwen27b.sh`：通过；`bash -n setup-vllm.sh`：通过。
- flag 逐项核对（见 §2 表）：4 个已验证项 + 2 个环境变量导出全部在位，
  默认值与调优轮次实际传值一致；`vllm serve` 命令行里 4 组新参数全部在位。
- 未验证项：两个脚本的改动都**没在有 GPU 的机器上跑过**——`serve-qwen27b.sh`
  的目标仍是裸机单卡、未在 Blackwell 上验证；`setup-vllm.sh` 的 apt 装
  `cuda-nvcc-12-8` 一步在有源/无源两种分支下的行为都只做了静态阅读，
  待真机验证（见就绪检查 G7）。

## 5. 待真机验证（与就绪检查的缺口对应）

- G1：说明表初稿已编，真机复测后按 T1–T7 重填。
- G7 代码部分已做；验证部分仍待比赛机（Blackwell）或另租 Hopper 卡。
- 未做的（仍缺）：成型自测报告（G2，依赖官方包 + 比赛机）、DeepSeek 部署验证
  （G3）、容器镜像/服务地址（G4）、第二款模型（G5，待用户定）、官方包（G8）、
  SLO 最终锁定与提交渠道/截止时间（G9，依赖组织方）。
