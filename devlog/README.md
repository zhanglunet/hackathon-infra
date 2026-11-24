# 开发日志

每次部署验证一篇，文件名 `YYYY-MM-DD-主题.md`。只写已核实的事实，
不确定的写“未核实”；不写任何密钥、token、密码、验证码。

| 日期 | 部署对象 | 结果 | 链接 |
|---|---|---|---|
| 2026-11-21 | Modal 部署代码（Qwen3.8-27B FP8） | 已验证，见 H100 冒烟篇 | [2026-11-21-modal-27b-fp8.md](2026-11-21-modal-27b-fp8.md) |
| 2026-11-21 | Modal H100 冒烟（FP8 短上下文） | 通过 3/4，复测后实质全过 | [2026-11-21-modal-h100-冒烟.md](2026-11-21-modal-h100-冒烟.md) |
| 2026-11-21 | Modal H200 272K 长上下文探测 | 装得下，TPOT 通过，冷 TTFT 51–53 秒爆 SLO | [2026-11-21-modal-h200-272k.md](2026-11-21-modal-h200-272k.md) |
| 2026-11-21 | Modal H200 冷 TTFT 调优（3 轮） | 未达 15 秒目标：262144 初始化失败，65536 只快 1.5%，VLLM_COMPILE 静默降级回 eager；冷 TTFT 仍约 51 秒 | [2026-11-21-modal-h200-冷ttft调优.md](2026-11-21-modal-h200-冷ttft调优.md) |
| 2026-11-21 | Modal H200 profiling（瓶颈定位） | 瓶颈已定位：全注意力 FA3 约 65% + FP8 GEMM 约 25%，GDN 仅约 3%；GDN triton 轮 53.04 秒无改善；mode 降级原因已在源码定位（Qwen3.5 默认 breakable graphs） | [2026-11-21-modal-h200-profiling.md](2026-11-21-modal-h200-profiling.md) |
| 2026-11-21 | 资源现状盘点 | 已核实，见正文 | [2026-11-21-资源就绪.md](2026-11-21-资源就绪.md) |
| 2026-11-22 | Modal H100 EvalScope 精度自测（FP8） | AIME 17/30=56.67%，GPQA 35/50=70%，LCB 17/40=42.5%；BFCL 未跑（ModelScope 卡未批 + 本地文件 schema 不匹配）；Modal 并发修复（@modal.concurrent，吞吐 80→~850 tok/s） | [2026-11-22-modal-h100-evalscope.md](2026-11-22-modal-h100-evalscope.md) |
| 2026-11-22 | Modal H100 bfcl_eval CLI 补跑 BFCL v3（FP8） | simple 前 50 题 **47/50=94.00%**；原生 tools 部署未开不可用，改走 qwen_agent；v3 数据垫片塞进 v4 loader；全程约 7 分钟，约 $0.47 | [2026-11-22-modal-h100-bfcl.md](2026-11-22-modal-h100-bfcl.md) |
| 2026-11-22 | Modal H200 冷 TTFT 调优第 4 轮（关 breakable cudagraph 再试 mode=3） | mode 保持 VLLM_COMPILE (3)，降级已修；但模型不支持 torch.compile，无 compiled submodule，图回退 FULL_DECODE_ONLY；冷 TTFT 53.64 / 55.76 秒、热 2.37 秒，无改善 | [2026-11-22-modal-h200-冷ttft-mode3.md](2026-11-22-modal-h200-冷ttft-mode3.md) |
| 2026-11-22 | Modal H200 冷 TTFT 调优第 5 轮（全注意力后端对照 FlashInfer / Triton vs FA3） | FA3 冷 52.52 / 54.94 秒、热 2.29 秒仍最快；FlashInfer 引擎起不来（镜像缺 nvcc，JIT 编译失败）；Triton 冷 252.0 / 252.41 秒、热 4.25 秒，慢约 4.8 倍 | [2026-11-22-modal-h200-注意力后端对照.md](2026-11-22-modal-h200-注意力后端对照.md) |
| 2026-11-22 | Modal H100 BF16 对照（Qwen3.8-27B 官方权重 vs FP8） | 170 题合计 FP8 116 题、BF16 113 题，无净损失；按集：AIME 14/30 vs 17/30、GPQA 34/50 vs 35/50、LCB 17/40 持平、BFCL 48/50 vs 47/50；差异在噪声内（McNemar p≈0.55） | [2026-11-22-modal-h100-bf16.md](2026-11-22-modal-h100-bf16.md) |
| 2026-11-22 | Modal H200 冷 TTFT 调优第 6 轮（给镜像加 nvcc 再测 FlashInfer） | FlashInfer 终于跑起来，比 FA3 快约 21%：冷 41.09 / 43.8 秒、热 2.14 秒、TPOT 约 12.2 毫秒；15 秒目标仍远；镜像开关 `USE_CUDA_IMAGE=1` 走 CUDA 12.8 devel 镜像，nvcc 12.8.61，torch 却是 cu130（版本错位，JIT 仍通；DeepGEMM 被禁） | [2026-11-22-modal-h200-flashinfer-nvcc.md](2026-11-22-modal-h200-flashinfer-nvcc.md) |
| 2026-11-23 | Modal H200 冷 TTFT 调优第 7 轮（DeepGEMM 代码核实） | DeepGEMM 不适用：本模型是 per-tensor FP8，DeepGEMM 只在 block 路候选表里，`use_deep_gemm` 是死属性；换 13.0 镜像也改变不了内核选择；未上 GPU，$0 | [2026-11-23-modal-h200-deepgemm.md](2026-11-23-modal-h200-deepgemm.md) |
| 2026-11-23 | Modal H200 冷 TTFT 调优第 8 轮（FlashInfer FP8 线性核代码核实） | FlashInfer 线性核两道门都拦：算力须 ≥100（H200 是 90），且激活须 per-tensor（本模型是动态 per-token）；`--linear-backend` 也绕不过；未上 GPU，$0 | [2026-11-23-modal-h200-flashinfer-linear.md](2026-11-23-modal-h200-flashinfer-linear.md) |
| 2026-11-23 | 提交材料就绪检查（对照比赛规则核三样交件） | 三样全不齐：自测报告半缺（性能部分空白、只 1 款模型有数、官方包未到）、服务脚本半缺（缺 4 个已验证 flag、无 DeepSeek 脚本、无镜像、未在裸机验证）、说明表缺失；关键风险是时间，机器须 11-24 至 11-25 到手才排得开 | [2026-11-23-submission-readiness.md](2026-11-23-submission-readiness.md) |
| 2026-11-23 | 提交材料补齐：说明表初稿 + 启动脚本补 flag | 新建 ENVIRONMENT.md（初稿，TBD 项 T1–T7 待真机重填）；serve-qwen27b.sh 补 4 个已验证项（VLLM_ALLOW_LONG_MAX_MODEL_LEN=1、--max-num-batched-tokens 65536、--attention-backend FLASHINFER、GDN/编译开关）；setup-vllm.sh 加 flashinfer + nvcc 安装（第 5/6 步）；bash -n 均过，未上 GPU、$0 | [2026-11-23-submission-prep.md](2026-11-23-submission-prep.md) |
| 2026-11-23 | issue #5 三项代码修复（裸机下载入口、Modal 评测隔离、测量工具校准） | download-model.sh：先建目录再 df、完整权重判定、HF 回退续下；run_modal_p0_eval.sh：部署成功才置 DEPLOYED、残留预检、端点验名、BFCL 垫片改隔离环境；check_truncation.py 三态 + 稳定题目 ID + 逐集配对（离线回归与旧配对逐项一致，p=0.3438）；probe 两脚本计 reasoning、流失败非零退出（fake SSE 四组全过）；$0，未上 GPU | [2026-11-23-issue5-fixes.md](2026-11-23-issue5-fixes.md) |
| 2026-11-24 | issue #6 P0-3 native BFCL 验证（Modal H100） | 原生 tools 探测通过（tool_calls 正常、reasoning 字段在）；BFCL v3 simple 前 50 题 **49/50=98.00%**（上一轮 qwen_agent 47/50），唯一错题仍是 _13（interval 整数 vs 浮点）；费用 $0.27、实付 $0.00，部署已停、/health 404；cleanup 费用估算行有 NameError bug 未修 | [2026-11-24-issue6-p0-3.md](2026-11-24-issue6-p0-3.md) |
| 2026-11-24 | issue #6 P0-1/P0-2 FP8 全套 32K 精度重跑（Modal H100） | AIME 26/30=86.7%（4096 是 17/30）、GPQA 44/50=88.0%（4096 是 35/50），截断率 70%/38–48% 降到 10%/6%；BFCL native 49/50=98.00% 复现，错题仍是 _13；LCB 中断只跑了 27/40（26/27 通过，部分集有偏）；运行时重启致 trap 未停机，手动停，费用 $2.73、实付 $0.00，本月剩 $6.46；cleanup 费用估算 bug 已修 | [2026-11-24-issue6-p0-1-p0-2-fp8.md](2026-11-24-issue6-p0-1-p0-2-fp8.md) |
