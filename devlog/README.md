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
