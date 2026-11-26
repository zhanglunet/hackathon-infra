# devlog：ThinkPad 构建 vLLM 镜像成功（第 5 次，清华 PyPI 源，2026-11-26）

## 背景

`hackathon-vllm:0.34.0`（vLLM 0.34.0 + FlashInfer）前 4 次在 ThinkPad 上都没建成：
第 1 次直连 docker.io TLS 握手超时，第 2 次 DNS 解析超时，第 3 次改走 DaoCloud 源、
基础镜像拉完后在 Dockerfile:52 算缓存键时读层文件报 I/O 错误，第 4 次清缓存重建、
过了 Dockerfile:52，又在 5/12 步 pip 装 vLLM 时因容器内访问 PyPI 中断而失败
（`Could not find a version that satisfies the requirement click`）。
用户 2026-11-26 07:46 点选“用清华源重建”。

## 改了什么（仓库没动，构建命令只加一个参数）

Dockerfile 本来就支持 `PIP_INDEX_URL`（第 42 行 `ARG PIP_INDEX_URL=""`，
第 80–81 行传了就 `pip config set global.index-url`），仓库 `main @ 92ddd533`
的文件一个字没改。构建命令：

```bash
docker build --platform linux/amd64 \
  --build-arg BASE_IMAGE=docker.m.daocloud.io/nvidia/cuda:13.0.0-devel-ubuntu24.04 \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  -t hackathon-vllm:0.34.0 .
```

构建机 ThinkPad（`thinkpad` 用户已在 docker 组，Docker 29.9.2，无 NVIDIA GPU）。
动手前核过：构建上下文 10 个顶层条目都在（与 main @ 92ddd533 一致）；
`docker images` 是空的，但构建缓存里还有 11.63GB（第 4 次拉下来的层），
所以不用重拉 26 分钟。日志单独记 `~/hackathon-build-5.log`（第 4 次那份不动）。

## 结果

- **成功**：07:48:18–08:34:13 CST，**45 分 55 秒**（比事先估计的 10 分钟长得多，
  装 vLLM 和最后解包花的时间比预期多；只看构建器计时的 #17 DONE 是 1076.9 秒，
  其中解包一步就占 375.3 秒）。
- 镜像 `hackathon-vllm:0.34.0` 已存在，ID `a29ffb1bd0c6`，26.2GB，
  `docker images` 与日志尾部双重核实。
- 构建期自检通过：第 7/12 步
  `python -c "import vllm, flashinfer; print('vllm', vllm.__version__)"`
  输出 `vllm 0.34.0`（日志第 1071–1072 行，共 1108 行）。
- 唯一告警非致命：`FromPlatformFlagConstDisallowed`（FROM --platform 用常量
  `linux/amd64`，line 39），只是 lint 提醒。

## 未验证

- ThinkPad 没有 NVIDIA 显卡，镜像还没真正点火跑过模型；真机点火等 11-30 比赛机。
- SGLang 镜像还没建；分发到 4 台比赛机（`scripts/distribute-image.sh`）还没跑。

## 同批进仓（本次提交另含，与构建无关）

- PR #15 已合（merge commit `88c36804`，肖娅的 loadtest 后续：共同 system prompt
  选项 + `router_check.py`）。
- ε 默认值 0.25 → 1.0（肖娅在 #14 下的验证：ε=0.25 时经路由器命中率 agent 0.54 /
  coding 0.78，ε=1.0 时 0.63 / 0.85，直连基线 0.65 / 0.87；PR #15 的 devlog 也
  证实 ε=1 后与直连一致）。只改了默认值，没加她提到的 `+2` 绝对余量。
- `serve-deepseek.sh`、`serve-glm.sh`、`serve-qwen27b.sh` 与
  `router/prefix_router.py` 恢复 755（92ddd533 把它们写成了 644）。
- `probe/probe_result.json` 与 `probe/cold_probe_result.json` 同步到 11-23 最终成绩
  （冷 41.09 / 43.8 秒；main 上原来是 53.64 / 55.76 秒的旧数）。
- `DAY1-RUNBOOK.md` 第 49 行命中率数字改成经路由器实测（0.63 / 0.85），
  注明直连基线 0.65 / 0.87。

## 遗留

- 肖娅 #14 评论里另外两处待确认没动：`/health` 在有副本挂掉时返回 503
  （降级时是否仍返回 200 待定）、共同前缀放 user 消息时需 `--prefix-chars`
  调到比前缀长（等官方数据下发看前缀位置和长度再定）。
