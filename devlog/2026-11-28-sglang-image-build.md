# devlog：ThinkPad 构建 SGLang 镜像两次失败（同一类 blob 读 I/O 错误，2026-11-28）

## 背景

目标镜像 `deepseek-v41-sglang-v0523-sm120:local`。`sglang-patches/Dockerfile`
是在 stock 镜像 `lmsysorg/sglang:v0.5.23-cu130`（digest 钉死
`sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f`）
之上 COPY 补丁文件的薄层构建，5 步，无 pip、无源码编译。

构建机 ThinkPad（Docker 29.9.2，无 GPU）。构建上下文 26 个文件，4 个关键文件
sha 与本地逐一对上（Dockerfile `0462e056…`、runtime/flash_mla_sm120.py
`41afb1af…`、adapter/sitecustomize.py `52e03a1f…`、launch.sh `b4c97fdd…`）。
Dockerfile 无 build-arg，DaoCloud 镜像源只改了 ThinkPad 副本的 FROM 行，
仓库原文件没动。用户 2026-11-28 08:57 点选“开建 SGLang 镜像”。

## 第 1 次（09:01:14–10:12:44 JST，约 1 小时 11 分钟）

- 启动 08:01:14 CST，进程 pid 862436，日志 `~/hackathon-build-sglang.log`
  （2948 行，290756 字节）。
- 死在第 2/5 步 COPY runtime/flash_mla_sm120.py 算缓存键时，报错原文：

```
ERROR: failed to build: failed to solve: failed to compute cache key: read /var/lib/containerd/io.containerd.content.v1.content/blobs/sha256/f99bfd0537cd8bfbfc0cc5e78a973b39c02caeb095a727163ec63c28864200c4: input/output error
```

- 基础镜像层当时还在拉（09:30 巡查时某层 1.21GB/2.84GB），镜像没建成。
- 10:15 JST 巡查确认进程已不在、镜像不存在。

## 中间诊断：docker pull 也失败

- 先 `docker pull` 把基础镜像拉成正式镜像（想复用第 1 次已下载的层），也失败：
  解压 f99bfd05 层时报同一 I/O 错误。证实坏 blob 就是底镜像的一层，
  复用已下载层这条路走不通。日志 `~/sglang-pull-2.log`（551 字节）。
- 用户随后点选“清缓存重建”。

## 清缓存（只清 buildx 缓存）

- `docker buildx prune -af`：构建缓存 50.6GB（98 条）清到 2 条、4.096kB。
- 没跑 `docker system prune`，vLLM 镜像 `hackathon-vllm:0.34.0`
  （`a29ffb1bd0c6`，26.2GB）完好。

## 第 2 次（10:37:19–11:25:14 JST，约 48 分钟）

- 启动 09:37:19 CST，进程 pid 886311，日志 `~/hackathon-build-sglang-2.log`
  （2431 行，235866 字节，2 处 ERROR）。
- 同一步、同一类错误，但坏 blob 不同（`422df934…`），报错原文：

```
#6 ERROR: read /var/lib/containerd/io.containerd.content.v1.content/blobs/sha256/422df9348ee27f10873c491986aea9b2bc83b8a026331e26b83740902588fa54: input/output error
ERROR: failed to build: failed to solve: failed to compute cache key: read /var/lib/containerd/io.containerd.content.v1.content/blobs/sha256/422df9348ee27f10873c491986aea9b2bc83b8a026331e26b83740902588fa54: input/output error
```

- 关键：这次坏的层是全新下载的。日志里 f99bfd05 层下载 400.3 秒、解压 37.0 秒；
  422df934 层下载 361.8 秒、解压 15.9 秒，两层都在第 2 次里重新下载并解压成功，
  之后 COPY 步读取 422df934 才报 I/O 错误。
- 所以不是单个缓存脏项：清掉全部缓存、重新下载，还是在不同 blob 上撞上
  同样的读错误。
- 11:30 JST 巡查确认进程已不在、镜像不存在；盯梢任务已按规则自删。

## 磁盘现状（已核实，未定论）

- ThinkPad 根盘是 500G 机械硬盘 ST500LT012-1DG142（SATA，rotational=1），
  LUKS 加密 + btrfs；464G 总量，393G 空闲（16%）。
- `btrfs device stats /` 全 0（read_io_errs 0、corruption_errs 0 等）；
  机器已连续运行 12 天没重启；从没跑过 scrub。
- 但我这边无免密 sudo，dmesg 和内核日志看不到，smartctl 也没装。
  btrfs 零错误这个数和“盘要坏了”对不上，盘的问题既没坐实也没排除。
- 待用户在 ThinkPad 本地跑：dmesg 查 I/O 错 + 装 smartmontools 跑
  `smartctl -H /dev/sda`；若都干净，再跑 btrfs scrub 全盘校验。

## 未验证 / 遗留

- SGLang 镜像仍未建成；vLLM 镜像完好；分发到 4 台比赛机未跑。
- R2 2026-11-30 9:00 开赛，SGLang 线是双线 2+2 的一半；下一步等用户定
  （查盘结果 / 换机器建 / 直接重建）。
- 盯梢任务已删；再建需重新设盯梢。
