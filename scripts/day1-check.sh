#!/usr/bin/env bash
# day1-check.sh — 比赛机到位第一天的首检（只读：不装东西、不改配置、不下载权重）
#
# 用法：
#   bash scripts/day1-check.sh                       # 自动搜权重目录
#   bash scripts/day1-check.sh --model-dir /data/xxx # 指定权重根目录
#
# 退出码：0 = 无 FAIL；1 = 有 FAIL（WARN 不影响退出码）。
# 在本机（无 GPU）跑会报 FAIL，是预期行为，只看脚本本身有没有报错。
#
# 查什么（按 2026-11-25 已知的比赛实际环境）：
#   GPU 8x RTX 6000D 84GB / CPU 约 190 核 / 内存约 1TB / 磁盘 /data/nvme1n1
#   系统 Ubuntu 24.04.4 LTS / docker + 国内源 / 12x400G RoCE（8 个可用）
#   权重在 NFS 只读共享存储（挂载点未知，自动搜）

MODEL_DIR_ARG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --model-dir) MODEL_DIR_ARG="$2"; shift 2 ;;
        --model-dir=*) MODEL_DIR_ARG="${1#--model-dir=}"; shift ;;
        *) echo "[用法] bash scripts/day1-check.sh [--model-dir /path]"; exit 2 ;;
    esac
done

PASS=0; WARN=0; FAIL=0
ok()   { PASS=$((PASS+1)); echo "[PASS] $1"; }
warn() { WARN=$((WARN+1)); echo "[WARN] $1"; }
fail() { FAIL=$((FAIL+1)); echo "[FAIL] $1"; }
note() { echo "       $1"; }

echo "== Day-1 首检：$(date '+%Y-%m-%d %H:%M:%S') / 主机 $(hostname) =="

echo "--- 1. 系统 ---"
if [ -f /etc/os-release ]; then
    PRETTY="$(grep '^PRETTY_NAME=' /etc/os-release | cut -d= -f2 | tr -d '"')"
    echo "系统：$PRETTY"
    case "$PRETTY" in
        *Ubuntu*24.04*) ok "Ubuntu 24.04（与组织方公布一致）" ;;
        *) warn "系统不是 Ubuntu 24.04，Dockerfile 基础镜像与 setup-vllm.sh 均按 Ubuntu 写" ;;
    esac
else
    fail "读不到 /etc/os-release"
fi

echo "--- 2. CPU / 内存 ---"
NPROC="$(nproc 2>/dev/null || echo 0)"
MEM_GB="$(free -g 2>/dev/null | awk '/^Mem:/ {print $2}')"
echo "CPU 核数：$NPROC / 内存：${MEM_GB} GB"
[ "$NPROC" -ge 100 ] 2>/dev/null && ok "CPU 核数 $NPROC（组织方口径约 190 核）" || warn "CPU 核数 $NPROC，低于组织方口径约 190 核"
if [ "${MEM_GB:-0}" -ge 400 ] 2>/dev/null; then
    ok "内存 ${MEM_GB} GB：DeepSeek Engram host 表（约 189GiB）+ HiCache（约 162GB）够放"
else
    warn "内存 ${MEM_GB} GB < 400GB：Engram host 表 + HiCache 可能放不下，须缩容"
fi

echo "--- 3. GPU ---"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "找不到 nvidia-smi（无驱动或无显卡）"
else
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -dc '0-9')"
    GPU_MEM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)"
    echo "显卡数：$GPU_COUNT / 单卡显存：${GPU_MEM} MiB"
    [ "$GPU_COUNT" = "8" ] && ok "8 张卡" || fail "显卡数 $GPU_COUNT，不是 8 张"
    if [ "${GPU_MEM:-0}" -ge 80000 ] 2>/dev/null; then ok "单卡显存 ${GPU_MEM} MiB（RTX 6000D 84GB）"; else fail "单卡显存 ${GPU_MEM} MiB，小于 80GB"; fi
fi

echo "--- 4. 磁盘 ---"
df -h / /data/nvme1n1 "$HOME" 2>/dev/null | tail -n +2
if [ -d /data/nvme1n1 ]; then ok "数据盘 /data/nvme1n1 存在"; else warn "没有 /data/nvme1n1（组织方 os-release 截图里的工作目录）"; fi

echo "--- 5. 网络（每项 10 秒超时） ---"
check_url() { # $1=名 $2=URL
    CODE="$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$2" 2>/dev/null || echo 000)"
    case "$CODE" in
        200|301|302|401) ok "$1 通（HTTP $CODE）" ;;
        *) warn "$1 不通（HTTP $CODE）：$2" ;;
    esac
}
check_url "通用外网" "https://www.baidu.com"
check_url "PyPI" "https://pypi.org/simple/"
check_url "GitHub raw" "https://raw.githubusercontent.com/vllm-project/vllm/v0.34.0/README.md"
check_url "ModelScope" "https://modelscope.cn"
check_url "HuggingFace" "https://huggingface.co"
check_url "docker.io 镜像库" "https://registry-1.docker.io/v2/"
note "docker.io 不通是 R1 已知的坑：改国内源（见 Dockerfile 注释与 Day-1 runbook）"

echo "--- 6. Docker ---"
if ! command -v docker >/dev/null 2>&1; then
    fail "找不到 docker（组织方口径：测试在 docker 容器内跑）"
else
    docker version --format 'client {{.Client.Version}} / server {{.Server.Version}}' 2>/dev/null || docker --version
    MIRRORS="$(docker info --format '{{.RegistryConfig.Mirrors}}' 2>/dev/null)"
    echo "registry mirrors：$MIRRORS"
    case "$MIRRORS" in
        "[]"|"") warn "docker 未配国内镜像源；docker.io 不通时先配源再 build" ;;
        *) ok "docker 已配镜像源" ;;
    esac
fi

echo "--- 7. RoCE / RDMA ---"
if [ -d /sys/class/infiniband ] && [ -n "$(ls /sys/class/infiniband 2>/dev/null)" ]; then
    ok "RDMA 设备：$(ls /sys/class/infiniband | tr '\n' ' ')"
else
    warn "未见 /sys/class/infiniband 设备（组织方口径 12x400G RoCE 只用 8 个；若无 RDMA，跨机走 TCP）"
fi
ip -br link 2>/dev/null | awk '{print $1, $2}' | head -20

echo "--- 8. 权重（NFS 只读共享存储，自动搜） ---"
echo "NFS 挂载："
NFS_MOUNTS="$(mount -t nfs,nfs4 2>/dev/null)"
if [ -n "$NFS_MOUNTS" ]; then echo "$NFS_MOUNTS"; else echo "（无 NFS 挂载）"; fi
CANDIDATES=""
[ -n "$MODEL_DIR_ARG" ] && CANDIDATES="$CANDIDATES $MODEL_DIR_ARG"
[ -n "${MODEL_DIR:-}" ] && CANDIDATES="$CANDIDATES $MODEL_DIR"
CANDIDATES="$CANDIDATES /data/nvme1n1 /data/models /models ./models /mnt/models"
FOUND_ANY=0
check_model() { # $1=显示名 $2=子目录 $3=预期分片数（0=只看 index.json）
    local name="$1" sub="$2" expect="$3" root found=""
    for root in $CANDIDATES; do
        if [ -f "$root/$sub/config.json" ]; then found="$root/$sub"; break; fi
    done
    if [ -z "$found" ]; then
        fail "找不到 $name（搜过：$CANDIDATES）"
        return
    fi
    FOUND_ANY=1
    if [ "$expect" = "0" ]; then
        if [ -f "$found/model.safetensors.index.json" ]; then ok "$name 在 $found（index.json 在）"; else warn "$name 在 $found，但缺 model.safetensors.index.json"; fi
    else
        local n
        n="$(ls "$found"/*.safetensors 2>/dev/null | wc -l | tr -dc '0-9')"
        if [ "$n" = "$expect" ]; then ok "$name 在 $found，分片 $n/$expect"; else warn "$name 在 $found，分片 $n/$expect（可能未下完或版本不同）"; fi
    fi
    note "MODEL_DIR=$found 可直接给 serve-*.sh 用（MODEL_DIR=$found bash serve-xxx.sh）"
}
check_model "DeepSeek-V4.1-Flash-1017" "DeepSeek-V4.1-Flash-1017" 48
check_model "GLM-5.3-Flash" "GLM-5.3-Flash" 0
check_model "Qwen3.8-27B-FP8" "Qwen3.8-27B-FP8" 66
[ "$FOUND_ANY" = "1" ] || fail "三个模型一个都没找到：先找 NFS 挂载点，再核对目录名"

echo "--- 9. 工具链 ---"
python3 --version 2>/dev/null || fail "找不到 python3"
python3 -c "import vllm; print('vllm', vllm.__version__)" 2>/dev/null || warn "未装 vllm（setup-vllm.sh 可装，或走 Dockerfile）"
command -v nvcc >/dev/null 2>&1 && nvcc --version | tail -1 || warn "无 nvcc（FlashInfer JIT 要它；Dockerfile 的 devel 镜像自带）"
python3 -c "import flashinfer; print('flashinfer OK')" 2>/dev/null || warn "未装 flashinfer（setup-vllm.sh 第 5 步补装）"

echo
echo "== 汇总：PASS $PASS / WARN $WARN / FAIL $FAIL =="
if [ "$FAIL" -gt 0 ]; then
    echo "结论：有 FAIL，先修 FAIL 再起服务。"
    exit 1
else
    echo "结论：首检通过（WARN 只看不拦）。下一步：冒烟 GLM -> 起 DeepSeek -> SGLang A/B 门。"
    exit 0
fi
