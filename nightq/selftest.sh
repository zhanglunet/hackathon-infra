#!/usr/bin/env bash
# selftest.sh — 不用 GPU、$0，约 1 分钟验证夜跑队列的每种结局都处理对了。
#
# 用假服务（loadtest/mock_server.py）故意制造各种情况，核对 nightq.py 的判定：
#   正常跑完、服务起不来、等就绪超时、服务中途挂掉（进程退出 / 探活不通）、步骤超时、
#   步骤失败（停 / 继续）、任务停用、端口被占、续跑、重跑失败项、被打断、同机互斥、写错键名。
#
# 用法：bash nightq/selftest.sh

set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python3}"
PORT="${PORT:-18771}"
WORK="${WORK:-$(mktemp -d -t nightq-selftest-XXXX)}"
LOCK="$WORK/lock"
NQ=("$PY" nightq/nightq.py)
PASS=0
FAIL=0

ok()  { echo "[通过] $1"; PASS=$((PASS+1)); }
bad() { echo "[失败] $1"; FAIL=$((FAIL+1)); }
check() { if eval "$2"; then ok "$1"; else bad "$1（条件：$2）"; fi; }
status() { "$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['status'])" "$1/job.json" 2>/dev/null || echo "无记录"; }
reason() { "$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['reason'])" "$1/job.json" 2>/dev/null; }
port_open() { "$PY" -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('127.0.0.1',$PORT))==0 else 1)"; }
STRAY=""
cleanup() { [ -n "$STRAY" ] && kill "$STRAY" 2>/dev/null; pkill -f "mock_server.py --port $PORT" 2>/dev/null; true; }
trap cleanup EXIT

if port_open; then echo "[错误] 端口 $PORT 已被占用，换一个：PORT=xxxx bash nightq/selftest.sh"; exit 1; fi
MOCK="$PY loadtest/mock_server.py --port $PORT --speedup 10"

echo "== 1/7 主队列：各种结局各来一个 =="
cat >"$WORK/main.ini" <<EOF
[DEFAULT]
ready_url = http://127.0.0.1:$PORT/health
ready_timeout = 30
ready_interval = 1
step_timeout = 60
settle = 0
stop_grace = 5
health_interval = 1
health_fail_limit = 3

[ok]
start = $MOCK
env =
    FOO=bar
steps =
    curl -sf http://127.0.0.1:$PORT/v1/models
    test "\$FOO" = bar && test "\$NIGHTQ_JOB" = ok && echo done > \$NIGHTQ_OUT/marker

[start-exits]
start = echo boom; exit 3

[ready-timeout]
start = sleep 31337
ready_timeout = 3

[server-dies]
start = $MOCK & PID=\$!; sleep 4; kill \$PID; wait
steps =
    sleep 60

[detached-dies]
detached = true
start = $MOCK >/dev/null 2>&1 & echo \$! > \$NIGHTQ_OUT/pid; (sleep 5; kill \$(cat \$NIGHTQ_OUT/pid)) >/dev/null 2>&1 &
stop = kill \$(cat \$NIGHTQ_OUT/pid) 2>/dev/null; true
steps =
    sleep 60

[step-timeout]
start = $MOCK
step_timeout = 3
steps =
    sleep 60

[fail-stop]
steps =
    false
    echo never > \$NIGHTQ_OUT/never

[fail-continue]
on_step_fail = continue
steps =
    false
    echo yes > \$NIGHTQ_OUT/ran

[disabled]
enabled = false
steps =
    echo x

[last-ok]
start = $MOCK
steps =
    curl -sf http://127.0.0.1:$PORT/health
EOF
O="$WORK/main"
"${NQ[@]}" "$WORK/main.ini" --out "$O" --lock "$LOCK" >"$WORK/main.out" 2>&1
RC=$?
check "有失败任务时退出码是 1" "[ $RC -eq 1 ]"
check "ok：成功，两步都跑了，env 和 NIGHTQ_* 变量生效" "[ \"\$(status $O/ok)\" = ok ] && [ -f $O/ok/marker ]"
check "start-exits：判服务没起来" "[ \"\$(status $O/start-exits)\" = start_failed ]"
check "ready-timeout：判服务没起来，且残留进程已清掉" "[ \"\$(status $O/ready-timeout)\" = start_failed ] && ! pgrep -f 'sleep 31337' >/dev/null"
check "server-dies：进程退出，判服务中途挂了" "[ \"\$(status $O/server-dies)\" = server_died ]"
check "detached-dies：探活连续不通，判服务中途挂了" "[ \"\$(status $O/detached-dies)\" = server_died ] && reason $O/detached-dies | grep -q 探不通"
check "step-timeout：判步骤超时" "[ \"\$(status $O/step-timeout)\" = step_timeout ]"
check "fail-stop：步骤失败后不跑后面的" "[ \"\$(status $O/fail-stop)\" = step_failed ] && [ ! -e $O/fail-stop/never ]"
check "fail-continue：步骤失败后继续跑后面的" "[ \"\$(status $O/fail-continue)\" = step_failed ] && [ -f $O/fail-continue/ran ]"
check "disabled：没跑" "[ ! -e $O/disabled ]"
check "last-ok：前面各种失败之后端口都放开了，最后一个任务照常成功" "[ \"\$(status $O/last-ok)\" = ok ]"
check "summary.md 列出了全部 10 个任务并附失败日志" "[ \$(grep -c '^| ' $O/summary.md) -eq 11 ] && grep -q '失败任务的日志尾巴' $O/summary.md"
check "跑完后端口是空的" "! port_open"

echo "== 2/7 续跑：成功的跳过，失败的默认不重跑；--retry-failed 才重跑 =="
"${NQ[@]}" "$WORK/main.ini" --out "$O" --lock "$LOCK" >"$WORK/resume.out" 2>&1
check "续跑没有重跑任何任务" "! ls -d $O/*.attempt* >/dev/null 2>&1 && grep -q '上次已成功，跳过' $WORK/resume.out"
"${NQ[@]}" "$WORK/main.ini" --out "$O" --lock "$LOCK" --retry-failed --only start-exits >"$WORK/retry.out" 2>&1
check "--retry-failed --only 只重跑指定的失败任务，旧目录留档" "[ -d $O/start-exits.attempt1 ] && [ \"\$(ls -d $O/*.attempt* | wc -l | tr -d ' ')\" = 1 ]"
check "--only 之后 summary.md 仍列出全部 10 个任务" "[ \$(grep -c '^| ' $O/summary.md) -eq 11 ]"

echo "== 3/7 端口被占：不动别人的服务，任务判失败 =="
$MOCK >/dev/null 2>&1 &
STRAY=$!
for _ in $(seq 1 50); do port_open && break; sleep 0.2; done
"${NQ[@]}" "$WORK/main.ini" --out "$WORK/busy" --lock "$LOCK" --only ok >"$WORK/busy.out" 2>&1
check "端口被占时判 port_busy，原来的服务还活着" "[ \"\$(status $WORK/busy/ok)\" = port_busy ] && kill -0 $STRAY 2>/dev/null"
kill "$STRAY" 2>/dev/null; wait "$STRAY" 2>/dev/null; STRAY=""
for _ in $(seq 1 50); do port_open || break; sleep 0.2; done

echo "== 4/7 被打断 + 同机互斥 =="
cat >"$WORK/long.ini" <<EOF
[DEFAULT]
ready_url = http://127.0.0.1:$PORT/health
ready_interval = 1
settle = 0
stop_grace = 5

[long]
start = $MOCK
steps =
    sleep 120

[after]
steps =
    echo should-not-run > \$NIGHTQ_OUT/x
EOF
"${NQ[@]}" "$WORK/long.ini" --out "$WORK/long" --lock "$LOCK" >"$WORK/long.out" 2>&1 &
RUNNER=$!
for _ in $(seq 1 100); do grep -q '步骤 1/1' "$WORK/long.out" 2>/dev/null && break; sleep 0.2; done
"${NQ[@]}" "$WORK/long.ini" --out "$WORK/long2" --lock "$LOCK" >"$WORK/lock.out" 2>&1
check "已有队列在跑时，第二个队列拒绝启动" "[ $? -ne 0 ] && grep -q '已经有一个夜跑队列在跑' $WORK/lock.out && [ ! -e $WORK/long2/long ]"
kill -TERM "$RUNNER"
wait "$RUNNER"
RC=$?
check "收到 TERM：退出码 130，任务记为被打断，服务已停，后面的任务没跑" \
    "[ $RC -eq 130 ] && [ \"\$(status $WORK/long/long)\" = interrupted ] && ! port_open && [ ! -e $WORK/long/after ]"
"${NQ[@]}" "$WORK/long.ini" --out "$WORK/long" --lock "$LOCK" --only after >"$WORK/long-resume.out" 2>&1
check "打断后续跑：锁已释放，剩下的任务能跑" "[ \"\$(status $WORK/long/after)\" = ok ]"

echo "== 5/7 写错键名：开跑前就报错 =="
printf '[typo]\nstart = true\nstep =\n    echo x\n' >"$WORK/typo.ini"
"${NQ[@]}" "$WORK/typo.ini" --dry-run >"$WORK/typo.out" 2>&1
check "键名写错时退出码 2 并指出是哪个键" "[ $? -eq 2 ] && grep -q '不认识的键「step」' $WORK/typo.out"

echo "== 6/7 仓库里的队列文件都能通过检查 =="
for q in nightq/*.ini; do
    "${NQ[@]}" "$q" --dry-run >"$WORK/dry.out" 2>&1
    check "$q 通过 --dry-run" "[ $? -eq 0 ]"
done

echo "== 7/7 串真的 serve-deepseek.sh + smoke-test.sh（vllm 和 nvidia-smi 用假的顶替）=="
# 只验证外壳：环境变量传得进去、tee 管道起的服务停得掉、日志落在任务目录。不代表真 vLLM 能起来。
# LC_ALL=C 是给 macOS 自带的 bash 3.2 用的：UTF-8 环境下它会把「$变量）」里全角括号的第一个字节
# 并进变量名，serve-*.sh / smoke-test.sh 开着 set -u 就报 unbound variable（2026-11-26 本机实测）。
mkdir -p "$WORK/bin" "$WORK/fake-model"
echo '{}' >"$WORK/fake-model/config.json"
cat >"$WORK/bin/vllm" <<EOF
#!/usr/bin/env bash
while [ \$# -gt 0 ]; do
    case "\$1" in
        --port) P="\$2"; shift ;;
        --served-model-name) N="\$2"; shift ;;
    esac
    shift
done
exec $PY "$PWD/loadtest/mock_server.py" --port "\$P" --served-name "\$N" --speedup 10
EOF
cat >"$WORK/bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
case "$*" in
    *memory.total*) for _ in 1 2 3 4 5 6 7 8; do echo 85000; done ;;
    *) for _ in 1 2 3 4 5 6 7 8; do echo "FAKE RTX 6000D"; done ;;
esac
EOF
chmod +x "$WORK/bin/vllm" "$WORK/bin/nvidia-smi"
cat >"$WORK/real.ini" <<EOF
[DEFAULT]
ready_url = http://127.0.0.1:$PORT/health
ready_timeout = 30
ready_interval = 1
settle = 0
stop_grace = 5

[ds-fake]
start = bash serve-deepseek.sh
env =
    PATH=$WORK/bin:$PATH
    MODEL_PATH=$WORK/fake-model
    PORT=$PORT
    LOG_DIR=\$NIGHTQ_OUT/logs
    DSPARK_TOKENS=5
    LC_ALL=C
snapshot =
    nvidia-smi --query-gpu=name --format=csv,noheader
steps =
    BASE_URL=http://127.0.0.1:$PORT MODEL=deepseek-v4.1-flash bash smoke-test.sh
EOF
"${NQ[@]}" "$WORK/real.ini" --out "$WORK/real" --lock "$LOCK" >"$WORK/real.out" 2>&1
check "真脚本串联：任务成功，冒烟 4 项全过" "[ \"\$(status $WORK/real/ds-fake)\" = ok ] && grep -q '通过 4 / 失败 0' $WORK/real/ds-fake/step-1.log"
check "env 里的 \$NIGHTQ_OUT 被替换，服务日志落在任务目录" "ls $WORK/real/ds-fake/logs/vllm-deepseek-*.log >/dev/null 2>&1"
check "tee 管道起的服务被停干净，端口放开" "! port_open && ! pgrep -f 'mock_server.py --port $PORT' >/dev/null"
check "就绪后和收尾前都留了快照" "[ -s $WORK/real/ds-fake/snapshot-ready.log ] && [ -s $WORK/real/ds-fake/snapshot-end.log ]"

echo
echo "通过 $PASS 项，失败 $FAIL 项。工作目录：$WORK"
[ "$FAIL" -eq 0 ]
