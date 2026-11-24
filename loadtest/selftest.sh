#!/usr/bin/env bash
# selftest.sh — 不用 GPU、$0，1～2 分钟验证整条压测链路能跑通。
#
# 做什么：起一个假推理服务 → 生成缩小版数据 → 每档只跑并发 8 → 生成报告 → 关掉假服务。
# 假服务的数字没有意义，只看「有没有报错、四档是否都有结果、实测命中率是否接近 0.70/0.90」。
#
# 用法：
#   bash loadtest/selftest.sh                 # 内置引擎，只需要 python3
#   ENGINE=aiperf bash loadtest/selftest.sh   # 用 AIPerf（需已 pip install aiperf）
#   AIPERF=/path/to/aiperf ENGINE=aiperf bash loadtest/selftest.sh

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python3}"
ENGINE="${ENGINE:-builtin}"
AIPERF="${AIPERF:-aiperf}"
PORT="${PORT:-18765}"
WORK="${WORK:-$(mktemp -d -t loadtest-selftest-XXXX)}"

echo "== 1/4 起假服务（端口 $PORT）=="
"$PY" loadtest/mock_server.py --port "$PORT" --speedup 10 >"$WORK/mock.log" 2>&1 &
MOCK_PID=$!
trap 'kill $MOCK_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 50); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
    sleep 0.2
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo "[错误] 假服务没起来，见 $WORK/mock.log"; exit 1; }

echo "== 2/4 生成缩小版数据（scale 0.02，只要并发 8）=="
"$PY" loadtest/make_datasets.py --out "$WORK/data" --scale 0.02 --concurrency 8

echo "== 3/4 跑阶梯（引擎 $ENGINE，每档只跑并发 8）=="
"$PY" loadtest/run_ladder.py --url "http://127.0.0.1:$PORT" --data "$WORK/data" --out "$WORK/results" \
    --engine "$ENGINE" --aiperf "$AIPERF" --tokenizer builtin --concurrency 8 --label "selftest（假服务）"

echo "== 4/4 检查报告 =="
REPORT="$WORK/results/report.md"
N=$(grep -c "^| 8 |" "$REPORT" || true)
if [ "$N" -ne 4 ]; then
    echo "[失败] 报告里应有 4 档并发 8 的结果，实际 $N 档。见 $REPORT"
    exit 1
fi
echo "[通过] 四档都有结果。报告：$REPORT"
sed -n '/## 总览/,/^## 智能体/p' "$REPORT" | head -n 9
