#!/usr/bin/env bash
# smoke-test.sh — 服务冒烟测试：确认 /v1/chat/completions 和 /metrics 都能用
#
# 用法（先让 serve-qwen27b.sh 跑起来，再另开一个终端跑这个）：
#   bash smoke-test.sh
#   BASE_URL=http://127.0.0.1:8000 MODEL=qwen3.8-27b bash smoke-test.sh
#
# 测什么：
#   1. /health 服务是否活着
#   2. /v1/models 里有没有我们的模型
#   3. 流式发一句短提问，量首 token 耗时（TTFT）和平均每 token 耗时（TPOT）
#   4. /metrics 里有没有 vllm 的 Prometheus 指标
#
# 关于 TTFT/TPOT：
#   比赛 SLO（TTFT≤15s、TPOT≤75ms）是针对长上下文（50K~272K 输入）场景的；
#   这里短提问的 TTFT 正常应在 1 秒以内——它只证明“服务能出 token”，
#   不代表长上下文能过 SLO。长上下文压测等组织方发评测包后用 AIPerf 跑。
#   测 TTFT 用内嵌的 Python（只用标准库 urllib，不依赖 requests）。

set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MODEL="${MODEL:-qwen3.8-27b}"
PASS=0
FAIL=0

ok()   { echo "[通过] $1"; PASS=$((PASS+1)); }
bad()  { echo "[失败] $1"; FAIL=$((FAIL+1)); }

echo "== 冒烟测试：$BASE_URL（模型 $MODEL）=="

echo "--- 1/4 /health ---"
if curl -sf -m 10 "$BASE_URL/health" >/dev/null; then
    ok "服务活着（/health 返回 200）"
else
    bad "连不上 /health：服务还没起好，或端口不对。先看 serve 终端的日志。"
fi

echo "--- 2/4 /v1/models ---"
MODELS_JSON="$(curl -sf -m 10 "$BASE_URL/v1/models" || true)"
if echo "$MODELS_JSON" | grep -q "$MODEL"; then
    ok "模型列表里有 $MODEL"
else
    bad "模型列表里没有 $MODEL。实际返回：${MODELS_JSON:-<空>}"
fi

echo "--- 3/4 流式 chat completions（测 TTFT / TPOT）---"
python3 - "$BASE_URL" "$MODEL" <<'PYEOF'
import json, sys, time, urllib.request

base, model = sys.argv[1], sys.argv[2]
payload = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": "用一句话介绍你自己。"}],
    "max_tokens": 64,
    "temperature": 0,
    "stream": True,
}).encode()
req = urllib.request.Request(
    base + "/v1/chat/completions", data=payload,
    headers={"Content-Type": "application/json"}, method="POST")
t0 = time.time()
t_first = None
t_last = None
n_tokens = 0
text_parts = []
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content")
            if content:
                now = time.time()
                if t_first is None:
                    t_first = now
                t_last = now
                n_tokens += 1
                text_parts.append(content)
except Exception as e:
    print(f"[失败] 请求出错：{e}")
    sys.exit(1)

if t_first is None:
    print("[失败] 没收到任何 token，服务可能没正常生成。")
    sys.exit(1)
ttft = t_first - t0
tpot = ((t_last - t_first) / (n_tokens - 1) * 1000) if n_tokens > 1 else 0.0
print(f"首 token 耗时 TTFT：{ttft:.2f} 秒")
print(f"平均每 token 耗时 TPOT：{tpot:.1f} 毫秒（共 {n_tokens} 个 token）")
print("模型回答：" + "".join(text_parts)[:200])
# 短提问的宽松门槛：TTFT < 15s（比赛 SLO 上限），TPOT < 75ms
ok = True
if ttft >= 15:
    print("[失败] TTFT ≥ 15 秒，短提问都不该这么慢，检查 GPU 是否被占满或服务刚启动还在预热。")
    ok = False
if tpot >= 75:
    print("[失败] TPOT ≥ 75 毫秒，解码太慢，检查是否用了 eager 模式或显存不足导致换页。")
    ok = False
if ok:
    print("[通过] 流式生成正常，TTFT/TPOT 在短提问门槛内")
sys.exit(0 if ok else 1)
PYEOF
if [ $? -eq 0 ]; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi

echo "--- 4/4 /metrics ---"
METRICS="$(curl -sf -m 10 "$BASE_URL/metrics" || true)"
if echo "$METRICS" | grep -q "vllm:num_requests_running"; then
    ok "/metrics 有 vllm 指标（vllm:num_requests_running）"
else
    bad "/metrics 里没有 vllm 指标。可能服务版本不同，或 /metrics 被关了。"
fi

echo
echo "== 结果：通过 $PASS / 失败 $FAIL =="
if [ "$FAIL" -eq 0 ]; then
    echo "冒烟测试全过。可以开始下一步：长上下文验证和 EvalScope 精度对照。"
    exit 0
else
    echo "有 $FAIL 项没过，先看 serve 终端的日志（logs/vllm-serve-*.log）再排查。"
    exit 1
fi
