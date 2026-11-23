#!/usr/bin/env python3
"""单次冷请求：对已部署的服务发一次 271500-token 提示，只测冷 TTFT/TPOT。
配合 profiling 用：外部先 POST /start_profile，跑完本脚本再 POST
/stop_profile，这样 trace 里只有这一次冷 prefill。用法：
python3 cold_only.py https://<部署地址> [prompt 文件，默认 prompt_272k.txt]
只用标准库。结果打印到 stdout，并写 cold_probe_result.json。
"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1].rstrip("/")
PROMPT_FILE = sys.argv[2] if len(sys.argv) > 2 else "prompt_272k.txt"
MODEL = "qwen3.8-27b"
MAX_TOKENS = 64

with open(PROMPT_FILE, encoding="utf-8") as f:
    PROMPT = f.read()

payload = json.dumps({
    "model": MODEL,
    "messages": [{"role": "user", "content": PROMPT}],
    "max_tokens": MAX_TOKENS,
    "temperature": 0,
    "stream": True,
}).encode()
req = urllib.request.Request(
    BASE + "/v1/chat/completions", data=payload,
    headers={"Content-Type": "application/json"}, method="POST")
t0 = time.time()
t_first = t_last = None
n_content = 0
n_reasoning = 0
text_parts = []
reasoning_parts = []
finish_reason = None
saw_done = False
stream_error = None
try:
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                saw_done = True
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            # 服务端错误以 chunk 形式返回时不算正常结束。
            if isinstance(chunk, dict) and chunk.get("error"):
                stream_error = str(chunk["error"])
                break
            choice = (chunk.get("choices") or [{}])[0]
            fr = choice.get("finish_reason")
            if fr:
                finish_reason = fr
            delta = choice.get("delta") or {}
            # reasoning 与 content 都算输出 token：开 reasoning-parser 后
            # 思考先以独立 reasoning 字段流出，只数 content 会漏掉整段
            # 思考、TTFT 也会被高估（#5 评审）。
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            content = delta.get("content")
            if reasoning:
                now = time.time()
                if t_first is None:
                    t_first = now
                t_last = now
                n_reasoning += 1
                reasoning_parts.append(reasoning)
            if content:
                now = time.time()
                if t_first is None:
                    t_first = now
                t_last = now
                n_content += 1
                text_parts.append(content)
except Exception as e:
    print(json.dumps({"tag": "cold", "error": str(e), "elapsed": round(time.time() - t0, 2)},
                     ensure_ascii=False, indent=2))
    sys.exit(1)

if stream_error:
    print(json.dumps({"tag": "cold", "error": f"stream error: {stream_error}",
                      "elapsed": round(time.time() - t0, 2)}, ensure_ascii=False, indent=2))
    sys.exit(1)

if not saw_done:
    # 流在 [DONE] 之前断了（连接丢了）：结果不可信，算失败。
    print(json.dumps({"tag": "cold", "error": "stream ended without [DONE]",
                      "elapsed": round(time.time() - t0, 2)}, ensure_ascii=False, indent=2))
    sys.exit(1)

n_tokens = n_content + n_reasoning
if t_first is None:
    print(json.dumps({"tag": "cold", "error": "no tokens received",
                      "elapsed": round(time.time() - t0, 2)}, ensure_ascii=False, indent=2))
    sys.exit(1)

result = {
    "tag": "cold",
    "ttft_s": round(t_first - t0, 2),
    "tpot_ms": round((t_last - t_first) / (n_tokens - 1) * 1000, 1) if n_tokens > 1 else 0.0,
    "tokens": n_tokens,
    "tokens_content": n_content,
    "tokens_reasoning": n_reasoning,
    "total_s": round(t_last - t0, 2),
    "finish_reason": finish_reason,
    # 没开 reasoning-parser 时 reasoning 为空，tokens 就是原来的 content 计数。
    "answer_head": "".join(text_parts)[:200] or "".join(reasoning_parts)[:200],
}
if finish_reason == "length":
    # 在 max_tokens 处截断：延迟数字仍有效，但答案是不完整的，标出来。
    result["truncated"] = True
print(json.dumps(result, ensure_ascii=False, indent=2))
with open("cold_probe_result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print("wrote cold_probe_result.json")
