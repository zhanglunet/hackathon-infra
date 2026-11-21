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
n_tokens = 0
text_parts = []
try:
    with urllib.request.urlopen(req, timeout=600) as resp:
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
            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
            if content:
                now = time.time()
                if t_first is None:
                    t_first = now
                t_last = now
                n_tokens += 1
                text_parts.append(content)
except Exception as e:
    print(json.dumps({"tag": "cold", "error": str(e), "elapsed": time.time() - t0},
                     ensure_ascii=False, indent=2))
    sys.exit(1)

if t_first is None:
    print(json.dumps({"tag": "cold", "error": "no tokens received",
                      "elapsed": time.time() - t0}, ensure_ascii=False, indent=2))
    sys.exit(1)

result = {
    "tag": "cold",
    "ttft_s": round(t_first - t0, 2),
    "tpot_ms": round((t_last - t_first) / (n_tokens - 1) * 1000, 1) if n_tokens > 1 else 0.0,
    "tokens": n_tokens,
    "total_s": round(t_last - t0, 2),
    "answer_head": "".join(text_parts)[:200],
}
print(json.dumps(result, ensure_ascii=False, indent=2))
with open("cold_probe_result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print("wrote cold_probe_result.json")
