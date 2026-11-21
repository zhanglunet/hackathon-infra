#!/usr/bin/env python3
"""272K 长上下文探测：对已部署的服务发一次 271500-token 提示，测
TTFT / TPOT，并读 /metrics 的 KV 缓存占用。同一个提示连发两次：
第 1 次冷请求（完整 prefill），第 2 次热请求（prefix caching 命中）。
用法：python3 run_probe.py https://<部署地址>
只用标准库。结果打印到 stdout，并写 probe_result.json。
"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1].rstrip("/")
MODEL = "qwen3.8-27b"
MAX_TOKENS = 64

with open("prompt_272k.txt", encoding="utf-8") as f:
    PROMPT = f.read()
with open("probe_meta.json", encoding="utf-8") as f:
    META = json.load(f)


def get_metrics():
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=15) as r:
            text = r.read().decode()
        kv = None
        running = None
        for line in text.splitlines():
            if line.startswith("vllm:kv_cache_usage_perc"):
                kv = line.rsplit(" ", 1)[-1]
            if line.startswith("vllm:num_requests_running"):
                running = line.rsplit(" ", 1)[-1]
        return {"kv_cache_usage_perc": kv, "num_requests_running": running}
    except Exception as e:
        return {"error": str(e)}


def run_once(tag):
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
        return {"tag": tag, "error": str(e), "elapsed": time.time() - t0}
    if t_first is None:
        return {"tag": tag, "error": "no tokens received", "elapsed": time.time() - t0}
    ttft = t_first - t0
    tpot = ((t_last - t_first) / (n_tokens - 1) * 1000) if n_tokens > 1 else 0.0
    return {
        "tag": tag,
        "ttft_s": round(ttft, 2),
        "tpot_ms": round(tpot, 1),
        "tokens": n_tokens,
        "total_s": round(t_last - t0, 2),
        "answer_head": "".join(text_parts)[:200],
    }


print(f"prompt tokens (templated): {META['prompt_tokens_templated']}")
print("metrics before:", get_metrics())
results = []
for tag in ("cold", "hot"):
    print(f"== {tag} 请求开始 ==")
    r = run_once(tag)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    results.append(r)
    print("metrics after:", get_metrics())
    time.sleep(5)

out = {
    "model": MODEL,
    "prompt_tokens": META["prompt_tokens_templated"],
    "max_tokens": MAX_TOKENS,
    "runs": results,
}
with open("probe_result.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("wrote probe_result.json")
