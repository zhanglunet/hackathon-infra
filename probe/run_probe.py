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
        return {"tag": tag, "error": str(e), "elapsed": round(time.time() - t0, 2)}
    if stream_error:
        return {"tag": tag, "error": f"stream error: {stream_error}",
                "elapsed": round(time.time() - t0, 2)}
    if not saw_done:
        # 流在 [DONE] 之前断了（连接丢了）：结果不可信，算失败。
        return {"tag": tag, "error": "stream ended without [DONE]",
                "elapsed": round(time.time() - t0, 2)}
    n_tokens = n_content + n_reasoning
    if t_first is None:
        return {"tag": tag, "error": "no tokens received", "elapsed": round(time.time() - t0, 2)}
    ttft = t_first - t0
    tpot = ((t_last - t_first) / (n_tokens - 1) * 1000) if n_tokens > 1 else 0.0
    result = {
        "tag": tag,
        "ttft_s": round(ttft, 2),
        "tpot_ms": round(tpot, 1),
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
    return result


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
if any("error" in r for r in results):
    # 以前失败也返回 0，下游以为测完了；现在失败即非零（#5 评审）。
    print("[错误] 有请求失败，见上（probe_result.json 已写）")
    sys.exit(1)
