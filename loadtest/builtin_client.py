"""内置压测客户端（只用标准库）：AIPerf 装不上时的备胎，也可以拿来和 AIPerf 的数互相核对。

语义与 AIPerf 的 --concurrency 一致：始终保持 N 个请求在途，一个结束立刻补下一个。
每个请求都走流式（stream=true），记录：
  TTFT = 发出请求 → 收到第一个 token（content 或 reasoning 都算）
  TPOT = (最后一个 token 时刻 − 第一个 token 时刻) / (输出 token 数 − 1)
输出 token 数优先取服务端 usage.completion_tokens，没有时按收到的 chunk 数估。
"""
import json
import threading
import time
import urllib.error
import urllib.request

from common import now, percentile


def one_request(url, model, rec, extra, timeout):
    body = {
        "model": model,
        "messages": ([{"role": "system", "content": rec["system"]}] if rec.get("system") else [])
        + [{"role": "user", "content": rec["text"]}],
        "max_tokens": rec["output_tokens"],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    body.update(extra)
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = {"input_tokens_est": rec.get("input_tokens"), "error": None}
    t0 = now()
    t_first = t_last = None
    chunks = 0
    usage = None
    finish = None
    got_done = False
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    got_done = True
                    break
                try:
                    ch = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if "error" in ch:
                    raise RuntimeError(f"stream error: {ch['error']}")
                if ch.get("usage"):
                    usage = ch["usage"]
                for c in ch.get("choices") or []:
                    d = c.get("delta") or {}
                    if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                        t = now()
                        if t_first is None:
                            t_first = t
                        t_last = t
                        chunks += 1
                    if c.get("finish_reason"):
                        finish = c["finish_reason"]
        if not got_done:
            raise RuntimeError("stream ended without [DONE]")
        if t_first is None:
            raise RuntimeError("no token received")
    except Exception as e:  # noqa: BLE001  任何失败都记为错误请求
        detail = ""
        if isinstance(e, urllib.error.HTTPError):
            try:
                detail = " " + e.read().decode("utf-8", "replace")[:200]
            except Exception:  # noqa: BLE001
                pass
        r["error"] = f"{type(e).__name__}: {e}{detail}"[:300]
        r["latency_s"] = now() - t0
        return r
    out_n = (usage or {}).get("completion_tokens") or chunks
    r.update({
        "ttft_s": t_first - t0,
        "tpot_ms": ((t_last - t_first) / (out_n - 1) * 1000) if out_n > 1 else None,
        "output_tokens": out_n,
        "input_tokens": (usage or {}).get("prompt_tokens"),
        "latency_s": t_last - t0,
        "finish_reason": finish,
        "end_s": t_last,
    })
    return r


def run(base_url, model, records, concurrency, extra=None, timeout=1800):
    """跑完 records，返回 (逐请求结果, 墙钟秒数)。"""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    extra = extra or {}
    results = [None] * len(records)
    idx = [0]
    lock = threading.Lock()

    def worker():
        while True:
            with lock:
                i = idx[0]
                idx[0] += 1
            if i >= len(records):
                return
            results[i] = one_request(url, model, records[i], extra, timeout)

    t0 = now()
    ths = [threading.Thread(target=worker, daemon=True) for _ in range(min(concurrency, len(records)))]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return results, now() - t0


def summarize(results, wall_s, sc):
    """逐请求结果 → 一级并发的汇总（与 aiperf_engine.summarize 字段一致）。"""
    ok = [r for r in results if r and not r["error"]]
    good = [r for r in ok if r["ttft_s"] <= sc["ttft_s"]
            and (r["tpot_ms"] is None or r["tpot_ms"] <= sc["tpot_ms"])]
    n = len(results)
    errors = [r["error"] for r in results if r and r["error"]]
    return {
        "requests": n,
        "errors": len(errors),
        "error_rate": len(errors) / n if n else 0.0,
        "error_samples": sorted(set(errors))[:3],
        "duration_s": wall_s,
        "p50_ttft_s": percentile([r["ttft_s"] for r in ok], 50),
        "p90_ttft_s": percentile([r["ttft_s"] for r in ok], 90),
        "p99_ttft_s": percentile([r["ttft_s"] for r in ok], 99),
        "p50_tpot_ms": percentile([r["tpot_ms"] for r in ok], 50),
        "p90_tpot_ms": percentile([r["tpot_ms"] for r in ok], 90),
        "p99_tpot_ms": percentile([r["tpot_ms"] for r in ok], 99),
        "request_throughput_rps": len(ok) / wall_s if wall_s else None,
        "output_tps": sum(r["output_tokens"] for r in ok) / wall_s if wall_s else None,
        "goodput_rps": len(good) / wall_s if wall_s else None,
        "good_requests": len(good),
        "input_tokens_mean": (sum(r["input_tokens"] or 0 for r in ok) / len(ok)) if ok else None,
        "output_tokens_mean": (sum(r["output_tokens"] for r in ok) / len(ok)) if ok else None,
        # 模型自己提前结束（输出没到设定长度）的请求数；ignore_eos 生效时应为 0
        "stopped_early": sum(1 for r in ok if r.get("finish_reason") == "stop"),
    }
