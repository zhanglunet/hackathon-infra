#!/usr/bin/env python3
"""假推理服务：不推理，只按一个粗略的性能模型「吐字」，用来在没有 GPU 时调通压测链路。

只用标准库（Python 3.9+ 可跑），对外接口模仿 vLLM 的 OpenAI 兼容服务：
  GET  /health                 200
  GET  /v1/models              返回 served 名
  GET  /metrics                Prometheus 文本，含 vllm:prefix_cache_queries_total /
                               vllm:prefix_cache_hits_total / num_requests_running/waiting
  POST /v1/chat/completions    支持 stream（SSE）和非 stream，支持 max_tokens /
                               max_completion_tokens / ignore_eos / stream_options.include_usage

性能模型（数字都是假的，只为让阶梯在高并发时能「失败」，好验证判定逻辑）：
  - token 数 = 字符数 / --chars-per-token
  - 前缀缓存：按 --block-tokens 切块、链式哈希，LRU 容量 --cache-tokens；
    命中的块不用 prefill
  - prefill 串行（一把锁），耗时 = 未命中 token / --prefill-tps
  - 同时运行的请求数上限 --max-num-seqs，超出的排队（计入 TTFT）
  - decode 每 token 耗时 = --tpot-ms × (1 + --decode-slope × (正在 decode 的请求数 − 1))

用法：
  python3 loadtest/mock_server.py --port 8000
  python3 loadtest/mock_server.py --port 8000 --speedup 10   # 所有耗时 ÷10，调试更快
"""
import argparse
import asyncio
import hashlib
import json
import time
import uuid
from collections import OrderedDict

ARGS = None


class State:
    def __init__(self, a):
        self.prefill_lock = asyncio.Lock()
        self.seq_sem = asyncio.Semaphore(a.max_num_seqs)
        self.cache = OrderedDict()  # block_hash -> None（LRU）
        self.cache_cap_blocks = max(1, a.cache_tokens // a.block_tokens)
        self.running = 0
        self.waiting = 0
        self.decoding = 0
        self.q_tokens = 0  # prefix cache 查询 token 数（累计）
        self.h_tokens = 0  # prefix cache 命中 token 数（累计）
        self.prompt_tokens = 0
        self.gen_tokens = 0
        self.requests_ok = 0


S = None


def n_tokens(text):
    return max(1, int(len(text) / ARGS.chars_per_token))


def lookup_and_insert(text):
    """返回 (总 token, 命中 token)。按块链式哈希，模拟 vLLM 的 prefix caching。"""
    block_chars = int(ARGS.block_tokens * ARGS.chars_per_token)
    total = n_tokens(text)
    hashes = []
    h = b""
    for i in range(0, len(text) - block_chars + 1, block_chars):
        h = hashlib.sha1(h + text[i:i + block_chars].encode()).digest()
        hashes.append(h)
    hit_blocks = 0
    for hb in hashes:
        if hb in S.cache:
            S.cache.move_to_end(hb)
            hit_blocks += 1
        else:
            break
    for hb in hashes[hit_blocks:]:
        S.cache[hb] = None
        if len(S.cache) > S.cache_cap_blocks:
            S.cache.popitem(last=False)
    hit = min(total, hit_blocks * ARGS.block_tokens)
    S.q_tokens += total
    S.h_tokens += hit
    return total, hit


def prompt_text(body):
    parts = []
    for m in body.get("messages", []):
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            parts.extend(x.get("text", "") for x in c if isinstance(x, dict))
    return "\n".join(parts)


async def sleep_s(sec):
    if sec > 0:
        await asyncio.sleep(sec / ARGS.speedup)


async def generate(body, emit):
    """跑一个请求：排队 → prefill → decode。emit(kind, payload) 负责输出。"""
    text = prompt_text(body)
    max_tokens = body.get("max_completion_tokens") or body.get("max_tokens") or 256
    ignore_eos = bool(body.get("ignore_eos"))
    # 不强制长度时，模拟模型自己提前结束
    out_n = max_tokens if ignore_eos else min(max_tokens, ARGS.natural_len)
    S.waiting += 1
    async with S.seq_sem:
        S.waiting -= 1
        S.running += 1
        try:
            async with S.prefill_lock:
                total, hit = lookup_and_insert(text)
                await sleep_s((total - hit) / ARGS.prefill_tps)
            S.prompt_tokens += total
            S.decoding += 1
            try:
                for i in range(out_n):
                    if i > 0:
                        step = ARGS.tpot_ms * (1 + ARGS.decode_slope * (S.decoding - 1))
                        await sleep_s(step / 1000)
                    await emit("token", i)
                    S.gen_tokens += 1
            finally:
                S.decoding -= 1
            finish = "length" if out_n >= max_tokens else "stop"
            S.requests_ok += 1
            return total, out_n, finish
        finally:
            S.running -= 1


async def read_request(reader):
    line = await reader.readline()
    if not line:
        return None
    method, path, _ = line.decode().split(" ", 2)
    headers = {}
    while True:
        h = await reader.readline()
        if h in (b"\r\n", b"\n", b""):
            break
        k, v = h.decode().split(":", 1)
        headers[k.strip().lower()] = v.strip()
    body = b""
    if "content-length" in headers:
        body = await reader.readexactly(int(headers["content-length"]))
    return method, path.split("?")[0], headers, body


def respond(writer, status, ctype, payload):
    data = payload if isinstance(payload, bytes) else payload.encode()
    writer.write(
        f"HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {len(data)}\r\n"
        f"Connection: close\r\n\r\n".encode() + data)


def metrics_text():
    m = ARGS.served_name
    lines = [
        "# TYPE vllm:prefix_cache_queries counter",
        f'vllm:prefix_cache_queries_total{{model_name="{m}"}} {S.q_tokens}',
        "# TYPE vllm:prefix_cache_hits counter",
        f'vllm:prefix_cache_hits_total{{model_name="{m}"}} {S.h_tokens}',
        "# TYPE vllm:num_requests_running gauge",
        f'vllm:num_requests_running{{model_name="{m}"}} {S.running}',
        "# TYPE vllm:num_requests_waiting gauge",
        f'vllm:num_requests_waiting{{model_name="{m}"}} {S.waiting}',
        "# TYPE vllm:prompt_tokens counter",
        f'vllm:prompt_tokens_total{{model_name="{m}"}} {S.prompt_tokens}',
        "# TYPE vllm:generation_tokens counter",
        f'vllm:generation_tokens_total{{model_name="{m}"}} {S.gen_tokens}',
    ]
    return "\n".join(lines) + "\n"


async def handle(reader, writer):
    try:
        req = await read_request(reader)
        if req is None:
            return
        method, path, _, raw = req
        if method == "GET" and path == "/health":
            respond(writer, "200 OK", "text/plain", "")
        elif method == "GET" and path == "/v1/models":
            respond(writer, "200 OK", "application/json", json.dumps(
                {"object": "list", "data": [{"id": ARGS.served_name, "object": "model",
                                             "max_model_len": ARGS.max_model_len}]}))
        elif method == "GET" and path == "/metrics":
            respond(writer, "200 OK", "text/plain; version=0.0.4", metrics_text())
        elif method == "POST" and path == "/v1/chat/completions":
            await chat(writer, json.loads(raw or b"{}"))
        else:
            respond(writer, "404 Not Found", "application/json", '{"error":"not found"}')
        await writer.drain()
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def chat(writer, body):
    rid = "chatcmpl-" + uuid.uuid4().hex[:12]
    created = int(time.time())
    model = body.get("model", ARGS.served_name)
    if n_tokens(prompt_text(body)) > ARGS.max_model_len:
        respond(writer, "400 Bad Request", "application/json",
                json.dumps({"error": {"message": "prompt longer than max_model_len"}}))
        return
    if not body.get("stream"):
        total, out_n, finish = await generate(body, lambda *_: asyncio.sleep(0))
        respond(writer, "200 OK", "application/json", json.dumps({
            "id": rid, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": "tok " * out_n}}],
            "usage": {"prompt_tokens": total, "completion_tokens": out_n,
                      "total_tokens": total + out_n}}))
        return

    writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                 b"Cache-Control: no-cache\r\nConnection: close\r\n\r\n")

    def chunk(delta, finish=None, usage=None):
        d = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
             "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if usage is not None:
            d["choices"] = []
            d["usage"] = usage
        return b"data: " + json.dumps(d).encode() + b"\n\n"

    async def emit(kind, i):
        delta = {"content": "tok "}
        if i == 0:
            delta["role"] = "assistant"
        writer.write(chunk(delta))
        await writer.drain()

    total, out_n, finish = await generate(body, emit)
    writer.write(chunk({}, finish))
    if (body.get("stream_options") or {}).get("include_usage"):
        writer.write(chunk({}, usage={"prompt_tokens": total, "completion_tokens": out_n,
                                      "total_tokens": total + out_n}))
    writer.write(b"data: [DONE]\n\n")


def main():
    global ARGS, S
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--served-name", default="mock-model")
    p.add_argument("--max-model-len", type=int, default=272000)
    p.add_argument("--chars-per-token", type=float, default=4.0)
    p.add_argument("--block-tokens", type=int, default=64)
    p.add_argument("--cache-tokens", type=int, default=2_000_000, help="前缀缓存容量（token）")
    p.add_argument("--prefill-tps", type=float, default=20000, help="prefill 速度（token/秒）")
    p.add_argument("--tpot-ms", type=float, default=20, help="单请求 decode 每 token 毫秒")
    p.add_argument("--decode-slope", type=float, default=0.02, help="每多一个并发 decode，TPOT 增加的比例")
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--natural-len", type=int, default=200, help="不带 ignore_eos 时的自然输出长度")
    p.add_argument("--speedup", type=float, default=1.0, help="所有耗时除以这个数（只为加快调试）")
    ARGS = p.parse_args()

    async def run():
        global S
        S = State(ARGS)
        srv = await asyncio.start_server(handle, ARGS.host, ARGS.port, limit=2 ** 26, backlog=1024)
        print(f"mock server on http://{ARGS.host}:{ARGS.port}  (speedup ×{ARGS.speedup})", flush=True)
        async with srv:
            await srv.serve_forever()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
