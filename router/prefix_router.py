#!/usr/bin/env python3
"""prefix_router.py — 多副本前缀粘滞路由器（纯标准库，零第三方依赖）

背景（回应 PR #9 建议 #2）：
  单机 8 卡起多个独立 vLLM 副本（或跨 4 台机起多副本）对外只暴露 1 个服务地址时，
  普通轮询会把同一会话/相同长前缀的请求打散到不同副本，导致前缀缓存命中率骤降、
  agent（50K/0.70）与 coding（100K/0.90）档 P90 TTFT 暴涨。

原理：
  - 对 POST /v1/chat/completions（及 /v1/completions）解析请求 JSON，取前 PREFIX_CHARS
    字符（默认 4096，覆盖 system prompt + 共享长前缀，不受末尾问题变化影响），
    用 SHA-256 做一致性哈希，将相同前缀稳定路由到同一后端副本。
  - 原样流式透传 SSE（stream: true），不整包缓冲，不影响 TTFT/TPOT。
  - /health：检查所有后端（全部 200 返回 200，否则 503）。
  - /v1/models：转发到首个健康后端。
  - /metrics：聚合所有后端的关键 Prometheus Counter/Gauge（如
    vllm:prefix_cache_queries_total、vllm:prefix_cache_hits_total、
    vllm:num_requests_running），使 loadtest/run_ladder.py 抓单入口 /metrics
    也能准确算出集群级缓存命中率。

用法：
  python3 router/prefix_router.py \
      --backends http://127.0.0.1:8001,http://127.0.0.1:8002 \
      --host 0.0.0.0 --port 8000
"""

import argparse
import hashlib
import http.client
import json
import re
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Tuple

METRIC_LINE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*(?:\{[^}]*\})?)\s+([-+eE0-9.]+)\s*$")
# 这些指标跨副本求和，保证 loadtest/run_ladder.py 与 smoke-test.sh 读数准确
SUM_METRIC_PREFIXES = (
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:gpu_prefix_cache_queries_total",
    "vllm:gpu_prefix_cache_hits_total",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
)


def extract_prefix_key(body: bytes, prefix_chars: int) -> str:
    """从 OpenAI 请求体提取稳定前缀键（不含会变化的尾部提问）。"""
    try:
        data = json.loads(body.decode("utf-8", "replace"))
    except Exception:
        return body[:prefix_chars].decode("utf-8", "replace")

    parts: List[str] = []
    if isinstance(data, dict):
        msgs = data.get("messages")
        if isinstance(msgs, list):
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                content = m.get("content")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and isinstance(item.get("text"), str):
                            parts.append(item["text"])
                if sum(len(p) for p in parts) >= prefix_chars:
                    break
        elif isinstance(data.get("prompt"), str):
            parts.append(data["prompt"])
    joined = "\n".join(parts)
    return joined[:prefix_chars]


def pick_backend(backends: List[Tuple[str, str, int]], key: str) -> Tuple[str, str, int]:
    digest = hashlib.sha256(key.encode("utf-8", "replace")).digest()
    idx = int.from_bytes(digest[:8], "big") % len(backends)
    return backends[idx]


def _make_conn(scheme: str, host: str, port: int, timeout: float) -> http.client.HTTPConnection:
    if scheme == "https":
        return http.client.HTTPSConnection(host, port, timeout=timeout)
    return http.client.HTTPConnection(host, port, timeout=timeout)


def _fetch_backend(
    backend: Tuple[str, str, int], path: str, timeout: float = 10.0
) -> Tuple[int, bytes, str]:
    scheme, host, port = backend
    conn = _make_conn(scheme, host, port, timeout)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        body = resp.read()
        ctype = resp.getheader("Content-Type", "text/plain; charset=utf-8")
        return resp.status, body, ctype
    finally:
        conn.close()


def aggregate_metrics(backends: List[Tuple[str, str, int]]) -> bytes:
    """汇总所有后端的 Prometheus 指标（关键 counter/gauge 求和）。"""
    totals: Dict[str, float] = {}
    passthrough: List[str] = []
    for i, b in enumerate(backends):
        try:
            status, body, _ = _fetch_backend(b, "/metrics", timeout=10.0)
            if status != 200:
                continue
        except Exception:
            continue
        for line in body.decode("utf-8", "replace").splitlines():
            if not line or line.startswith("#"):
                if i == 0 and line:
                    passthrough.append(line)
                continue
            m = METRIC_LINE_RE.match(line.strip())
            if not m:
                continue
            metric_key, val_s = m.group(1), m.group(2)
            base_name = metric_key.split("{", 1)[0]
            if any(base_name == p or base_name.startswith(p) for p in SUM_METRIC_PREFIXES):
                try:
                    totals[metric_key] = totals.get(metric_key, 0.0) + float(val_s)
                except ValueError:
                    pass
            elif i == 0:
                passthrough.append(line)

    out_lines = list(passthrough)
    for k, v in sorted(totals.items()):
        out_lines.append(f"{k} {v:g}")
    return ("\n".join(out_lines) + "\n").encode("utf-8")


def make_handler(backends: List[Tuple[str, str, int]], prefix_chars: int, timeout: float):
    class PrefixRouterHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            # 压测高并发下关闭逐请求 stderr 打印，避免锁竞争拖慢路由
            return

        def _send_bytes(self, status: int, body: bytes, content_type: str = "application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                ok = 0
                for b in backends:
                    try:
                        st, _, _ = _fetch_backend(b, "/health", timeout=5.0)
                        if st == 200:
                            ok += 1
                    except Exception:
                        pass
                if ok == len(backends):
                    self._send_bytes(200, b'{"status":"ok"}\n')
                else:
                    msg = json.dumps({"status": "degraded", "healthy": ok, "total": len(backends)}).encode()
                    self._send_bytes(503, msg)
                return

            if self.path == "/metrics":
                self._send_bytes(200, aggregate_metrics(backends), "text/plain; version=0.0.4; charset=utf-8")
                return

            # /v1/models 或其他 GET 请求转发给首个后端
            try:
                st, body, ctype = _fetch_backend(backends[0], self.path, timeout=15.0)
                self._send_bytes(st, body, ctype)
            except Exception as e:
                self._send_bytes(502, json.dumps({"error": str(e)}).encode())

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length > 0 else b""
            key = extract_prefix_key(body, prefix_chars)
            scheme, host, port = pick_backend(backends, key)

            fwd_headers = {
                "Content-Type": self.headers.get("Content-Type", "application/json"),
                "Content-Length": str(len(body)),
            }
            auth = self.headers.get("Authorization")
            if auth:
                fwd_headers["Authorization"] = auth

            conn = _make_conn(scheme, host, port, timeout)
            try:
                conn.request("POST", self.path, body=body, headers=fwd_headers)
                resp = conn.getresponse()
                ctype = resp.getheader("Content-Type", "application/json")
                is_sse = "text/event-stream" in ctype.lower()

                if is_sse:
                    self.send_response(resp.status)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    # 逐行/逐块透传，保证 TTFT/TPOT 零缓冲延迟
                    while True:
                        line = resp.readline()
                        if not line:
                            break
                        self.wfile.write(line)
                        self.wfile.flush()
                else:
                    resp_body = resp.read()
                    self._send_bytes(resp.status, resp_body, ctype)
            except Exception as e:
                self._send_bytes(502, json.dumps({"error": f"upstream {host}:{port} failed: {e}"}).encode())
            finally:
                conn.close()

    return PrefixRouterHandler


def parse_backends(raw: str) -> List[Tuple[str, str, int]]:
    out = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        u = urllib.parse.urlparse(item)
        scheme = u.scheme or "http"
        host = u.hostname or "127.0.0.1"
        port = u.port or (443 if scheme == "https" else 80)
        out.append((scheme, host, port))
    if not out:
        raise ValueError("至少需要提供一个后端 URL（--backends）")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="多副本前缀粘滞路由器（保前缀缓存命中）")
    parser.add_argument("--backends", required=True, help="逗号分隔的后端列表，如 http://127.0.0.1:8001,http://127.0.0.1:8002")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prefix-chars", type=int, default=512, help="用于一致性哈希的前缀字符数（默认 512，避开尾部变化内容）")
    parser.add_argument("--timeout", type=float, default=600.0, help="上游请求超时秒数（默认 600）")
    args = parser.parse_args()

    backends = parse_backends(args.backends)
    handler = make_handler(backends, args.prefix_chars, args.timeout)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"[prefix_router] 监听 http://{args.host}:{args.port} -> {len(backends)} 个后端：{args.backends}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
