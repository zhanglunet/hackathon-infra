#!/usr/bin/env python3
"""prefix_router.py — 多副本前缀粘滞路由器（纯标准库，零第三方依赖）

背景（回应 PR #9 建议 #2，2026-11-25 按 issue #14 重写）：
  单机 8 卡起多个独立 vLLM 副本（或跨 4 台机起多副本）对外只暴露 1 个服务地址时，
  普通轮询会把同一会话/相同长前缀的请求打散到不同副本，导致前缀缓存命中率骤降、
  agent（50K/0.70）与 coding（100K/0.90）档 P90 TTFT 暴涨。

路由（issue #14 的两个 P0 之后）：
  - 路由键：去掉开头的 system 消息（各会话共用的长 system prompt 不再进入键），
    取剩余对话内容的前 PREFIX_CHARS 字符（默认 512）。同一会话的键是对话的真前缀，
    随轮次增长不变，所以同一会话稳定落在同一副本（保缓存命中）；不同会话的对话
    内容不同，键不同，不会撞到一起。
  - rendezvous hashing（HRW）：对每个后端算 score = sha256(key | backend)，
    取分最高的。不再是哈希取模：增删一个后端时，只有原来落在它的那部分会话
    需要迁移，其余会话不动（旧的「一致性哈希」注释说错了，实际是取模，已改）。
  - 有界负载（bounded loads）：分最高的后端在途请求数超过 平均值 × (1 + ε)
    （ε 默认 0.25）时，顺着分数往下找下一个后端。共同前缀导致热点时自动溢出，
    不会让 1 个副本干 100% 的活。
  - 后台探活：每 --health-interval 秒（默认 5 秒）并行探各后端的 /health，
    不健康的后端摘出路由环，恢复后加回；/health 端点直接返回缓存的探活结果，
    1 秒内返回。转发时连接失败（后端刚挂、探活还没发现），在还没给客户端发
    任何字节的前提下立刻重试环上下一个后端；一旦开始向客户端发字节就不再重试。
  - /v1/models：转发给第一个健康的后端；/metrics：并行抓取各后端（单个超时 3 秒），
    HELP/TYPE 行取第一个健康后端，关键 counter/gauge 跨副本求和。
  - /router/stats：各后端在途请求数、累计请求数、健康状态、总路由数与故障转移数。
  - 上游超时默认 21600 秒（与 AIPerf 默认一致）；request_queue_size=1024，
    daemon_threads=True，80 并发不再被拒。

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
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Tuple

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

Backend = Tuple[str, str, int]


def _backend_id(backend: Backend) -> str:
    scheme, host, port = backend
    return f"{scheme}://{host}:{port}"


def _rendezvous_score(key: str, backend: Backend) -> int:
    material = f"{key}|{_backend_id(backend)}".encode("utf-8", "replace")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def extract_prefix_key(body: bytes, prefix_chars: int) -> str:
    """从 OpenAI 请求体提取稳定前缀键（不含会变化的尾部提问）。

    开头的 system 消息跳过：各会话共用的长 system prompt 不进入键，
    否则所有请求的键完全相同（issue #14 P0-1）。剩下的对话内容是真前缀，
    同一会话随轮次增长键不变，不同会话键不同。
    """
    try:
        data = json.loads(body.decode("utf-8", "replace"))
    except Exception:
        return body[:prefix_chars].decode("utf-8", "replace")

    parts: List[str] = []
    if isinstance(data, dict):
        msgs = data.get("messages")
        if isinstance(msgs, list):
            started = False
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                # 跳过开头的 system 消息（共用长 system prompt）
                if not started and m.get("role") == "system":
                    continue
                started = True
                content = m.get("content")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and isinstance(item.get("text"), str):
                            parts.append(item["text"])
                if sum(len(p) for p in parts) >= prefix_chars:
                    break
            if not parts:
                # 退化情况：请求里只有 system 消息，用全部内容当键
                for m in msgs:
                    if not isinstance(m, dict):
                        continue
                    content = m.get("content")
                    if isinstance(content, str):
                        parts.append(content)
        elif isinstance(data.get("prompt"), str):
            parts.append(data["prompt"])
    joined = "\n".join(parts)
    return joined[:prefix_chars]


def pick_backend(backends: List[Backend], key: str) -> Backend:
    """无状态的 rendezvous 选法（不看负载与健康，仅保兼容）。

    线上路径走 RouterState.rank_backends（有界负载 + 只选健康后端），
    这个函数只保留给直接调用的老用法。
    """
    return max(backends, key=lambda b: _rendezvous_score(key, b))


def _make_conn(scheme: str, host: str, port: int, timeout: float) -> http.client.HTTPConnection:
    if scheme == "https":
        return http.client.HTTPSConnection(host, port, timeout=timeout)
    return http.client.HTTPConnection(host, port, timeout=timeout)


def _fetch_backend(
    backend: Backend, path: str, timeout: float = 10.0
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


class RouterState:
    """路由器的共享状态：健康缓存、在途计数、统计。"""

    def __init__(self, backends: List[Backend], health_interval: float = 5.0, epsilon: float = 0.25):
        self.backends = list(backends)
        self.health_interval = health_interval
        self.epsilon = epsilon
        self.started_at = time.time()
        self.lock = threading.Lock()
        self.health: Dict[Backend, bool] = {b: True for b in self.backends}
        self.last_probe: Dict[Backend, float] = {b: 0.0 for b in self.backends}
        self.in_flight: Dict[Backend, int] = {b: 0 for b in self.backends}
        self.total_requests: Dict[Backend, int] = {b: 0 for b in self.backends}
        self.total_routed = 0
        self.total_failover = 0

    def probe_once(self) -> None:
        """并行探一次各后端 /health（单个超时 2 秒），更新健康缓存。"""
        with ThreadPoolExecutor(max_workers=max(1, len(self.backends))) as ex:
            futs = {ex.submit(_fetch_backend, b, "/health", 2.0): b for b in self.backends}
            for fut in as_completed(futs):
                backend = futs[fut]
                try:
                    status, _, _ = fut.result()
                    ok = status == 200
                except Exception:
                    ok = False
                with self.lock:
                    self.health[backend] = ok
                    self.last_probe[backend] = time.time()

    def probe_loop(self) -> None:
        while True:
            try:
                self.probe_once()
            except Exception:
                pass
            time.sleep(self.health_interval)

    def start_probing(self) -> None:
        t = threading.Thread(target=self.probe_loop, daemon=True, name="health-prober")
        t.start()

    def rank_backends(self, key: str) -> List[Backend]:
        """按 rendezvous 分数排序的候选后端（有界负载 + 只用健康后端）。

        分最高的后端在途数超过 平均值 × (1 + ε) 时，把它往后排，让请求溢出
        到下一个后端；如果没有健康后端，用全部后端（尽力而为）。
        """
        with self.lock:
            healthy = [b for b in self.backends if self.health.get(b, True)]
            inflight = dict(self.in_flight)
        pool = healthy if healthy else list(self.backends)
        scored = sorted(pool, key=lambda b: _rendezvous_score(key, b), reverse=True)
        if len(scored) > 1:
            avg = sum(inflight.get(b, 0) for b in scored) / len(scored)
            cap = max(avg, 1.0) * (1.0 + self.epsilon)
            under = [b for b in scored if inflight.get(b, 0) <= cap]
            if under:
                first = under[0]
                return [first] + [b for b in scored if b != first]
        return scored

    def inflight_inc(self, backend: Backend) -> None:
        with self.lock:
            self.in_flight[backend] = self.in_flight.get(backend, 0) + 1
            self.total_requests[backend] = self.total_requests.get(backend, 0) + 1
            self.total_routed += 1

    def inflight_dec(self, backend: Backend) -> None:
        with self.lock:
            self.in_flight[backend] = max(0, self.in_flight.get(backend, 0) - 1)

    def note_failover(self) -> None:
        with self.lock:
            self.total_failover += 1

    def health_snapshot(self) -> List[Dict]:
        with self.lock:
            return [
                {
                    "backend": _backend_id(b),
                    "healthy": self.health.get(b, True),
                    "in_flight": self.in_flight.get(b, 0),
                    "total_requests": self.total_requests.get(b, 0),
                    "last_probe_age_s": round(time.time() - self.last_probe.get(b, 0.0), 1),
                }
                for b in self.backends
            ]

    def stats(self) -> Dict:
        return {
            "uptime_s": round(time.time() - self.started_at, 1),
            "total_routed": self.total_routed,
            "total_failover": self.total_failover,
            "backends": self.health_snapshot(),
        }


def aggregate_metrics(state: RouterState) -> bytes:
    """并行汇总所有后端的 Prometheus 指标（关键 counter/gauge 求和）。

    HELP/TYPE 行取第一个健康后端（旧版只取 backend 0，它挂了这些行就没了）。
    单个后端超时 3 秒，超时或失败的后端直接跳过，不拖慢整个 /metrics。
    """
    with state.lock:
        backends = list(state.backends)
        healthy = [b for b in backends if state.health.get(b, True)]
    ordered = (healthy or backends) + [b for b in backends if b not in (healthy or backends)]
    fetched: Dict[Backend, Tuple[int, bytes]] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(backends))) as ex:
        futs = {ex.submit(_fetch_backend, b, "/metrics", 3.0): b for b in backends}
        for fut in as_completed(futs):
            backend = futs[fut]
            try:
                status, body, _ = fut.result()
                if status == 200:
                    fetched[backend] = (status, body)
            except Exception:
                pass

    totals: Dict[str, float] = {}
    passthrough: List[str] = []
    first_ok = next((b for b in ordered if b in fetched), None)
    for backend in ordered:
        if backend not in fetched:
            continue
        _, body = fetched[backend]
        for line in body.decode("utf-8", "replace").splitlines():
            if not line or line.startswith("#"):
                if backend == first_ok and line:
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
            elif backend == first_ok:
                passthrough.append(line)

    out_lines = list(passthrough)
    for k, v in sorted(totals.items()):
        out_lines.append(f"{k} {v:g}")
    return ("\n".join(out_lines) + "\n").encode("utf-8")


def make_handler(state: RouterState, prefix_chars: int, timeout: float):
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
                # 直接返回缓存的探活结果，不现探（1 秒内返回）
                snap = state.health_snapshot()
                healthy = sum(1 for s in snap if s["healthy"])
                if healthy == len(snap):
                    self._send_bytes(200, b'{"status":"ok"}\n')
                else:
                    msg = json.dumps(
                        {"status": "degraded", "healthy": healthy, "total": len(snap), "backends": snap}
                    ).encode()
                    self._send_bytes(503, msg)
                return

            if self.path == "/metrics":
                self._send_bytes(200, aggregate_metrics(state), "text/plain; version=0.0.4; charset=utf-8")
                return

            if self.path == "/router/stats":
                self._send_bytes(200, json.dumps(state.stats()).encode())
                return

            # /v1/models 或其他 GET 请求转发给第一个健康后端
            with state.lock:
                healthy = [b for b in state.backends if state.health.get(b, True)]
            target = (healthy or state.backends)[0]
            try:
                st, body, ctype = _fetch_backend(target, self.path, timeout=15.0)
                self._send_bytes(st, body, ctype)
            except Exception as e:
                self._send_bytes(502, json.dumps({"error": str(e)}).encode())

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length > 0 else b""
            key = extract_prefix_key(body, prefix_chars)
            candidates = state.rank_backends(key)

            fwd_headers = {
                "Content-Type": self.headers.get("Content-Type", "application/json"),
                "Content-Length": str(len(body)),
            }
            auth = self.headers.get("Authorization")
            if auth:
                fwd_headers["Authorization"] = auth

            last_err: Optional[Exception] = None
            for backend in candidates:
                scheme, host, port = backend
                state.inflight_inc(backend)
                conn = _make_conn(scheme, host, port, timeout)
                try:
                    conn.request("POST", self.path, body=body, headers=fwd_headers)
                    resp = conn.getresponse()
                except Exception as e:
                    # 还没拿到上游响应（连接失败/后端刚挂）：重试下一个后端
                    conn.close()
                    state.inflight_dec(backend)
                    state.note_failover()
                    last_err = e
                    continue

                # 已拿到上游响应：之后不再重试（尤其 SSE 已经开始向客户端发字节，
                # 再发一个 502 会往已开始的响应里写乱数据，见 issue #14 P1-5）
                try:
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
                except (BrokenPipeError, ConnectionResetError):
                    pass  # 客户端先断开了
                except Exception:
                    pass  # 上游中途断流：直接关掉，不再补发 502
                finally:
                    state.inflight_dec(backend)
                    conn.close()
                return

            # 所有候选后端在拿到响应前都失败了
            self._send_bytes(
                502,
                json.dumps({"error": f"all backends failed: {last_err}"}).encode(),
            )

    return PrefixRouterHandler


class RouterHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 1024


def parse_backends(raw: str) -> List[Backend]:
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
    parser.add_argument("--prefix-chars", type=int, default=512, help="用于路由键的对话字符数（默认 512，开头的 system 消息不计入）")
    parser.add_argument("--timeout", type=float, default=21600.0, help="上游请求超时秒数（默认 21600，与 AIPerf 默认一致）")
    parser.add_argument("--health-interval", type=float, default=5.0, help="后台探活间隔秒数（默认 5）")
    parser.add_argument("--epsilon", type=float, default=0.25, help="有界负载系数 ε：后端在途数超过 平均值×(1+ε) 时溢出（默认 0.25）")
    args = parser.parse_args()

    backends = parse_backends(args.backends)
    state = RouterState(backends, health_interval=args.health_interval, epsilon=args.epsilon)
    state.probe_once()
    state.start_probing()
    handler = make_handler(state, args.prefix_chars, args.timeout)
    server = RouterHTTPServer((args.host, args.port), handler)
    print(f"[prefix_router] 监听 http://{args.host}:{args.port} -> {len(backends)} 个后端：{args.backends}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
