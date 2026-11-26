#!/usr/bin/env python3
"""路由器对照：同一份数据「直连各副本」和「经 router/prefix_router.py」各跑一遍，并排比较。

为什么要有：路由器夹在压测工具和各副本中间，如果它把同一会话送到不同副本，
前缀缓存就会失效，agent / coding 档的命中率和 TTFT 直接变差，而单看经路由器的那一份结果
是看不出来的。和直连对照，差多少一目了然。issue #14 的验证就是这么做的。

两路都用内置引擎（builtin），保证差别只来自路由：
  直连   run_ladder.py --url 副本1,副本2,...   客户端按 session_id 把会话固定到副本（理想粘滞）
  经路由 run_ladder.py --url <router>          由路由器自己按前缀选副本
每一路开跑前都会换新的随机标记（run_ladder 自带），上一路留在缓存里的内容不会帮下一路作弊。

比较什么：每级的实测命中率、P90 TTFT / TPOT、错误数、Goodput；经路由那一路再加
各副本分到的请求占比（读路由器 /router/stats）和故障转移次数。

用法：
  # 上机：副本已经起好（8001～8008），路由器也已经起好（8000）
  python3 loadtest/router_check.py --backends http://127.0.0.1:8001,...,http://127.0.0.1:8008 \\
      --router http://127.0.0.1:8000 --data loadtest/data --out loadtest/results/router-check

  # 不给 --router：脚本自己在 --router-port 起一个 prefix_router.py（--router-args 透传参数）
  python3 loadtest/router_check.py --backends ... --data loadtest/data --out ... --router-args "--epsilon 1"

  # 本地无 GPU：--mock 4 起 4 个假服务当副本，并自动生成带共同 system prompt 的缩小版数据
  python3 loadtest/router_check.py --mock 4 --out /tmp/router-check
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from scenarios import SCENARIOS  # noqa: E402

PY = sys.executable


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_http(url, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status in (200, 503):
                    return True
        except urllib.error.HTTPError as e:
            if e.code == 503:  # 路由器降级时 /health 返回 503，但服务在
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


def get_json(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.load(r)
    except Exception:
        return None


def load_levels(out_dir):
    levels = {}
    for scen in os.listdir(out_dir) if os.path.isdir(out_dir) else []:
        d = os.path.join(out_dir, scen)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.startswith("c") and f.endswith(".json"):
                x = json.load(open(os.path.join(d, f)))
                levels[(x["scenario"], x["concurrency"])] = x
    return levels


def fmt(v, spec):
    return "—" if v is None else spec.format(v)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backends", default="", help="逗号分隔的副本地址（直连那一路用，也是自起路由器的后端）")
    p.add_argument("--router", default="", help="已经起好的路由器地址；不给则自己起一个")
    p.add_argument("--router-port", type=int, default=0, help="自起路由器的端口（默认随机空闲端口）")
    p.add_argument("--router-args", default="", help="透传给 prefix_router.py 的参数，如 \"--epsilon 1\"")
    p.add_argument("--data", default="", help="make_datasets.py 的输出目录（--mock 时可不给，自动生成）")
    p.add_argument("--out", required=True, help="结果目录（下面会有 direct/ 和 router/）")
    p.add_argument("--scenarios", default="agent,coding", help="默认只比有缓存命中的两档")
    p.add_argument("--concurrency", default="32", help="跑哪几级并发（逗号分隔），默认 32")
    p.add_argument("--model", default="")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--mock", type=int, default=0, help="本地自测：起 N 个假服务当副本")
    p.add_argument("--force", action="store_true", help="--out 已存在时清空")
    a = p.parse_args()

    if os.path.exists(a.out):
        if not a.force:
            sys.exit(f"[错误] {a.out} 已存在，换一个目录或加 --force")
        shutil.rmtree(a.out)
    os.makedirs(a.out)
    procs = []

    def spawn(cmd, log):
        procs.append(subprocess.Popen(cmd, stdout=open(os.path.join(a.out, log), "w"), stderr=subprocess.STDOUT))

    try:
        backends = [u.strip().rstrip("/") for u in a.backends.split(",") if u.strip()]
        if a.mock:
            backends = []
            for i in range(a.mock):
                port = free_port()
                spawn([PY, os.path.join(HERE, "mock_server.py"), "--port", str(port), "--speedup", "10",
                       "--served-name", "mock-model"], f"mock{i}.log")
                backends.append(f"http://127.0.0.1:{port}")
            for b in backends:
                if not wait_http(b + "/health"):
                    sys.exit(f"[错误] 假服务 {b} 没起来")
        if not backends:
            sys.exit("[错误] 需要 --backends（或 --mock N）")

        data = a.data
        if not data:
            if not a.mock:
                sys.exit("[错误] 需要 --data（make_datasets.py 的输出目录）")
            data = os.path.join(a.out, "data")
            print("== 生成缩小版数据（scale 0.02，带 2 万 token 的共同 system prompt）==", flush=True)
            subprocess.check_call([PY, os.path.join(HERE, "make_datasets.py"), "--out", data, "--scale", "0.02",
                                   "--scenarios", a.scenarios, "--concurrency", a.concurrency,
                                   "--shared-system-tokens", "20000"], stdout=subprocess.DEVNULL)

        router = a.router.rstrip("/")
        if not router:
            port = a.router_port or free_port()
            spawn([PY, os.path.join(REPO, "router", "prefix_router.py"), "--backends", ",".join(backends),
                   "--host", "127.0.0.1", "--port", str(port)] + a.router_args.split(), "router.log")
            router = f"http://127.0.0.1:{port}"
            if not wait_http(router + "/health"):
                sys.exit(f"[错误] 路由器没起来，见 {a.out}/router.log")
            print(f"== 自起路由器 {router}  参数：{a.router_args or '（默认）'} ==", flush=True)

        common = ["--data", data, "--engine", "builtin", "--tokenizer", "builtin", "--scenarios", a.scenarios,
                  "--concurrency", a.concurrency, "--no-stop-on-fail", "--timeout", str(a.timeout)]
        if a.model:
            common += ["--model", a.model]
        ladder = os.path.join(HERE, "run_ladder.py")

        print(f"\n== 1/2 直连 {len(backends)} 个副本（客户端按会话固定）==", flush=True)
        subprocess.check_call([PY, ladder, "--url", ",".join(backends), "--out", os.path.join(a.out, "direct"),
                               "--label", "直连各副本"] + common)

        stats0 = get_json(router + "/router/stats")
        print(f"\n== 2/2 经路由器 {router} ==", flush=True)
        subprocess.check_call([PY, ladder, "--url", router, "--out", os.path.join(a.out, "router"),
                               "--label", "经 prefix_router"] + common)
        stats1 = get_json(router + "/router/stats")

        d_lv, r_lv = load_levels(os.path.join(a.out, "direct")), load_levels(os.path.join(a.out, "router"))
        L = ["# 路由器对照：直连 vs 经 prefix_router\n",
             f"- 副本：{', '.join(backends)}",
             f"- 路由器：{router}" + (f"（自起，参数 `{a.router_args}`）" if not a.router and a.router_args else ""),
             f"- 数据：`{data}`；两路都用 builtin 引擎\n",
             "| 场景 | 并发 | 实测命中 直连 → 路由 | P90 TTFT 直连 → 路由 | P90 TPOT 直连 → 路由 | 错误 直连 / 路由 | Goodput 直连 → 路由 |",
             "|---|---|---|---|---|---|---|"]
        warn = []
        for key in sorted(set(d_lv) | set(r_lv)):
            d, r = d_lv.get(key, {}), r_lv.get(key, {})
            hd, hr = d.get("measured_cache_hit"), r.get("measured_cache_hit")
            L.append(f"| {SCENARIOS[key[0]]['title']} | {key[1]} | {fmt(hd, '{:.2f}')} → **{fmt(hr, '{:.2f}')}** | "
                     f"{fmt(d.get('p90_ttft_s'), '{:.2f}s')} → {fmt(r.get('p90_ttft_s'), '{:.2f}s')} | "
                     f"{fmt(d.get('p90_tpot_ms'), '{:.1f}ms')} → {fmt(r.get('p90_tpot_ms'), '{:.1f}ms')} | "
                     f"{d.get('errors', '—')} / {r.get('errors', '—')} | "
                     f"{fmt(d.get('goodput_rps'), '{:.3f}')} → {fmt(r.get('goodput_rps'), '{:.3f}')} |")
            if hd is not None and hr is not None and hd - hr > 0.03:
                warn.append(f"{key[0]} c{key[1]} 命中率经路由器低了 {hd - hr:.2f}")
        if stats0 and stats1:
            b0 = {b["backend"]: b["total_requests"] for b in stats0["backends"]}
            share = [(b["backend"], b["total_requests"] - b0.get(b["backend"], 0), b["healthy"])
                     for b in stats1["backends"]]
            tot = sum(n for _, n, _ in share) or 1
            L.append("\n经路由器那一路各副本分到的请求（含预热；读 /router/stats 前后差）：\n")
            L.append("| 副本 | 请求数 | 占比 | 健康 |")
            L.append("|---|---|---|---|")
            for b, n, h in share:
                L.append(f"| {b} | {n} | {n / tot:.0%} | {'✅' if h else '❌'} |")
            fo = stats1["total_failover"] - stats0["total_failover"]
            L.append(f"\n故障转移次数：{fo}")
            mx = max(n for _, n, _ in share) / tot
            if len(share) > 1 and mx > 2.0 / len(share):
                warn.append(f"负载不均：最忙的副本拿了 {mx:.0%}（均分应为 {1 / len(share):.0%}）")
        else:
            L.append("\n（没读到 /router/stats：路由器不是 prefix_router.py，或版本太旧）")
        L.append("\n## 结论\n")
        if warn:
            L.extend(f"- ⚠️ {w}" for w in warn)
            L.append("- 命中率掉了：先试 `--router-args \"--epsilon 1\"`（有界负载放宽，会话不容易被挤到别的副本）；"
                     "共同前缀写在 user 消息里时，把 `--prefix-chars` 调到比共同前缀长。")
        else:
            L.append("- ✅ 经路由器的命中率和直连相差不超过 0.03，负载没有明显偏斜")
        path = os.path.join(a.out, "router_check.md")
        open(path, "w").write("\n".join(L) + "\n")
        print("\n" + "\n".join(L) + f"\n\n报告：{path}", flush=True)
    finally:
        for pr in procs:
            pr.terminate()
        for pr in procs:
            try:
                pr.wait(timeout=5)
            except Exception:
                pr.kill()


if __name__ == "__main__":
    main()
