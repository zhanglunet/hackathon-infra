"""压测脚本共用的小工具：HTTP、/metrics 解析、分位数、SLO 判定。只用标准库。"""
import json
import math
import re
import time
import urllib.request


def http_get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def served_model(base_url):
    """从 /v1/models 取第一个模型名。"""
    d = json.loads(http_get(base_url.rstrip("/") + "/v1/models"))
    return d["data"][0]["id"]


_METRIC_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9eE+.\-]+|NaN|\+Inf|-Inf)\s*$")


def scrape_metrics(base_url):
    """抓 /metrics，返回 {指标名: 所有标签求和后的值}。抓不到返回 None（不影响压测本身）。"""
    try:
        text = http_get(base_url.rstrip("/") + "/metrics")
    except Exception:
        return None
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _METRIC_RE.match(line)
        if not m:
            continue
        try:
            v = float(m.group(3))
        except ValueError:
            continue
        out[m.group(1)] = out.get(m.group(1), 0.0) + v
    return out


def _find(metrics, *needles, exclude=()):
    """按名字片段找计数器（vLLM 各版本指标名略有差异，这里宽松匹配 *_total）。"""
    for k, v in metrics.items():
        if all(n in k for n in needles) and not any(e in k for e in exclude) and k.endswith("_total"):
            return v
    return None


def cache_hit_between(before, after):
    """用两次 /metrics 之间的增量算实际前缀缓存命中率（token 口径）。

    vLLM V1 的指标是 vllm:prefix_cache_queries_total / vllm:prefix_cache_hits_total
    （单位是 token）。外部 KV 连接器的 external_prefix_cache_* 不算在内。
    vLLM 0.34.0 的确切指标名未在真机核实；找不到就返回 None。
    """
    if not before or not after:
        return None
    vals = []
    for snap in (before, after):
        q = _find(snap, "prefix_cache_queries", exclude=("external", "connector"))
        h = _find(snap, "prefix_cache_hits", exclude=("external", "connector"))
        if q is None or h is None:
            return None
        vals.append((q, h))
    dq = vals[1][0] - vals[0][0]
    dh = vals[1][1] - vals[0][1]
    if dq <= 0:
        return None
    return dh / dq


def percentile(xs, p):
    """线性插值分位数（与 numpy 默认一致）。"""
    xs = sorted(x for x in xs if x is not None and not math.isnan(x))
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100.0
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def judge(level, sc, max_error_rate):
    """一级并发的判定：P90 TTFT、P90 TPOT 都在 SLO 内，且错误率不超标，才算过线。"""
    reasons = []
    if level.get("p90_ttft_s") is None or level.get("p90_tpot_ms") is None:
        reasons.append("没有有效数据")
    else:
        if level["p90_ttft_s"] > sc["ttft_s"]:
            reasons.append(f"P90 TTFT {level['p90_ttft_s']:.2f}s > {sc['ttft_s']}s")
        if level["p90_tpot_ms"] > sc["tpot_ms"]:
            reasons.append(f"P90 TPOT {level['p90_tpot_ms']:.1f}ms > {sc['tpot_ms']}ms")
    if level.get("error_rate", 0) > max_error_rate:
        reasons.append(f"错误率 {level['error_rate']:.1%} > {max_error_rate:.0%}")
    return (not reasons), reasons


def now():
    return time.perf_counter()
