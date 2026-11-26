"""用 NVIDIA AIPerf（评审组用的工具）跑一级并发，把它的输出换成和 builtin_client 一样的字段。

已在 AIPerf 0.14.0 上核实的用法（2026-11-24，本地对假服务实跑）：
  - 自定义数据：--custom-dataset-type single_turn，每行 {"text": ..., "output_length": N}
    output_length 会变成请求里的 max_tokens。
    注意：**不能带 session_id**。带了的话，同一 session 的多行会被拼成一段多轮对话，
    后一轮会带上前面所有轮次和模型回复，输入长度就不对了。
  - --dataset-sampling-strategy sequential：按文件顺序发（会话轮转顺序由 make_datasets 排好）
  - --goodput "time_to_first_token:<ms> inter_token_latency:<ms>"：按单个请求是否同时满足两条 SLO 算 Goodput
  - --use-server-token-count：token 数用服务端 usage（不依赖本地 tokenizer 和模型完全一致）
  - 结果在 <artifact-dir>/profile_export_aiperf.json：各指标是 {unit, avg, p50, p90, p99, ...}
  - AIPerf 的 inter_token_latency = 每个请求 (最后 token − 首 token) / (输出 token − 1)，就是 TPOT
"""
import json
import os
import subprocess


def _v(d, key, stat="avg"):
    x = d.get(key)
    if isinstance(x, dict):
        return x.get(stat)
    return None


def run(aiperf, url, model, tokenizer, records, conc, sc, extra, art_dir, timeout):
    os.makedirs(art_dir, exist_ok=True)
    inp = os.path.join(art_dir, "input.jsonl")
    with open(inp, "w") as f:
        for r in records:
            f.write(json.dumps({"text": r["text"], "output_length": r["output_tokens"]}, ensure_ascii=False) + "\n")
    sys_prompts = {r.get("system") for r in records}
    sys_args = []
    if sys_prompts != {None}:
        # 共同 system prompt 走 AIPerf 的 --system-prompt-file（0.14.0 会把它作为 system 消息
        # 加到每个请求开头），要求整级所有请求的 system 完全相同
        if len(sys_prompts) != 1:
            raise ValueError("aiperf 引擎要求同一级所有请求的 system prompt 相同")
        sys_p = os.path.join(art_dir, "system_prompt.txt")
        with open(sys_p, "w") as f:
            f.write(sys_prompts.pop())
        sys_args = ["--system-prompt-file", sys_p]
    cmd = [
        aiperf, "profile",
        "-m", model,
        "--url", url,
        "--endpoint-type", "chat",
        "--streaming",
        "--input-file", inp,
        "--custom-dataset-type", "single_turn",
        "--dataset-sampling-strategy", "sequential",
        "--concurrency", str(conc),
        "--request-count", str(len(records)),
        "--tokenizer", tokenizer,
        "--use-server-token-count",
        "--request-timeout-seconds", str(timeout),
        "--goodput", f"time_to_first_token:{sc['ttft_s'] * 1000:g} inter_token_latency:{sc['tpot_ms']:g}",
        "--artifact-dir", art_dir,
    ] + sys_args
    for k, v in extra.items():
        cmd += ["--extra-inputs", f"{k}:{str(v).lower() if isinstance(v, bool) else v}"]
    log = os.path.join(art_dir, "aiperf_stdout.log")
    with open(os.path.join(art_dir, "command.txt"), "w") as f:
        f.write(" ".join(cmd) + "\n")
    with open(log, "w") as lf:
        rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=art_dir)
    res_p = os.path.join(art_dir, "profile_export_aiperf.json")
    if rc != 0 or not os.path.exists(res_p):
        return {"requests": len(records), "errors": len(records), "error_rate": 1.0,
                "error_samples": [f"aiperf 退出码 {rc}，见 {log}"],
                "p50_ttft_s": None, "p90_ttft_s": None, "p99_ttft_s": None,
                "p50_tpot_ms": None, "p90_tpot_ms": None, "p99_tpot_ms": None,
                "request_throughput_rps": None, "output_tps": None, "goodput_rps": None,
                "good_requests": 0, "duration_s": None}
    d = json.load(open(res_p))
    ms = lambda key, stat: (_v(d, key, stat) / 1000.0) if _v(d, key, stat) is not None else None  # noqa: E731
    # 0.14.0 实测：request_count = 成功的请求数；completed_request_count 含失败的
    ok_n = _v(d, "request_count") or 0
    err_rate_pct = _v(d, "request_error_rate")
    if err_rate_pct is not None:
        errors = round(len(records) * err_rate_pct / 100.0)
    else:
        errors = max(0, len(records) - int(ok_n))
    samples = []
    for e in d.get("error_summary") or []:
        if isinstance(e, dict):
            samples.append(str(e.get("error_details") or e)[:300])
    return {
        "requests": len(records),
        "errors": errors,
        "error_rate": errors / len(records) if records else 0.0,
        "error_samples": samples[:3],
        "duration_s": _v(d, "benchmark_duration"),
        "p50_ttft_s": ms("time_to_first_token", "p50"),
        "p90_ttft_s": ms("time_to_first_token", "p90"),
        "p99_ttft_s": ms("time_to_first_token", "p99"),
        "p50_tpot_ms": _v(d, "inter_token_latency", "p50"),
        "p90_tpot_ms": _v(d, "inter_token_latency", "p90"),
        "p99_tpot_ms": _v(d, "inter_token_latency", "p99"),
        "request_throughput_rps": _v(d, "request_throughput"),
        "output_tps": _v(d, "output_token_throughput"),
        "goodput_rps": _v(d, "goodput"),
        "good_requests": _v(d, "good_request_count"),
        "input_tokens_mean": _v(d, "input_sequence_length"),
        "output_tokens_mean": _v(d, "output_sequence_length"),
        "aiperf_version": d.get("aiperf_version"),
    }
