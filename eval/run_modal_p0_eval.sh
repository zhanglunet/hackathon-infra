#!/usr/bin/env bash
# run_modal_p0_eval.sh -- 零空转执行 P0-3（native BFCL）+ P0-1/P0-2（272K + YaRN + 32K 精度对照）
#
# 设计原则：
#   1. 离线先备齐所有数据、依赖与 BFCL 垫片，再起 Modal 部署（杜绝 GPU 空转）
#   2. 用 trap 保证任何报错或 Ctrl-C 都自动停掉本轮 Modal 部署（仅当本轮
#      部署成功时）并删除 BFCL 隔离环境；已装的 bfcl_eval 包原样不动
#   3. FP8 阶段把 P0-3（native BFCL）与 P0-1/P0-2（AIME/GPQA/LCB）合并在同一次部署内跑完，省一次冷启动
#   4. 部署前先排掉残留的同名部署，/health 200 后再验 /v1/models 的 served 名，
#      防止旧实例冒充本轮端点（#5 评审加的隔离要求）
#
# 用法（在已配好 modal / evalscope / bfcl_eval 的机器上、仓库根目录执行）：
#   # 第 0 步：纯离线预检 + 拉取缺失的 LCB parquet（$0，不碰 GPU）
#   bash eval/run_modal_p0_eval.sh prep
#
#   # 第 1 步：FP8 全套（272K + YaRN + qwen3_coder/qwen3 parser；先测 native BFCL，再跑 32K 三集）
#   # 预计 H100 墙钟 20–33 分钟，约 $1.4–$2.2
#   bash eval/run_modal_p0_eval.sh fp8
#
#   # 只跑 P0-3（native BFCL 冒烟 + 50 题），预计 H100 5–8 分钟，约 $0.35–$0.55
#   RUN_EVALSCOPE=0 bash eval/run_modal_p0_eval.sh fp8
#
#   # 第 2 步：BF16 同口径对照（跑完 FP8 核对余额后再跑；可设 RUN_DATASETS="aime lcb" 省 ~40% 费用）
#   # 预计 H100 25–40 分钟（约 $1.7–$2.6），若显存紧张可设 MODAL_GPU=H200
#   bash eval/run_modal_p0_eval.sh bf16
#
# 可选环境变量：
#   MODAL_GPU         默认 H100（BF16 显存吃紧时可改 H200）
#   MAX_MODEL_LEN     默认 272000（比赛要求 272K + YaRN）
#   MAX_NUM_SEQS      默认 32
#   MAX_TOKENS        默认 32768（解决 P0-1 4096 截断）
#   TEMPERATURE       默认 1.0（对齐 Qwen3.8-27B generation_config.json；如需对齐官方包可覆盖）
#   TOP_P             默认 0.95
#   EVAL_BATCH_SIZE   默认 32（配合 MODAL_MAX_INPUTS=128 拉高并发压低墙钟）
#   RUN_BFCL          默认 1（fp8 跑 native BFCL；bf16 默认 0，可设 1）
#   RUN_EVALSCOPE     默认 1
#   RUN_DATASETS      默认 "aime gpqa lcb"（可缩减为 "aime lcb" 省费用）
#   MODAL_APP_URL     默认 https://zhanglunet--qwen27b-vllm-serve.us-east.modal.run
#   TAG               输出目录时间戳后缀，默认 YYYYMMDD_HHMMSS

set -euo pipefail

MODE="${1:-prep}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export PATH="$HOME/.local/bin:$PATH"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1}"

MODAL_GPU="${MODAL_GPU:-H100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-272000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
RUN_DATASETS="${RUN_DATASETS:-aime gpqa lcb}"
MODAL_APP_URL="${MODAL_APP_URL:-https://zhanglunet--qwen27b-vllm-serve.us-east.modal.run}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"

if [ "$MODE" = "bf16" ]; then
    QUANT="bf16"
    RUN_BFCL="${RUN_BFCL:-0}"
    BFCL_MODEL_ID="qwen3.8-27b-bf16-modal-FC"
else
    QUANT="fp8"
    RUN_BFCL="${RUN_BFCL:-1}"
    BFCL_MODEL_ID="qwen3.8-27b-fp8-modal-FC"
fi
RUN_EVALSCOPE="${RUN_EVALSCOPE:-1}"

LCB_PARQUET="$REPO_ROOT/eval/data/lcb_v6_flat/test.parquet"
GPQA_JSONL="$REPO_ROOT/eval/data/gpqa_diamond.jsonl"
SHIM_Q="$REPO_ROOT/eval/data/bfcl_v3_shim/BFCL_v3_simple_50.json"
SHIM_A="$REPO_ROOT/eval/data/bfcl_v3_shim/BFCL_v3_simple_50_possible_answer.json"

DEPLOYED=0
BFCL_ISOLATED_DIR=""
T_DEPLOY_START=0

cleanup() {
    local rc=$?
    set +e
    # BFCL 垫片只装在本轮临时目录的包副本里，已装的包原样不动，
    # 这里只需删临时目录，不用恢复原文件（#5 评审：垫片独立环境）。
    if [ -n "$BFCL_ISOLATED_DIR" ] && [ -d "$BFCL_ISOLATED_DIR" ]; then
        rm -rf "$BFCL_ISOLATED_DIR"
        echo "== [清理] 已删除本轮 BFCL 隔离环境 $BFCL_ISOLATED_DIR =="
    fi
    if [ "$DEPLOYED" -eq 1 ]; then
        echo
        echo "== [清理] 停止本轮 Modal 部署 qwen27b-vllm（防止 GPU 空转）=="
        python3 -m modal app stop --yes qwen27b-vllm || true
        local t_end
        t_end="$(date +%s)"
        local elapsed=$((t_end - T_DEPLOY_START))
        local rate="3.95"
        if [[ "$MODAL_GPU" == H200* ]]; then rate="4.54"; fi
        python3 -c "print(f'Modal 墙钟：{${elapsed}/60:.1f} 分钟（按 ${MODAL_GPU} \${rate}/h 估算约 \${${elapsed}/3600*${rate}:.2f}）')"
        local code
        code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 "$MODAL_APP_URL/health" || echo 000)"
        echo "停机复核 /health HTTP 状态码：$code（预期 404）"
        if [ "$code" != "404" ] && [ "$code" != "000" ]; then
            echo "[警告] 停机后 /health 仍返回 $code，部署可能还在跑、继续计费！"
            echo "       请手动执行：python3 -m modal app stop --yes qwen27b-vllm"
        fi
    fi
    exit "$rc"
}
trap cleanup EXIT INT TERM

check_prereqs() {
    echo "== 离线预检（\$0，不碰 GPU）=="
    python3 -c "import modal" 2>/dev/null || { echo "[错误] 未安装 modal，请先 pip install modal && modal setup"; exit 1; }
    if [ "$RUN_EVALSCOPE" = "1" ]; then
        command -v evalscope >/dev/null 2>&1 || { echo "[错误] 找不到 evalscope，请确认 PATH 含 ~/.local/bin"; exit 1; }
    fi
    if [ "$RUN_BFCL" = "1" ]; then
        python3 -c "import bfcl_eval" 2>/dev/null || { echo "[错误] 未安装 bfcl_eval"; exit 1; }
        [ -f "$SHIM_Q" ] && [ -f "$SHIM_A" ] || { echo "[错误] 缺少 BFCL v3 垫片文件：$SHIM_Q"; exit 1; }
    fi
    [ -f "$GPQA_JSONL" ] || { echo "[错误] 缺少 $GPQA_JSONL"; exit 1; }

    if [[ " $RUN_DATASETS " == *" lcb "* ]] && [ ! -f "$LCB_PARQUET" ]; then
        echo "[提示] 本地缺少 LCB v6 parquet，离线下载（约 124 MB）..."
        mkdir -p "$(dirname "$LCB_PARQUET")"
        curl -fL "https://huggingface.co/datasets/marianna13/livecodebench_code_generation_lite_parquet/resolve/main/v6/test-00000-of-00001.parquet" \
            -o "$LCB_PARQUET"
    fi
    echo "[通过] 依赖与数据全部就绪。"
}

install_bfcl_shim() {
    # 垫片隔离：把已装的 bfcl_eval 整个包拷到本轮临时目录，垫片只装进副本，
    # 再把副本目录插到 PYTHONPATH 最前面；已装的包原样不动，退出时只删临时目录。
    # 之前是直接改 site-packages 里的数据文件：并发轮次会互相覆盖，
    # 被 kill -9 时 trap 跑不到、垫片残留，下轮会拿 v3 数据当 v4 跑（#5 评审）。
    local pkg_root pkg_data n_v4 import_path
    pkg_root="$(python3 -c 'import bfcl_eval, pathlib; print(pathlib.Path(bfcl_eval.__file__).parent.parent)')"
    pkg_data="$pkg_root/bfcl_eval/data"
    # 守卫：先确认已装的是原版 v4 数据（400 行）；若只有 50 行，
    # 说明之前某轮被强杀、垫片残留，复制过去也会是错的，先报错停下。
    n_v4="$(grep -c . "$pkg_data/BFCL_v4_simple_python.json" || true)"
    if [ "$n_v4" != "400" ]; then
        echo "[错误] 已装的 bfcl_eval 数据文件只有 $n_v4 行（原版 v4 应为 400 行），疑似旧垫片残留。"
        echo "       请先恢复：pip install --force-reinstall --no-deps bfcl-eval==2025.10.27.1"
        exit 1
    fi
    BFCL_ISOLATED_DIR="$(mktemp -d)"
    cp -r "$pkg_root/bfcl_eval" "$BFCL_ISOLATED_DIR/bfcl_eval"
    # 删掉拷过来的 __pycache__，避免旧字节码与新路径的源文件 mtime 对不上。
    find "$BFCL_ISOLATED_DIR" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
    cp "$SHIM_Q" "$BFCL_ISOLATED_DIR/bfcl_eval/data/BFCL_v4_simple_python.json"
    cp "$SHIM_A" "$BFCL_ISOLATED_DIR/bfcl_eval/data/possible_answer/BFCL_v4_simple_python.json"
    export PYTHONPATH="$BFCL_ISOLATED_DIR${PYTHONPATH:+:$PYTHONPATH}"
    # 核对：本轮 python3 必须 import 到副本，而不是已装的包。
    import_path="$(python3 -c 'import bfcl_eval; print(bfcl_eval.__file__)')"
    case "$import_path" in
        "$BFCL_ISOLATED_DIR"/*)
            echo "[完成] BFCL v3 垫片已装进本轮隔离环境（$BFCL_ISOLATED_DIR），已装的包未动。"
            ;;
        *)
            echo "[错误] BFCL 隔离环境未生效，import 到的是 $import_path"
            exit 1
            ;;
    esac
}

ensure_no_stale_deploy() {
    # 部署前先确认没有残留的同名部署：若上一轮崩了没停干净，旧实例还在，
    # 本轮部署失败时 /health 照样 200，测到的会是旧实例（#5 评审）。
    # 本脚本按“同一时间只跑一轮”使用，残留部署一律先停掉再部署本轮。
    # 注意：modal app list 也会列出最近停掉的应用，名字命中不一定是在跑；
    # 误判的代价只是一次无害的 stop，宁可多停一次。
    echo "== 预检：确认无残留的 qwen27b-vllm 部署 =="
    local listing
    listing="$(python3 -m modal app list 2>/dev/null || true)"
    if [ -z "$listing" ]; then
        echo "[警告] 查不到 Modal 应用列表（可能离线），跳过残留检查。"
        return 0
    fi
    if echo "$listing" | grep -q "qwen27b-vllm"; then
        echo "[警告] 发现残留的 qwen27b-vllm，先停掉它再部署本轮。"
        python3 -m modal app stop --yes qwen27b-vllm || true
    else
        echo "[通过] 无残留部署。"
    fi
}

wait_health() {
    echo "== 等待 $MODAL_APP_URL/health 返回 200 =="
    for i in $(seq 1 60); do
        code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 60 "$MODAL_APP_URL/health" || echo 000)"
        if [ "$code" = "200" ]; then
            echo "[就绪] /health 返回 200（第 $i 次检查）"
            # 再验一次端点身份：/v1/models 里必须有本轮要用的 served 名，
            # 否则可能是旧实例（或别的部署）在冒充本轮端点（#5 评审）。
            local models
            models="$(curl -s --max-time 15 "$MODAL_APP_URL/v1/models" || true)"
            if echo "$models" | grep -q '"qwen3.8-27b"'; then
                echo "[就绪] /v1/models 确认 served 名 qwen3.8-27b，本轮端点无误"
                return 0
            fi
            echo "[错误] /health 是 200，但 /v1/models 里没有 qwen3.8-27b（端点不是本轮部署）"
            return 1
        fi
        echo "  [$i/60] HTTP $code，10 秒后重试..."
        sleep 10
    done
    echo "[错误] 等待 /health 超时"
    return 1
}

probe_native_tools() {
    echo "== P0-3 预检：验证原生 tools + reasoning_parser 不再报 400 =="
    python3 - "$MODAL_APP_URL/v1/chat/completions" <<'PYEOF'
import json, sys, urllib.request

url = sys.argv[1]
payload = json.dumps({
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "What is the factorial of 5? Use the math_factorial tool."}],
    "tools": [{
        "type": "function",
        "function": {
            "name": "math_factorial",
            "description": "Compute factorial of an integer.",
            "parameters": {
                "type": "object",
                "properties": {"number": {"type": "integer"}},
                "required": ["number"],
            },
        },
    }],
    "tool_choice": "auto",
    "max_tokens": 1024,
    "temperature": 0.0,
}).encode()
req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=120) as resp:
    data = json.loads(resp.read().decode())
msg = data["choices"][0]["message"]
tcs = msg.get("tool_calls") or []
has_reasoning = bool(msg.get("reasoning_content") or msg.get("reasoning"))
print("finish_reason:", data["choices"][0].get("finish_reason"))
print("has_reasoning_field:", has_reasoning)
print("tool_calls:", json.dumps(tcs, ensure_ascii=False))
assert len(tcs) >= 1, f"未解析出原生 tool_calls！完整 message={msg}"
print("[通过] 原生 tool_calls 解析正常！")
PYEOF
}

run_bfcl_native() {
    local out_dir="$REPO_ROOT/eval/outputs/bfcl_${TAG}_${QUANT}_native"
    mkdir -p "$out_dir"
    cat > "$out_dir/test_case_ids_to_generate.json" <<'JSON'
{"simple_python": ["simple_python_0", "simple_python_1", "simple_python_2"]}
JSON
    export BFCL_PROJECT_ROOT="$out_dir"
    export MODAL_BASE_URL="$MODAL_APP_URL/v1"
    export BFCL_MODAL_HANDLER="native"
    export BFCL_MODEL_ID="$BFCL_MODEL_ID"

    echo "== P0-3 小样本 3 题（native handler）=="
    python3 eval/run_bfcl_modal.py --phase generate --run-ids --temperature 0 --num-threads 3
    python3 eval/run_bfcl_modal.py --phase evaluate --partial-eval

    echo "== P0-3 全量 50 题（native handler）=="
    python3 eval/run_bfcl_modal.py --phase generate --temperature 0 --num-threads 16
    python3 eval/run_bfcl_modal.py --phase evaluate
    echo "[完成] native BFCL 结果目录：$out_dir"
}

run_evalscope_suite() {
    local gen_cfg
    gen_cfg="$(printf '{"max_tokens": %s, "temperature": %s, "top_p": %s}' \
        "$MAX_TOKENS" "$TEMPERATURE" "$TOP_P")"
    local work_dir="$REPO_ROOT/eval/outputs/p0_${TAG}_${QUANT}"
    mkdir -p "$work_dir"
    echo "== P0-1 + P0-2 EvalScope（272K + YaRN，gen_cfg=$gen_cfg，batch=$EVAL_BATCH_SIZE）=="

    for ds in $RUN_DATASETS; do
        case "$ds" in
            aime)
                echo "--- AIME 2025（30 题）---"
                evalscope eval --model qwen3.8-27b --eval-type openai_api \
                    --api-url "$MODAL_APP_URL/v1" --api-key EMPTY \
                    --datasets aime25 --dataset-hub huggingface \
                    --limit 30 --eval-batch-size "$EVAL_BATCH_SIZE" \
                    --generation-config "$gen_cfg" --work-dir "$work_dir"
                ;;
            gpqa)
                echo "--- GPQA Diamond（前 50 题）---"
                evalscope eval --model qwen3.8-27b --eval-type openai_api \
                    --api-url "$MODAL_APP_URL/v1" --api-key EMPTY \
                    --datasets general_mcq \
                    --dataset-args "{\"general_mcq\": {\"dataset_id\": \"$GPQA_JSONL\"}}" \
                    --limit 50 --eval-batch-size "$EVAL_BATCH_SIZE" \
                    --generation-config "$gen_cfg" --work-dir "$work_dir"
                ;;
            lcb)
                echo "--- LiveCodeBench v6（前 40 题）---"
                evalscope eval --model qwen3.8-27b --eval-type openai_api \
                    --api-url "$MODAL_APP_URL/v1" --api-key EMPTY \
                    --datasets live_code_bench \
                    --dataset-args "{\"live_code_bench\": {\"dataset_id\": \"$REPO_ROOT/eval/data/lcb_v6_flat\", \"subset_list\": [\"default\"]}}" \
                    --limit 40 --eval-batch-size "$EVAL_BATCH_SIZE" \
                    --generation-config "$gen_cfg" --work-dir "$work_dir"
                ;;
            *)
                echo "[警告] 未知数据集 '$ds'，跳过"
                ;;
        esac
    done

    echo
    echo "== 本轮截断率与精度汇总 =="
    python3 eval/check_truncation.py "$work_dir"/*
}

case "$MODE" in
    prep)
        check_prereqs
        echo "预检通过。下一步执行：bash eval/run_modal_p0_eval.sh fp8"
        ;;
    fp8|bf16)
        check_prereqs
        if [ "$RUN_BFCL" = "1" ]; then
            install_bfcl_shim
        fi
        ensure_no_stale_deploy
        echo "== 启动 Modal 部署：GPU=$MODAL_GPU QUANT=$QUANT MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS =="
        T_DEPLOY_START="$(date +%s)"
        # 注意：DEPLOYED 只在部署成功后才置 1。之前是部署前就置 1，
        # 部署失败时 trap 照样去停共享应用，可能杀掉别人的部署（#5 评审）。
        if MODAL_GPU="$MODAL_GPU" \
            QUANT="$QUANT" \
            MAX_MODEL_LEN="$MAX_MODEL_LEN" \
            MAX_NUM_SEQS="$MAX_NUM_SEQS" \
            TOOL_CALL_PARSER="qwen3_coder" \
            REASONING_PARSER="qwen3" \
            MODAL_MAX_INPUTS="128" \
                python3 -m modal deploy modal/modal_vllm_qwen27b.py; then
            DEPLOYED=1
        else
            echo "[错误] Modal 部署失败，本轮没有部署任何实例，不停共享应用。"
            exit 1
        fi
        wait_health
        if [ "$RUN_BFCL" = "1" ]; then
            probe_native_tools
            run_bfcl_native
        fi
        if [ "$RUN_EVALSCOPE" = "1" ]; then
            run_evalscope_suite
        fi
        ;;
    *)
        echo "用法：bash eval/run_modal_p0_eval.sh [prep|fp8|bf16]"
        exit 1
        ;;
esac
