"""run_bfcl_modal.py -- 用 bfcl_eval CLI 对 Modal 上的 Qwen3.8-27B-FP8 跑 BFCL v3

背景（见 devlog/2026-11-22-modal-h100-bfcl.md）：
- 本地装的 bfcl_eval 是 v4（VERSION_PREFIX="BFCL_v4"），只认 BFCL_v4_* 数据文件；
  而要跑的是 BFCL v3 simple 前 50 题（simple_0..simple_49）。
  解法：把 v3 simple 前 50 题（问题 + possible_answer）装进 v4 文件名
  （BFCL_v4_simple_python.json）放进包的 data/ 与 data/possible_answer/，
  跑完恢复原 v4 文件。数据本身是 v3，文件名只是过 loader 的垫片。
- 模型注册：bfcl_eval 的 --model 只认 MODEL_CONFIG_MAPPING 里的 id，
  这里运行时注入一个自定义 id `qwen3.8-27b-fp8-modal-FC`，指向 Modal 端点。
- 两种 handler（由环境变量 BFCL_MODAL_HANDLER 选，默认 native）：
  - native：OpenAICompletionsHandler 子类，走原生 tools + 原生 tool_calls；
    唯一改动是 _query_FC 里去掉 `store: False`（vLLM 0.34 不认这个字段，
    原样发会报 400）。base_url/api_key 走 OPENAI_BASE_URL / OPENAI_API_KEY。
  - qwen_agent：QwenAgentThinkHandler 子类（qwen_agent nous 文本格式，
    工具进提示词、模型输出 <functioncall> 文本，qwen_agent 再解析；
    不依赖 vLLM 的原生 tool_calls 解析）。改动两处：model_server 改成
    Modal 地址；generate_cfg 的 temperature 改成 CLI 传入值（原版写死 0.6）。
- underscore_to_dot=True：与官方 Qwen3 FC 配置一致。convert_to_tool 发工具时
  把 `math.factorial` 这类带点名字改成 `math_factorial`（OpenAI 名字正则
  不允许点），checker 比对时同样把 ground_truth 的点改成下划线，两边一致。

用法（先起好 Modal 部署，拿到 https 地址；见 modal/modal_vllm_qwen27b.py）：
  export BFCL_PROJECT_ROOT=$PWD/eval/outputs/bfcl_20261122   # 结果落这里
  export MODAL_BASE_URL=https://zhanglunet--qwen27b-vllm-serve.us-east.modal.run/v1
  export BFCL_MODAL_HANDLER=native            # 或 qwen_agent
  python3 eval/run_bfcl_modal.py --phase generate --run-ids   # 小样本：先写
      eval/outputs/bfcl_20261122/test_case_ids_to_generate.json =
      {"simple_python": ["simple_0", "simple_1", "simple_2"]}
  python3 eval/run_bfcl_modal.py --phase generate             # 全量 50 题
  python3 eval/run_bfcl_modal.py --phase evaluate             # 判分出成绩
"""

import argparse
import os
import sys

MODEL_ID = os.environ.get("BFCL_MODEL_ID", "qwen3.8-27b-fp8-modal-FC")
SERVED_NAME = "qwen3.8-27b"
# 显示名随权重精度走（BF16 对照时用 BFCL_MODEL_ID=qwen3.8-27b-bf16-modal-FC 注入）。
DISPLAY_NAME = (
    "Qwen3.8-27B-BF16 (Modal H100, FC)"
    if "bf16" in MODEL_ID
    else "Qwen3.8-27B-FP8 (Modal H100, FC)"
)


def _register_model():
    """运行时往 MODEL_CONFIG_MAPPING 里注入自定义模型。"""
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from bfcl_eval.model_handler.api_inference.openai_completion import (
        OpenAICompletionsHandler,
    )

    handler_name = os.environ.get("BFCL_MODAL_HANDLER", "native").strip().lower()
    base_url = os.environ.get("MODAL_BASE_URL", "").strip().rstrip("/")
    if not base_url:
        raise SystemExit("缺少 MODAL_BASE_URL（Modal 部署的 /v1 地址，如 https://...modal.run/v1）")

    # OpenAICompletionsHandler 在 __init__ 里就从环境变量建客户端（两个分支
    # 的父类都走这条路），所以先无条件设好；qwen_agent 分支另用自己的配置。
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

    if handler_name == "native":
        # 原生 tools：去掉 vLLM 不认的 `store` 字段，其余照抄父类。
        class ModalNativeHandler(OpenAICompletionsHandler):
            def _query_FC(self, inference_data: dict):
                message = inference_data["message"]
                tools = inference_data["tools"]
                inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}
                kwargs = {
                    "messages": message,
                    "model": self.model_name,
                    "temperature": self.temperature,
                }
                if len(tools) > 0:
                    kwargs["tools"] = tools
                return self.generate_with_backoff(**kwargs)

        handler_cls = ModalNativeHandler

    elif handler_name == "qwen_agent":
        from bfcl_eval.model_handler.api_inference.qwen import QwenAgentThinkHandler
        from qwen_agent.llm import get_chat_model

        class ModalQwenAgentThinkHandler(QwenAgentThinkHandler):
            def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs):
                # 先走父类 __init__（它会建一个 localhost 的 llm），再换成 Modal 的。
                super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
                self.llm = get_chat_model(
                    {
                        "model": model_name,
                        "model_type": "oai",
                        "model_server": base_url,
                        "api_key": "EMPTY",
                        "generate_cfg": {
                            "fncall_prompt_type": "nous",
                            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
                            "thought_in_content": True,
                            # 关键：父类把 temperature 写死 0.6，这里改成 CLI 传入值（0）。
                            "temperature": temperature,
                            "top_p": 1.0,
                            "top_k": -1,
                            "repetition_penalty": 1.0,
                            "presence_penalty": 0.0,
                            "max_input_tokens": 58000,
                            "timeout": 1000,
                            "max_tokens": 16384,
                        },
                    }
                )

        handler_cls = ModalQwenAgentThinkHandler

    else:
        raise SystemExit(f"未知的 BFCL_MODAL_HANDLER={handler_name!r}（native 或 qwen_agent）")

    MODEL_CONFIG_MAPPING[MODEL_ID] = ModelConfig(
        model_name=SERVED_NAME,
        display_name=DISPLAY_NAME,
        url="https://github.com/zhanglunet/hackathon-infra",
        org="Qwen",
        license="apache-2.0",
        model_handler=handler_cls,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=True,
    )
    return handler_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["generate", "evaluate", "all"], default="all")
    parser.add_argument("--run-ids", action="store_true",
                        help="只跑 test_case_ids_to_generate.json 里列的题（小样本用）")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--allow-overwrite", "-o", action="store_true")
    parser.add_argument("--partial-eval", action="store_true",
                        help="判分时允许结果不全（小样本 3 题时用；全量不用）")
    args = parser.parse_args()

    handler_name = _register_model()
    print(f"[接线] 模型 {MODEL_ID}，handler={handler_name}，"
          f"端点={os.environ.get('MODAL_BASE_URL') or os.environ.get('OPENAI_BASE_URL')}")

    from types import SimpleNamespace

    if args.phase in ("generate", "all"):
        from bfcl_eval._llm_response_generation import main as generation_main

        gen_args = SimpleNamespace(
            model=[MODEL_ID],
            test_category=["simple_python"],
            temperature=args.temperature,
            include_input_log=True,
            exclude_state_log=False,
            num_gpus=1,
            num_threads=args.num_threads,
            backend="vllm",
            gpu_memory_utilization=0.9,
            skip_server_setup=True,
            local_model_path=None,
            result_dir=None,
            run_ids=args.run_ids,
            allow_overwrite=args.allow_overwrite,
        )
        generation_main(gen_args)

    if args.phase in ("evaluate", "all"):
        from bfcl_eval.eval_checker.eval_runner import main as evaluation_main

        evaluation_main(
            [MODEL_ID],
            ["simple_python"],
            None,
            None,
            partial_eval=args.partial_eval,
        )


if __name__ == "__main__":
    main()
