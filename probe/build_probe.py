#!/usr/bin/env python3
"""Build a ~272K-token probe prompt from Moby Dick (real long text).

Counts with the exact model tokenizer (Qwen/Qwen3.8-27B-FP8) and the
chat template, so the number matches what vLLM sees. Target: prompt
tokens ~= 271500, leaving room for 64 output tokens inside
MAX_MODEL_LEN=272000. Writes prompt_272k.txt + probe_meta.json.
"""
import json

from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3.8-27B-FP8"
TARGET_PROMPT_TOKENS = 271500
MAX_OUTPUT_TOKENS = 64
MAX_MODEL_LEN = 272000

tok = AutoTokenizer.from_pretrained(MODEL)

with open("moby.txt", encoding="utf-8", errors="replace") as f:
    raw = f.read()

start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK 2701 ***"
end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK 2701 ***"
body = raw.split(start_marker, 1)[1] if start_marker in raw else raw
body = body.split(end_marker, 1)[0] if end_marker in body else body
body = body.strip()
print(f"body chars: {len(body)}")

QUESTION = "\n\n请用一句话总结上面这段文字的主题。"

# Tokenize the body once, then binary-search the char cut that makes the
# full templated prompt land on target (template overhead included).
body_ids = tok(body, add_special_tokens=False)["input_ids"]
print(f"body tokens: {len(body_ids)}")

# Build candidate messages and count with the chat template.
def prompt_tokens(n_body_tokens: int) -> int:
    content = tok.decode(body_ids[:n_body_tokens]) + QUESTION
    msgs = [{"role": "user", "content": content}]
    out = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    # transformers 5.x returns a BatchEncoding (dict-like, NOT a dict);
    # len() of it is the number of keys (2), not tokens.
    ids = out["input_ids"] if not isinstance(out, list) else out
    return len(ids), content

# Binary search n so prompt_tokens(n) is just under TARGET.
lo, hi = 0, len(body_ids)
while lo < hi:
    mid = (lo + hi + 1) // 2
    n, _ = prompt_tokens(mid)
    if n <= TARGET_PROMPT_TOKENS:
        lo = mid
    else:
        hi = mid - 1

n_tokens, content = prompt_tokens(lo)
print(f"body tokens used: {lo}, prompt tokens (templated): {n_tokens}")
print(f"target: {TARGET_PROMPT_TOKENS}, delta: {n_tokens - TARGET_PROMPT_TOKENS}")
print(f"prompt + {MAX_OUTPUT_TOKENS} output = {n_tokens + MAX_OUTPUT_TOKENS} "
      f"(limit {MAX_MODEL_LEN}, margin {MAX_MODEL_LEN - n_tokens - MAX_OUTPUT_TOKENS})")

with open("prompt_272k.txt", "w", encoding="utf-8") as f:
    f.write(content)

meta = {
    "model": MODEL,
    "source": "Moby Dick (Project Gutenberg #2701, pglaf.org mirror), body only",
    "body_tokens_used": lo,
    "body_tokens_total": len(body_ids),
    "prompt_tokens_templated": n_tokens,
    "target_prompt_tokens": TARGET_PROMPT_TOKENS,
    "delta_vs_target": n_tokens - TARGET_PROMPT_TOKENS,
    "max_output_tokens": MAX_OUTPUT_TOKENS,
    "prompt_plus_output": n_tokens + MAX_OUTPUT_TOKENS,
    "max_model_len": MAX_MODEL_LEN,
    "margin_tokens": MAX_MODEL_LEN - n_tokens - MAX_OUTPUT_TOKENS,
    "prompt_chars": len(content),
}
with open("probe_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print("wrote prompt_272k.txt + probe_meta.json")
