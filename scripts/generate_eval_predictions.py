from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_prompt(tokenizer: Any, row: dict[str, Any], task: str) -> str:
    prompt = str(row.get("prompt") or row.get("input") or "")
    if task == "choice":
        system = "你是严谨的中文医学考试助手。请只输出最终答案选项，例如 A 或 BCDE，不要输出解析。"
    elif task == "redflag":
        system = "你是谨慎的医疗分诊助手。请输出严格 JSON，包含 risk_level、possible_reason、key_actions、avoid_actions、disclaimer。"
    else:
        system = "你是专业、谨慎的中文医疗助手，回答不能替代线下医生诊断。"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--eval", required=True)
    parser.add_argument("--task", choices=["choice", "redflag", "dialog"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.eval), args.limit)
    print(json.dumps({"eval": args.eval, "rows": len(rows), "task": args.task}, ensure_ascii=False, indent=2))
    if args.dry_run:
        if rows:
            print(render_prompt(type("T", (), {"apply_chat_template": lambda *a, **k: "<chat-template-ok>"})(), rows[0], args.task))
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("CUDA is not available. Re-run after GPU boot, or pass --allow-cpu for tiny smoke tests.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    predictions = []
    started = time.perf_counter()
    for idx, row in enumerate(rows, 1):
        prompt = render_prompt(tokenizer, row, args.task)
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
        pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        merged = dict(row)
        merged["prediction"] = pred
        predictions.append(merged)
        if idx % 20 == 0:
            print(f"[eval-gen] {idx}/{len(rows)}")

    write_jsonl(Path(args.out), predictions)
    elapsed = time.perf_counter() - started
    print(json.dumps({"out": args.out, "rows": len(predictions), "seconds": elapsed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
