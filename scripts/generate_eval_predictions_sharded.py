from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterator


def iter_jsonl_slice(path: Path, start: int, limit: int | None) -> Iterator[tuple[int, dict[str, Any]]]:
    emitted = 0
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx < start:
                continue
            if limit is not None and emitted >= limit:
                break
            line = line.strip()
            if not line:
                continue
            yield line_idx, json.loads(line)
            emitted += 1


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def render_prompt(tokenizer: Any, row: dict[str, Any], task: str) -> str:
    prompt = str(row.get("prompt") or row.get("input") or "")
    if task == "choice":
        system = "你是严谨的中文医学考试助手。请只输出最终答案选项，例如 A 或 BCDEF，不要输出解析。"
    elif task == "redflag":
        system = (
            "你是谨慎的医疗分诊助手。请输出严格 JSON，包含 risk_level、possible_reason、"
            "key_actions、avoid_actions、disclaimer。"
        )
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
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows from --start.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--flush-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    if args.start < 0:
        raise SystemExit("--start must be non-negative")
    if args.limit < 0:
        raise SystemExit("--limit must be non-negative")

    out_path = Path(args.out)
    completed = count_jsonl(out_path) if args.resume else 0
    effective_start = args.start + completed
    effective_limit = None if args.limit == 0 else max(args.limit - completed, 0)
    mode = "a" if args.resume and completed else "w"

    meta = {
        "eval": args.eval,
        "task": args.task,
        "out": args.out,
        "start": args.start,
        "limit": args.limit,
        "completed_existing": completed,
        "effective_start": effective_start,
        "effective_limit": effective_limit,
    }
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)

    if args.dry_run:
        sample = next(iter_jsonl_slice(Path(args.eval), effective_start, 1), None)
        if sample is not None:
            fake_tokenizer = type("T", (), {"apply_chat_template": lambda *a, **k: "<chat-template-ok>"})()
            print(render_prompt(fake_tokenizer, sample[1], args.task), flush=True)
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

        print(json.dumps({"loading_adapter": args.adapter}, ensure_ascii=False), flush=True)
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    written = 0
    with out_path.open(mode, encoding="utf-8") as f:
        for row_index, row in iter_jsonl_slice(Path(args.eval), effective_start, effective_limit):
            row_started = time.perf_counter()
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
                    use_cache=True,
                )
            gen_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
            pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            merged = dict(row)
            merged["eval_index"] = row_index
            merged["prediction"] = pred
            merged["generation_seconds"] = round(time.perf_counter() - row_started, 4)
            f.write(json.dumps(merged, ensure_ascii=False) + "\n")
            written += 1
            if args.flush_every and written % args.flush_every == 0:
                f.flush()
            if args.progress_every and written % args.progress_every == 0:
                elapsed = time.perf_counter() - started
                rate = written / elapsed if elapsed > 0 else 0.0
                print(
                    json.dumps(
                        {
                            "progress": completed + written,
                            "written_this_run": written,
                            "last_eval_index": row_index,
                            "rows_per_second": round(rate, 4),
                            "elapsed_seconds": round(elapsed, 1),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "out": args.out,
                "written_this_run": written,
                "total_file_rows": completed + written,
                "seconds": round(elapsed, 3),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
