from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any


CHOICE_SYSTEM = (
    "\u4f60\u662f\u4e25\u8c28\u7684\u4e2d\u6587\u533b\u5b66\u8003\u8bd5\u52a9\u624b\u3002"
    "\u8bf7\u8f93\u51fa\u6700\u7ec8\u7b54\u6848\u9009\u9879\uff0c\u5e76\u53ef\u4ee5"
    "\u7ed9\u51fa\u4e00\u53e5\u7b80\u77ed\u89e3\u6790\u3002"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def normalize_choice(value: Any) -> str:
    return "".join(sorted(set(re.findall(r"[A-F]", str(value).upper()))))


def extract_choice(text: Any) -> str:
    raw = str(text or "").strip()
    upper = raw.upper()
    patterns = [
        r"(?:\u6b63\u786e\u7b54\u6848|\u7b54\u6848|\u6700\u7ec8\u9009\u9879|\u9009\u62e9|ANSWER|CHOICE)\s*[:\uff1a\u4e3a\u662f-]*\s*([A-F](?:\s*[\u3001,， ]?\s*[A-F]){0,5})",
        r"(?:^|\n)\s*([A-F](?:\s*[\u3001,， ]?\s*[A-F]){0,5})\s*(?:$|\n)",
    ]
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            value = normalize_choice(match.group(1))
            if value:
                return value
    compact = re.sub(r"\s+", "", upper)
    match = re.search(r"(?<![A-Z])([A-F]{1,6})(?![A-Z])", compact)
    return normalize_choice(match.group(1)) if match else ""


def is_multi(row: dict[str, Any]) -> bool:
    return len(normalize_choice(row.get("answer", ""))) > 1


def select_rows(rows: list[dict[str, Any]], mode: str, max_prompts: int, seed: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        gold = normalize_choice(row.get("answer", ""))
        if not gold:
            continue
        pred = extract_choice(row.get("prediction", ""))
        wrong = pred != gold
        if mode == "wrong" and not wrong:
            continue
        if mode == "wrong_or_multi" and not (wrong or is_multi(row)):
            continue
        selected.append(row)

    rng = random.Random(seed)
    if mode == "all":
        rng.shuffle(selected)
    else:
        wrong_rows = [row for row in selected if extract_choice(row.get("prediction", "")) != normalize_choice(row.get("answer", ""))]
        other_rows = [row for row in selected if row not in wrong_rows]
        rng.shuffle(other_rows)
        selected = wrong_rows + other_rows

    if max_prompts > 0:
        selected = selected[:max_prompts]
    return selected


def render_prompt(tokenizer: Any, row: dict[str, Any]) -> str:
    prompt = str(row.get("prompt") or row.get("input") or "")
    messages = [
        {"role": "system", "content": CHOICE_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"<|system|>\n{CHOICE_SYSTEM}\n<|user|>\n{prompt}\n<|assistant|>\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--selection-mode", choices=["wrong", "wrong_or_multi", "all"], default="wrong_or_multi")
    parser.add_argument("--max-prompts", type=int, default=5000)
    parser.add_argument("--samples-per-prompt", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--flush-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    rows = select_rows(read_jsonl(Path(args.predictions)), args.selection_mode, args.max_prompts, args.seed)
    out_path = Path(args.out)
    completed = count_jsonl(out_path) if args.resume else 0
    rows_to_run = rows[completed:]
    mode = "a" if args.resume and completed else "w"

    print(
        json.dumps(
            {
                "predictions": args.predictions,
                "out": args.out,
                "selection_mode": args.selection_mode,
                "selected_prompts": len(rows),
                "completed_existing": completed,
                "remaining_prompts": len(rows_to_run),
                "samples_per_prompt": args.samples_per_prompt,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("CUDA is not available. Re-run after GPU boot, or pass --allow-cpu for smoke tests.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
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
    total_samples = 0
    with out_path.open(mode, encoding="utf-8") as f:
        for prompt_pos, row in enumerate(rows_to_run, start=completed):
            prompt_started = time.perf_counter()
            prompt = render_prompt(tokenizer, row)
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            torch.manual_seed(args.seed + prompt_pos)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=1.05,
                    num_return_sequences=args.samples_per_prompt,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

            samples: list[dict[str, Any]] = []
            input_len = inputs["input_ids"].shape[-1]
            for sample_idx, one_output in enumerate(output_ids):
                gen_ids = one_output[input_len:]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                samples.append(
                    {
                        "sample_index": sample_idx,
                        "text": text,
                        "choice": extract_choice(text),
                    }
                )
            total_samples += len(samples)

            merged = dict(row)
            merged["sampled_prompt_index"] = prompt_pos
            merged["gold_choice"] = normalize_choice(row.get("answer", ""))
            merged["greedy_choice"] = extract_choice(row.get("prediction", ""))
            merged["samples"] = samples
            merged["sampling_seconds"] = round(time.perf_counter() - prompt_started, 4)
            f.write(json.dumps(merged, ensure_ascii=False) + "\n")
            written += 1
            if args.flush_every and written % args.flush_every == 0:
                f.flush()
            if args.progress_every and written % args.progress_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    json.dumps(
                        {
                            "progress_prompts": completed + written,
                            "written_this_run": written,
                            "total_samples_this_run": total_samples,
                            "samples_per_second": round(total_samples / elapsed, 4) if elapsed else 0.0,
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
                "written_prompts_this_run": written,
                "total_file_rows": completed + written,
                "total_samples_this_run": total_samples,
                "seconds": round(elapsed, 3),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
