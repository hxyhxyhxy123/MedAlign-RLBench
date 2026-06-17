from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


def normalize_choice(value: Any) -> str:
    return "".join(sorted(set(re.findall(r"[A-F]", str(value or "").upper()))))


def extract_choice(text: Any) -> str:
    upper = str(text or "").upper()
    patterns = [
        r"(?:正确答案|答案|最终选项|选择|ANSWER|CHOICE)\s*[:：是为-]*\s*([A-F](?:\s*[、,， ]?\s*[A-F]){0,5})",
        r"(?:^|\n)\s*([A-F](?:\s*[、,， ]?\s*[A-F]){0,5})\s*(?:$|\n)",
    ]
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            choice = normalize_choice(match.group(1))
            if choice:
                return choice
    compact = re.sub(r"\s+", "", upper)
    match = re.search(r"(?<![A-Z])([A-F]{1,6})(?![A-Z])", compact)
    return normalize_choice(match.group(1)) if match else ""


def completion_text(completion: Any) -> str:
    if isinstance(completion, list):
        if completion and isinstance(completion[0], dict):
            return str(completion[0].get("content") or "")
        return str(completion)
    return str(completion or "")


def load_jsonl_dataset(path: Path) -> Dataset:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return Dataset.from_list(rows)


def make_reward_fn() -> Any:
    """Verifiable reward aligned with the exact-match eval metric.

    Uses Jaccard overlap (|P∩G| / |P∪G|) which equals 1.0 only on an exact set
    match and is penalized symmetrically for BOTH missing and extra selections.
    For single-choice this reduces to 1.0/0.0. Unlike F1 it does not reward
    over-selection (a key failure of the first GRPO run on multi-choice), while
    still giving non-zero advantage variance on hard multi groups.
    """

    def choice_verifiable_reward(
        prompts,
        completions,
        completion_ids,
        gold_answer=None,
        is_multi=None,
        **kwargs,
    ):
        rewards: list[float] = []
        gold_list = gold_answer or []
        for idx, completion in enumerate(completions):
            gold = normalize_choice(gold_list[idx] if idx < len(gold_list) else "")
            pred = extract_choice(completion_text(completion))
            if not gold:
                rewards.append(0.0)
                continue
            if not pred:
                rewards.append(-0.3)
                continue
            gold_set, pred_set = set(gold), set(pred)
            union = gold_set | pred_set
            jaccard = len(gold_set & pred_set) / len(union) if union else 0.0
            rewards.append(jaccard)
        return rewards

    return choice_verifiable_reward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True, help="SFT LoRA adapter to continue from")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8,
                        help="number of completions per device; must be a multiple of num-generations")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=1400)
    parser.add_argument("--max-completion-length", type=int, default=24)
    parser.add_argument("--save-steps", type=int, default=80)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_path = Path(args.train_data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_device = args.per_device_train_batch_size
    if per_device % args.num_generations != 0:
        per_device = args.num_generations
        print(f"[grpo] adjusted per_device_train_batch_size -> {per_device} (multiple of num_generations)")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)

    dataset = load_jsonl_dataset(train_path)
    training_args = GRPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        temperature=args.temperature,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=4,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
        log_completions=True,
        num_completions_to_print=2,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=make_reward_fn(),
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    meta = {
        "base_model": args.base_model,
        "adapter": args.adapter,
        "train_data": str(train_path),
        "max_steps": args.max_steps,
        "num_generations": args.num_generations,
        "beta": args.beta,
        "rows": len(dataset),
    }
    (output_dir / "grpo_train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
