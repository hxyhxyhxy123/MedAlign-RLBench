from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import yaml


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


def read_many(paths: str | list[str], limit: int | None = None) -> list[dict[str, Any]]:
    if isinstance(paths, str):
        return read_jsonl(Path(paths), limit)
    rows = []
    for item in paths:
        remaining = None if limit is None else max(0, limit - len(rows))
        if remaining == 0:
            break
        rows.extend(read_jsonl(Path(item), remaining))
    return rows


def as_prompt(row: dict[str, Any]) -> str:
    if row.get("prompt"):
        return str(row["prompt"])
    instruction = str(row.get("instruction", ""))
    user_input = str(row.get("input", ""))
    return instruction if not user_input else f"{instruction}\n{user_input}"


def is_safety_row(row: dict[str, Any]) -> bool:
    text = json.dumps(row, ensure_ascii=False).lower()
    return any(key in text for key in ["redflag", "emergency", "急诊", "急救", "胸痛", "卒中", "中毒"])


def render_user_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"<|user|>\n{prompt}\n<|assistant|>\n"


def encode_response(tokenizer: Any, prompt: str, response: str, cutoff_len: int, device: str) -> dict[str, Any]:
    prompt_text = render_user_prompt(tokenizer, prompt)
    eos = tokenizer.eos_token or ""
    full_text = prompt_text + str(response) + eos
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=cutoff_len,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = input_ids.clone()
    prompt_len = min(len(prompt_ids), labels.shape[1])
    labels[:, :prompt_len] = -100
    if (labels != -100).sum().item() == 0:
        labels[:, -1] = input_ids[:, -1]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def sequence_logprob(model: Any, batch: dict[str, Any]) -> Any:
    import torch

    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = outputs.logits[:, :-1, :].float()
    labels = batch["labels"][:, 1:]
    mask = labels.ne(-100)
    safe_labels = labels.masked_fill(~mask, 0)
    token_logp = torch.log_softmax(logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    lengths = mask.sum(dim=-1).clamp(min=1)
    return (token_logp * mask).sum(dim=-1) / lengths


def preference_loss(model: Any, tokenizer: Any, rows: list[dict[str, Any]], cutoff_len: int, beta: float, device: str) -> Any:
    import torch.nn.functional as F

    losses = []
    for row in rows:
        prompt = as_prompt(row)
        chosen = row.get("chosen") or row.get("output") or ""
        rejected = row.get("rejected") or ""
        chosen_lp = sequence_logprob(model, encode_response(tokenizer, prompt, chosen, cutoff_len, device))
        rejected_lp = sequence_logprob(model, encode_response(tokenizer, prompt, rejected, cutoff_len, device))
        losses.append(-F.logsigmoid(beta * (chosen_lp - rejected_lp)).mean())
    return sum(losses) / max(1, len(losses))


def sft_loss(model: Any, tokenizer: Any, rows: list[dict[str, Any]], cutoff_len: int, device: str) -> Any:
    losses = []
    for row in rows:
        prompt = as_prompt(row)
        response = row.get("output") or row.get("chosen") or ""
        losses.append(-sequence_logprob(model, encode_response(tokenizer, prompt, response, cutoff_len, device)).mean())
    return sum(losses) / max(1, len(losses))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mpo_recipe.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    train_cfg = cfg.get("train", {})
    objective = cfg.get("objective", {})
    pref_rows = read_many(cfg["data"]["preference"], train_cfg.get("max_preference_samples"))
    sft_rows = read_many(cfg["data"]["sft_replay"], train_cfg.get("max_sft_samples"))
    report = {
        "preference_rows": len(pref_rows),
        "sft_replay_rows": len(sft_rows),
        "safety_rows": sum(1 for row in pref_rows if is_safety_row(row)),
        "objective": objective,
        "train": train_cfg,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("CUDA is not available. Re-run after GPU boot, or pass --allow-cpu for tiny smoke tests.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    adapter_path = Path(str(cfg.get("sft_adapter", "")))
    if adapter_path.exists():
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(train_cfg.get("lora_rank", 16)),
            lora_alpha=int(train_cfg.get("lora_alpha", 32)),
            lora_dropout=float(train_cfg.get("lora_dropout", 0.05)),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_cfg)
    model.train()
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(train_cfg.get("learning_rate", 5.0e-6)),
    )
    cutoff_len = int(train_cfg.get("cutoff_len", 2048))
    max_steps = int(args.max_steps or train_cfg.get("max_steps", 1000))
    pref_bsz = int(train_cfg.get("preference_batch_size", 1))
    sft_bsz = int(train_cfg.get("sft_batch_size", 1))
    grad_acc = int(train_cfg.get("gradient_accumulation_steps", 8))
    logging_steps = int(train_cfg.get("logging_steps", 10))
    save_steps = int(train_cfg.get("save_steps", 200))
    output_dir = Path(train_cfg.get("output_dir", "/root/autodl-tmp/qwen-med-runs/general-med-mpo-lora"))
    beta = float(objective.get("beta", 0.1))
    dpo_weight = float(objective.get("dpo_weight", 1.0))
    sft_weight = float(objective.get("sft_weight", 0.25))
    safety_weight = float(objective.get("safety_reward_weight", 0.5))
    safety_rows = [row for row in pref_rows if is_safety_row(row)]

    started = time.perf_counter()
    last_metrics: dict[str, float | int] = {}
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, max_steps + 1):
        pref_batch = random.sample(pref_rows, min(pref_bsz, len(pref_rows)))
        sft_batch = random.sample(sft_rows, min(sft_bsz, len(sft_rows)))
        loss_pref = preference_loss(model, tokenizer, pref_batch, cutoff_len, beta, device)
        loss_sft = sft_loss(model, tokenizer, sft_batch, cutoff_len, device)
        if safety_rows:
            safety_batch = random.sample(safety_rows, min(pref_bsz, len(safety_rows)))
            loss_safety = preference_loss(model, tokenizer, safety_batch, cutoff_len, beta, device)
        else:
            loss_safety = loss_pref.new_tensor(0.0)
        loss = (dpo_weight * loss_pref + sft_weight * loss_sft + safety_weight * loss_safety) / grad_acc
        loss.backward()
        if step % grad_acc == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("max_grad_norm", 1.0)))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step % logging_steps == 0:
            elapsed = time.perf_counter() - started
            last_metrics = {
                "step": step,
                "loss_pref": float(loss_pref.detach().cpu()),
                "loss_sft": float(loss_sft.detach().cpu()),
                "loss_safety": float(loss_safety.detach().cpu()),
                "seconds": elapsed,
            }
            print(json.dumps(last_metrics, ensure_ascii=False))
        if step % save_steps == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"[mpo] saved checkpoint to {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    elapsed = time.perf_counter() - started
    train_results = {
        "max_steps": max_steps,
        "preference_rows": len(pref_rows),
        "sft_replay_rows": len(sft_rows),
        "safety_rows": len(safety_rows),
        "train_runtime": elapsed,
        "train_steps_per_second": max_steps / elapsed if elapsed > 0 else 0.0,
        "train_loss": last_metrics.get("loss_pref", 0.0),
        **last_metrics,
    }
    (output_dir / "train_results.json").write_text(json.dumps(train_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[mpo] finished and saved adapter to {output_dir}")


if __name__ == "__main__":
    main()
