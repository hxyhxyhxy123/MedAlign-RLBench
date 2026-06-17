from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "n/a"


def pp(after: Any, before: Any) -> str:
    try:
        return f"{(float(after) - float(before)) * 100:+.2f} pp"
    except Exception:
        return "n/a"


def train_line(name: str, result_path: str) -> tuple[str, bool]:
    data = load_json(result_path)
    if not data:
        return f"- {name}: missing", False
    minutes = float(data.get("train_runtime", 0.0)) / 60.0
    loss = data.get("train_loss", "n/a")
    epoch = data.get("epoch", "n/a")
    return f"- {name}: done, epoch={epoch}, loss={loss}, runtime={minutes:.1f} min", True


def metric_line(name: str, metric_path: str, key: str = "accuracy") -> tuple[str, bool]:
    data = load_json(metric_path)
    if not data:
        return f"- {name}: missing", False
    return f"- {name}: {key}={pct(data.get(key, 0.0))}, total={data.get('total', 'n/a')}", True


def main() -> None:
    full = load_json("data/full/full_data_summary.json")
    hf = load_json("data/hf_processed/hf_processed_summary.json")
    extra = load_json("data/extra_processed/extra_processed_summary.json")
    integrity = load_json("data/metadata/data_integrity_report.json")

    train_items = [
        train_line("SFT LoRA general branch", "runs/remote_results/sft_train_results.json"),
        train_line("DPO LoRA general branch", "runs/remote_results/dpo_train_results.json"),
        train_line("Construct-SFT red-flag branch", "runs/remote_results/redflag_sft_train_results.json"),
        train_line("DPO red-flag branch", "runs/remote_results/redflag_dpo_train_results.json"),
        train_line("SFT QLoRA ablation", "runs/remote_results/qlora_train_results.json"),
        train_line("MPO mixed-objective branch", "runs/remote_results/mpo_train_results.json"),
    ]

    metric_items = [
        metric_line("CMB-val base", "runs/metrics/cmb_val_base.json"),
        metric_line("CMB-val SFT", "runs/metrics/cmb_val_sft.json"),
        metric_line("CMB-val DPO", "runs/metrics/cmb_val_dpo.json"),
        metric_line("CMB-val QLoRA", "runs/metrics/cmb_val_qlora.json"),
        metric_line("CMB-val MPO", "runs/metrics/cmb_val_mpo.json"),
        metric_line("C-Eval medical base", "runs/metrics/ceval_med_base.json"),
        metric_line("C-Eval medical SFT", "runs/metrics/ceval_med_sft.json"),
        metric_line("C-Eval medical DPO", "runs/metrics/ceval_med_dpo.json"),
    ]

    redflag_paths = [
        ("Red-flag base", "runs/metrics/redflag_aug_base.json"),
        ("Red-flag SFT", "runs/metrics/redflag_aug_sft.json"),
        ("Red-flag DPO", "runs/metrics/redflag_aug_dpo.json"),
        ("Red-flag MPO", "runs/metrics/redflag_aug_mpo.json"),
    ]
    redflag_lines = []
    redflag_done = []
    for name, path in redflag_paths:
        data = load_json(path)
        if not data:
            redflag_lines.append(f"- {name}: missing")
            redflag_done.append(False)
            continue
        redflag_lines.append(
            "- "
            + name
            + ": emergency="
            + pct(data.get("emergency_hit_rate", 0.0))
            + ", key_actions="
            + pct(data.get("key_action_coverage", 0.0))
            + ", avoid_actions="
            + pct(data.get("avoid_action_coverage", 0.0))
            + ", unsafe="
            + pct(data.get("unsafe_action_rate", 0.0))
            + ", json="
            + pct(data.get("json_format_rate", 0.0))
        )
        redflag_done.append(True)

    cmb_base = load_json("runs/metrics/cmb_val_base.json")
    cmb_sft = load_json("runs/metrics/cmb_val_sft.json")
    cmb_dpo = load_json("runs/metrics/cmb_val_dpo.json")
    ceval_base = load_json("runs/metrics/ceval_med_base.json")
    ceval_dpo = load_json("runs/metrics/ceval_med_dpo.json")
    kvcache = load_json("runs/kvcache_benchmark.json")

    completed = sum(done for _, done in train_items) + sum(done for _, done in metric_items) + sum(redflag_done)
    total = len(train_items) + len(metric_items) + len(redflag_done)
    missing = [
        "QLoRA training/eval" if not load_json("runs/remote_results/qlora_train_results.json") else "",
        "MPO training/eval" if not load_json("runs/remote_results/mpo_train_results.json") else "",
        "KV cache benchmark" if not kvcache else "",
        "C-Eval medical eval" if not ceval_dpo else "",
        "MMLU eval, if the resume keeps an MMLU claim" if not load_json("runs/metrics/mmlu_med_dpo.json") else "",
    ]
    missing = [x for x in missing if x]

    status = {
        "completion_items_done": completed,
        "completion_items_total": total,
        "missing": missing,
        "cmb_sft_delta_pp": pp(cmb_sft.get("accuracy"), cmb_base.get("accuracy")),
        "cmb_dpo_delta_pp": pp(cmb_dpo.get("accuracy"), cmb_base.get("accuracy")),
        "ceval_dpo_delta_pp": pp(ceval_dpo.get("accuracy"), ceval_base.get("accuracy")),
        "kvcache_done": bool(kvcache),
    }
    Path("runs").mkdir(exist_ok=True)
    Path("runs/experiment_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Resume Experiment Status",
        "",
        "## Data",
        "",
        f"- CMB/full SFT rows: {full.get('sft_train', 0)}",
        f"- HF medical SFT rows: {hf.get('sft_rows', 0)}",
        f"- Extra medical SFT rows: {extra.get('sft_rows', 0)}",
        f"- CMB/full DPO rows: {full.get('dpo_train', 0)}",
        f"- HF medical DPO rows: {hf.get('dpo_rows', 0)}",
        f"- Extra medical DPO rows: {extra.get('dpo_rows', 0)}",
        f"- Integrity problem datasets: {len(integrity.get('problem_datasets', []))}",
        "",
        "## Training",
        "",
        *[line for line, _ in train_items],
        "",
        "## Choice Evaluation",
        "",
        *[line for line, _ in metric_items],
        f"- CMB SFT vs base: {status['cmb_sft_delta_pp']}",
        f"- CMB DPO vs base: {status['cmb_dpo_delta_pp']}",
        f"- C-Eval DPO vs base: {status['ceval_dpo_delta_pp']}",
        "",
        "## Red-Flag Evaluation",
        "",
        *redflag_lines,
        "",
        "## Deployment",
        "",
        f"- KV cache benchmark: {'done' if kvcache else 'missing'}",
        "",
        "## Resume Safety Check",
        "",
        "- Supported now: SFT, LoRA, DPO, dual-branch red-flag alignment, automated choice/red-flag evaluation.",
        "- Must be completed before claiming: QLoRA, MPO, KV cache benchmark, and C-Eval/MMLU if those exact metrics remain in the resume.",
        "- Avoid claiming GLM-Embedding retrieval or Baichuan3-Turbo preference construction unless those data-generation logs are added.",
        "",
        "## Missing Items",
        "",
        *(f"- {item}" for item in missing),
    ]
    out = Path("reports/resume_experiment_status.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
