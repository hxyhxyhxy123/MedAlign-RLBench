from __future__ import annotations

import json
from pathlib import Path


def load_json(path: str):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def pct(data: dict, key: str) -> str:
    if key not in data:
        return "n/a"
    return f"{float(data[key]) * 100:.2f}%"


def runtime_minutes(data: dict) -> str:
    if "train_runtime" not in data:
        return "n/a"
    return f"{float(data['train_runtime']) / 60:.1f} min"


def main() -> None:
    hf = load_json("data/hf_processed/hf_processed_summary.json")
    extra = load_json("data/extra_processed/extra_processed_summary.json")
    full = load_json("data/full/full_data_summary.json")
    integrity = load_json("data/metadata/data_integrity_report.json")
    eval_summary = load_json("data/eval/eval_summary.json")
    sft_train = load_json("runs/remote_results/sft_train_results.json")
    dpo_train = load_json("runs/remote_results/dpo_train_results.json")
    redflag_sft = load_json("runs/remote_results/redflag_sft_train_results.json")
    redflag_dpo = load_json("runs/remote_results/redflag_dpo_train_results.json")
    qlora_train = load_json("runs/remote_results/qlora_train_results.json")
    mpo_train = load_json("runs/remote_results/mpo_train_results.json")
    cmb_base = load_json("runs/metrics/cmb_val_base.json")
    cmb_sft = load_json("runs/metrics/cmb_val_sft.json")
    cmb_dpo = load_json("runs/metrics/cmb_val_dpo.json")
    cmb_qlora = load_json("runs/metrics/cmb_val_qlora.json")
    cmb_mpo = load_json("runs/metrics/cmb_val_mpo.json")
    ceval_base = load_json("runs/metrics/ceval_med_base.json")
    ceval_sft = load_json("runs/metrics/ceval_med_sft.json")
    ceval_dpo = load_json("runs/metrics/ceval_med_dpo.json")
    red_base = load_json("runs/metrics/redflag_aug_base.json")
    red_sft = load_json("runs/metrics/redflag_aug_sft.json")
    red_dpo = load_json("runs/metrics/redflag_aug_dpo.json")
    red_mpo = load_json("runs/metrics/redflag_aug_mpo.json")
    kvcache = load_json("runs/kvcache_benchmark.json")
    lines = [
        "# Qwen Med Align Auto Report",
        "",
        "## Data Scale",
        "",
        f"- CMB full SFT rows: {full.get('sft_train', 0)}",
        f"- HF medical SFT rows: {hf.get('sft_rows', 0)}",
        f"- Extra medical SFT rows: {extra.get('sft_rows', 0)}",
        f"- CMB full DPO rows: {full.get('dpo_train', 0)}",
        f"- HF medical DPO rows: {hf.get('dpo_rows', 0)}",
        f"- Extra medical DPO rows: {extra.get('dpo_rows', 0)}",
        f"- Medical pretrain/retrieval rows: {hf.get('pretrain_rows', 0) + extra.get('pretrain_rows', 0)}",
        "",
        "## Evaluation Sets",
        "",
        f"- CMB-test choice: {eval_summary.get('cmb_test_choice_eval', 0)}",
        f"- CMB-Clin: {eval_summary.get('cmb_clin_eval', 0)}",
        f"- Red-flag: {eval_summary.get('redflag_eval', 0)}",
        f"- CMExam: {extra.get('cmexam_eval_rows', 0)}",
        "",
        "## Integrity",
        "",
        f"- Registered datasets: {integrity.get('dataset_count', 0)}",
        f"- Problem datasets: {len(integrity.get('problem_datasets', []))}",
        f"- Missing config refs: {len(integrity.get('missing_config_refs', []))}",
        "",
        "## Training Results",
        "",
        f"- SFT LoRA: runtime {runtime_minutes(sft_train)}, loss {sft_train.get('train_loss', 'n/a')}",
        f"- DPO LoRA: runtime {runtime_minutes(dpo_train)}, loss {dpo_train.get('train_loss', 'n/a')}",
        f"- Red-flag Construct-SFT: runtime {runtime_minutes(redflag_sft)}, loss {redflag_sft.get('train_loss', 'n/a')}",
        f"- Red-flag DPO: runtime {runtime_minutes(redflag_dpo)}, loss {redflag_dpo.get('train_loss', 'n/a')}",
        f"- QLoRA ablation: runtime {runtime_minutes(qlora_train)}, loss {qlora_train.get('train_loss', 'n/a')}",
        f"- MPO mixed objective: runtime {runtime_minutes(mpo_train)}, loss {mpo_train.get('train_loss', 'n/a')}",
        "",
        "## Metrics",
        "",
        f"- CMB-val base/SFT/DPO: {pct(cmb_base, 'accuracy')} / {pct(cmb_sft, 'accuracy')} / {pct(cmb_dpo, 'accuracy')}",
        f"- CMB-val QLoRA/MPO: {pct(cmb_qlora, 'accuracy')} / {pct(cmb_mpo, 'accuracy')}",
        f"- C-Eval medical base/SFT/DPO: {pct(ceval_base, 'accuracy')} / {pct(ceval_sft, 'accuracy')} / {pct(ceval_dpo, 'accuracy')}",
        f"- Red-flag key action coverage base/SFT/DPO/MPO: {pct(red_base, 'key_action_coverage')} / {pct(red_sft, 'key_action_coverage')} / {pct(red_dpo, 'key_action_coverage')} / {pct(red_mpo, 'key_action_coverage')}",
        f"- Red-flag unsafe action rate base/SFT/DPO/MPO: {pct(red_base, 'unsafe_action_rate')} / {pct(red_sft, 'unsafe_action_rate')} / {pct(red_dpo, 'unsafe_action_rate')} / {pct(red_mpo, 'unsafe_action_rate')}",
        "",
        "## Deployment",
        "",
        f"- KV cache benchmark: {'done' if kvcache else 'missing'}",
        f"- KV cache engine: {kvcache.get('engine', 'n/a') if kvcache else 'n/a'}",
    ]
    out = Path("reports/project_report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
