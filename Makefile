PYTHON ?= python
REDFLAG_OOD_EVAL ?= data/eval/redflag_ood_eval.jsonl
REDFLAG_OOD_PRED ?= runs/predictions/redflag_ood_redflag_dpo.jsonl
REDFLAG_OOD_NAME ?= redflag_dpo
REDFLAG_TRAIN ?= data/llamafactory/stage1_redflag_sft_aug.jsonl data/llamafactory/stage1_redflag_dpo_aug.jsonl

.PHONY: bootstrap-nogpu clone-baselines inspect-data seed-data stage1-data ready gpu-setup gpu-check sft-lora sft-qlora dpo mpo grpo-v12 eval-choice build-redflag-ood eval-redflag-ood-strict guard-redflag-ood bench-kvcache

bootstrap-nogpu:
	bash scripts/bootstrap_nogpu.sh

clone-baselines:
	bash scripts/clone_baselines.sh

inspect-data:
	$(PYTHON) scripts/inspect_datasets.py --manifest configs/data_manifest.yaml --max-rows 8

seed-data:
	$(PYTHON) scripts/build_seed_data.py --manifest configs/data_manifest.yaml

stage1-data:
	$(PYTHON) scripts/build_stage1_data.py --out data/stage1 --sft-size 30000 --dpo-size 25000

ready:
	$(PYTHON) scripts/check_project_ready.py

gpu-setup:
	bash scripts/setup_gpu_env.sh

gpu-check:
	$(PYTHON) scripts/check_gpu_env.py

sft-lora:
	cd baselines/LLaMA-Factory && llamafactory-cli train ../../configs/llamafactory/qwen25_3b_sft_lora.yaml

sft-qlora:
	cd baselines/LLaMA-Factory && llamafactory-cli train ../../configs/llamafactory/qwen25_3b_sft_qlora.yaml

dpo:
	cd baselines/LLaMA-Factory && llamafactory-cli train ../../configs/llamafactory/qwen25_3b_dpo_lora.yaml

mpo:
	$(PYTHON) scripts/train_mpo.py --config configs/mpo_recipe.yaml

grpo-v12:
	bash experiments/run_verifiable_grpo_v12.sh

eval-choice:
	$(PYTHON) scripts/generate_eval_predictions.py --eval data/eval/cmb_val_choice_eval.jsonl --task choice --adapter /root/autodl-tmp/qwen-med-runs/general-med-dpo-lora --out runs/predictions/cmb_val_choice.jsonl --limit 280

build-redflag-ood:
	$(PYTHON) scripts/build_redflag_ood_eval.py --out $(REDFLAG_OOD_EVAL) --summary data/metadata/redflag_ood_eval_summary.json

eval-redflag-ood-strict:
	mkdir -p runs/metrics/strict runs/metrics/strict_details
	$(PYTHON) scripts/eval_redflag_strict.py --predictions $(REDFLAG_OOD_PRED) --eval $(REDFLAG_OOD_EVAL) --train $(REDFLAG_TRAIN) --out runs/metrics/strict/redflag_ood_$(REDFLAG_OOD_NAME)_strict.json --details runs/metrics/strict_details/redflag_ood_$(REDFLAG_OOD_NAME)_details.json

guard-redflag-ood:
	mkdir -p runs/predictions/guarded runs/metrics/strict_guarded runs/metrics/strict_guarded_details
	$(PYTHON) scripts/redflag_safety_guard.py --predictions $(REDFLAG_OOD_PRED) --out runs/predictions/guarded/redflag_ood_$(REDFLAG_OOD_NAME)_guarded.jsonl --report runs/metrics/redflag_ood_$(REDFLAG_OOD_NAME)_guard_report.json
	$(PYTHON) scripts/eval_redflag_strict.py --predictions runs/predictions/guarded/redflag_ood_$(REDFLAG_OOD_NAME)_guarded.jsonl --eval $(REDFLAG_OOD_EVAL) --train $(REDFLAG_TRAIN) --out runs/metrics/strict_guarded/redflag_ood_$(REDFLAG_OOD_NAME)_guarded_strict.json --details runs/metrics/strict_guarded_details/redflag_ood_$(REDFLAG_OOD_NAME)_guarded_details.json

bench-kvcache:
	$(PYTHON) scripts/bench_kvcache.py --config configs/deploy_kvcache.yaml
