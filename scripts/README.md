# Scripts

脚本按功能可以分为：

| 类型 | 代表脚本 |
| --- | --- |
| 环境检查 | `check_gpu_env.py`, `check_project_ready.py`, `storage_guard.sh` |
| 数据处理 | `build_seed_data.py`, `build_stage1_data.py`, `convert_hf_medical_datasets.py` |
| 偏好数据 | `build_answer_only_preference.py`, `build_hard_exact_preference.py`, `build_rs_preference_from_samples.py` |
| GRPO | `build_grpo_indist_prompts.py`, `train_choice_grpo_lora.py` |
| 评测 | `generate_eval_predictions.py`, `generate_eval_predictions_sharded.py`, `eval_choice_predictions_robust.py` |
| 安全响应 | `build_redflag_ood_eval.py`, `eval_redflag_strict.py`, `redflag_safety_guard.py` |
| 部署测试 | `merge_lora.py`, `bench_kvcache.py` |

远程 SSH 调试脚本、临时 runner 和缓存文件没有放入发布版。
