# Results

本目录只保存可公开的指标摘要，不包含完整预测文件、模型权重或训练日志。

| 文件 | 内容 |
| --- | --- |
| `choice_qa_summary.csv` | CMB/C-Eval 医学选择题主结果 |
| `alignment_ablation_summary.csv` | SFT、DPO、MPO、GRPO 对照结果 |
| `grpo_v12_selection.json` | v12 GRPO checkpoint selection 摘要 |
| `redflag_mixed_summary.csv` | 红旗症状 mixed-risk 评测摘要 |
| `kvcache_benchmark.csv` | KV Cache benchmark 摘要 |

完整预测 JSONL、checkpoint、adapter、trainer state 和日志文件不建议放入 GitHub。如需归档，可单独上传到 Hugging Face Dataset/Model 或对象存储。
