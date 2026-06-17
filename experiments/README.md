# Experiments

这里保留关键实验入口脚本，方便复现实验路线。

| 文件 | 作用 |
| --- | --- |
| `run_choice_answer_dpo_v6.sh` | answer-only DPO 实验 |
| `run_choice_answer_checkpoint_sweep_v6.sh` | DPO checkpoint sweep |
| `run_rl_only_hard_exact_v8.sh` | hard exact preference DPO/IPO/ORPO 对照 |
| `run_rs_orpo_repair_v5.sh` | rejection sampling preference 对照 |
| `run_verifiable_grpo_v10.sh` | CMExam-style hard prompt GRPO 负结果 |
| `run_verifiable_grpo_v11.sh` | reward/data 结构修订后的 GRPO 对照 |
| `run_verifiable_grpo_v12.sh` | CMB 8.2k/3k disjoint split 诊断实验 |

这些脚本保留了远程训练时使用的路径变量。复现时需要根据自己的服务器路径修改 `BASE`、`SFT_ADAPTER`、`PROJECT_DIR` 等变量。
