# 实验说明

## 主线

项目主线是 `Base -> SFT -> Preference/RL diagnostics`：

1. 使用 Qwen2.5-3B-Instruct 作为基座模型。
2. 基于医学问答和考试题数据构建 stage-1 SFT 数据，使用 LoRA/QLoRA 完成监督微调。
3. 构建偏好对并进行 DPO/MPO/IPO/ORPO 等对照实验。
4. 使用 CMB/C-Eval 风格选择题评测主任务准确率。
5. 对红旗症状安全响应单独构建 OOD 和 routine 混合风险评测。
6. 最后做 LoRA 合并和 KV Cache benchmark。

## 为什么早期 DPO/GRPO 没涨点

早期 v6/v8/v10/v11 偏好或 GRPO 实验使用了 CMExam-style 或 hard prompt 数据，但主要评测在 CMB-test noleak 上完成。训练 reward 能上升，并不代表生成式选择题准确率一定提升；当训练数据和评测分布不一致时，模型可能学到的是训练集偏好而不是 CMB held-out 上的正确选项。

GRPO v10 的最好 checkpoint 只有 71.00%，低于 SFT 的 72.27%，因此被报告为一次重要负结果。

## v12 诊断实验

v12 从 CMB-test full 11200 题中构造严格题面去重的 `8.2k/3k` 诊断切分：

- 3000 条固定 held-out 样本继续作为评测集。
- 剩余约 8200 条作为 GRPO prompt 训练集。
- 训练 reward 使用选择题可验证 Jaccard reward，要求预测选项集合与标准答案集合一致。
- 最佳 checkpoint-480 在 held-out 上达到 73.30%，超过 SFT 72.27%。

这说明 GRPO 在同分布可验证任务上能够带来收益，也解释了之前外部分布实验不涨点的原因。该结果只作为诊断实验，不作为官方 leaderboard 成绩。
