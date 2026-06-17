# 面试问答备忘

## DPO 也是微调吗？

是。DPO 是一种偏好对齐阶段的微调，它不是单纯评测，也不是部署技巧。项目里的最终模型路线可以理解为：

```text
Base Qwen2.5-3B -> LoRA SFT -> DPO/MPO/GRPO 等对齐或诊断实验
```

其中 LoRA adapter 承载了参数更新，DPO/GRPO 是继续在 SFT 基础上做优化。

## 为什么早期 DPO/GRPO 没有涨点？

训练曲线显示 reward 或 reward accuracy 可以变好，但外部 CMB held-out 准确率不一定上升。主要原因是训练偏好数据与 CMB-test 评测分布不一致，模型学到了训练 reward，却没有迁移到最终选择题准确率。

因此项目保留了负结果，并追加 v12 同分布诊断实验验证该判断。

## v12 成功说明什么？

v12 使用 CMB 11200 题构造严格题面去重的 `8.2k/3k` 诊断切分，GRPO 最佳 checkpoint 将 3000 held-out 准确率从 72.27% 提升到 73.30%。这说明算法链路和 reward 并非完全无效，关键问题在于训练数据分布和评测分布需要对齐。

## 这个结果能刷榜吗？

不能直接这么说。v12 是内部诊断切分，不是官方 leaderboard 提交结果。它适合写成“同分布 held-out 诊断实验”，用于说明问题定位和算法验证。

## KV Cache 怎么体现？

部署侧用 `transformers generate(use_cache=True)` 做并发解码 benchmark，记录输出 token 数、耗时、decode tokens/s 和峰值显存。项目结果显示并发 8 时 decode 吞吐约 443 tokens/s，峰值显存约 5.87 GB。

## 红旗症状安全响应是否解决了？

没有完全解决。模型能显著提高急症召回，但在 routine 低风险场景中存在过度急诊化。项目把这个限制写入报告，并提供 rule-based guardrail 做部署侧修补。
