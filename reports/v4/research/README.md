# 可迁移的公开方法与边界

- 火山引擎[预置模型说明](https://docs.volcengine.com/docs/IntelligentRecommendationPlatform/Presetmodel?lang=zh)列出DCN等模型类型；字节跳动[Monolith层目录](https://github.com/bytedance/monolith/blob/135c491a52b151772b976af989d8bc938c44d210/monolith/native_training/layers/README.md)也列出DCN。本轮独立实现原始[DCN公式](https://arxiv.org/abs/1708.05123)，加到MF评分分支中，以观察特征交互是否有增益。只用用户/物品ID，训练为评分MSE；不宣称是火山生产模型、DCN-V2复现或使用了平台API。
- 用户澄清“特斯拉算法”实际指X（原Twitter）。审查官方[xai-org/x-algorithm的Phoenix说明](https://github.com/xai-org/x-algorithm/blob/3b6919f87532e57063bb3f1e85c2f509323b513e/phoenix/README.md)。可以迁移的是历史条件用户塔、独立物品塔、相似度检索及多路融合。我们的HistoryTower用最近50个训练正例的均值池化和用户ID，学习小型两塔与批内对比损失；训练时从上下文删除当前正例，并屏蔽所有已知训练正例作为负例。它**不是Phoenix复现**：没有Transformer、语义ID、多模态、社交图或停留时间/转发标签，也没有上游权重。小型迁移实验检验架构思想，不移植X的业务权重。
- [矩阵分解参考](https://doi.org/10.1109/MC.2009.263)：补充逐观测SGD、偏置热启动和学习率衰减。原生C++内核为本项目独立编写，单线程顺序更新，记录源文件哈希、编译器与选项，避免Python逐评分循环导致的规模限制。
- 官方[ML-10M数据](https://grouplens.org/datasets/movielens/10m/)提供更大的独立benchmark；版本、量表、用户/电影集合与ML-1M不同。跨数据集结果不是“只增加样本”的因果对照。采用相同用户内时间80/10/10及完整目录评估，不用随机划分/采样候选抬高数字。

source_manifest.json记录固定源码快照与所读文件的SHA256。只参考公开架构，未复制第三方模型代码。没有将外部业务指标或论文数字当作本地实验结果。
