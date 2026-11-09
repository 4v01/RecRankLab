# 同类 MovieLens 评分项目核对与本轮优化依据

本轮继续检索 GitHub 的 MovieLens 评分预测项目，重点核对可读源代码，未将外部报告值当作本地复现。下面固定的是**调研快照**，不代表能确认旧实验生成时的代码提交。源文件哈希及引用见 source_manifest.json。

| 参考项目／固定快照 | 外部报告 | 已核对的协议与限制 | 对本项目的启示 |
|---|---|---|---|
| [NicolasHug/Surprise](https://github.com/NicolasHug/Surprise/tree/93e1c30f3b2d15bdd0f2c4c816fbfce59f000fad) | README 的 ML-1M：SVD RMSE0.873、SVD++0.862、Baseline0.909 | examples/benchmark.py 是五折 CV，非用户内时间划分；AlgoBase.predict 默认 clip=True，裁剪到评分区间 | 更强正则化的带偏置 MF 是应补齐的评分基线；当前未裁剪时间任务的0.8706不能直接与0.862作优劣判定。SVD++随后作为单独冻结的四候选参考阶段实现并运行，RMSE0.873099；优化器、批处理与协议不同，不声称复现其0.862 |
| [gtshs2/Autorec](https://github.com/gtshs2/Autorec/tree/bca30ff948b916a3748b3c10f11c00e2027df2ef) | README 声明 ML-1M RMSE0.8402 | data_preprocessor.py 全局随机90/10；main.py 默认2000轮、seed1000；所审查 AutoRec.py test_model 将 **test_R 本身送入网络**，再比较输出和test_R，并 clip到1–5。该任务含测试目标输入，不是本项目的未知评分预测 | 不能为了追随0.8402而复制其测试函数。本项目每次评分只从训练历史构造输入；验证仅用于选型，测试从不作为模型输入。此观察只针对所审查实现，不归因于AutoRec原论文 |

## 为什么从选型和训练入手

第二版对评分模型也统一按验证NDCG选型；这回答排序研究问题，却没有直接优化RMSE。第三版将评分/排序任务分表：评分最小化验证RMSE，排序最大化验证NDCG。两者仍使用原有数据划分和原有最终测试指标。

可验证的改动包括：验证早停、MF更强正则、AutoRec容量与权重衰减搜索、训练集均值/用户偏置/物品偏置残差重建。每条配置和预算都在ML-1M v3实验前提交；不承诺某个预设低分。普通与残差AutoRec使用匹配的四候选网格，分别选型后重训，因此结果是“含调参的流程对比”，并非纯单因素因果效应。

ItemKNN、RP3β、SLIM和Multi-VAE针对二值正例的排序任务，**其输出不能直接当作1–5分预测来计算有意义的评分RMSE**。本轮不把这四种模型列进评分精度排行榜；它们共享完整目录、历史屏蔽和同一个NDCG/Recall实现。数学方法与此前GitHub调研来源的关系见../methodology.md及../../external_benchmark/README.md。

主RMSE保持未裁剪。新增裁剪RMSE仅为预先声明的辅助诊断，绝不替代主指标、改变模型排序选型或隐藏未改善的结果。真实结果见../technical_report.md；外部报告值没有混入本地实验表。
