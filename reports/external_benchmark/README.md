# GitHub 项目 benchmark：数据、方法、结果与可比性

调研日期：2026-11-09（Asia/Shanghai）。第二版已经完成 33 次真实最终实验、本地验收并推送后，才开展本次调研。**下表是外部项目保存的输出或作者报告，不是我们重新运行的结果。没有把不同协议的数字拼成排行榜。** 引用固定到仓库快照；快照 SHA 只保证引用可回看，不证明这些历史数字由该 SHA 生成。无法确定的原始实验提交、种子方差和配置均标为未建立。

完整数据见 [reported_results.csv](reported_results.csv)（30 条外部数值记录）、[protocol_comparison.csv](protocol_comparison.csv) 和 [source_manifest.json](source_manifest.json)（完整快照 SHA、来源文件 SHA-256、不可变引用和许可证观察）。下一版方案见 [upgrade_plan.md](upgrade_plan.md)。本文没有修改第二版模型、配置或测试协议。

## 1. 可直接比较的本地第二版结果

来自 [真实结果表](../results.csv)，ML-1M、评分≥4、用户内时间 80/10/10、完整训练支持目录、屏蔽全部已评分历史、相同 5,820 位用户：

| 模型 | NDCG@10 均值±种子 SD | 验证拟合/种子 | 平均总实验秒/种子 |
|---|---:|---:|---:|
| BPR-MF 原预算 | 0.084996±0.001543 | 2 | 31.0 |
| BPR 扩大预算 | 0.086720±0.001162 | 4 | 336.3 |
| LightGCN 原预算 | 0.088450±0.000410 | 2 | 118.8 |
| LightGCN 扩大预算 | 0.091230±0.001111 | 4 | 389.0 |
| EASE 网格 | 0.101701±0.000000 | 4 | 34.0 |
| EASE 验证记忆搜索 | 0.101701±0.000000 | 4 | 31.7 |

这些是同一评估协议下的系统比较，**并非等计算量的因果消融**。神经网络改进版训练轮数、维度、批量与搜索预算均变化；EASE 是解析拟合。共享云 CPU 的耗时只作资源记录。EASE 的重复种子不提供独立随机不确定性。

## 2. 外部项目的数据与方法

| 来源 | 数据与处理 | 划分／候选 | 方法和可借鉴部分 | 与第二版的关键差异 |
|---|---|---|---|---|
| [S1 Recommenders](https://github.com/recommenders-team/recommenders/tree/cc47866b0599721f8544d1cc850a5a1db091acb3)（原 microsoft/recommenders） | 保存的 notebook 实际跑 ML-100K，不是 ML-1M | 按 item 分层的随机 75/25；算法分别准备训练、真值和推荐 | SAR、Cornac BPR/BiVAE、LightGCN；工程接口与计时 | notebook 自称示例而非完整 benchmark；BPR 真值过滤到≥4，其他路径不同，不能默认共用人口／标签 |
| [S2 RecBole](https://github.com/RUCAIBox/RecBole/tree/7b02be5ec80a88310f2d04a27a82adfcbb5dc211) | ML-1M_general：≥3、10-core、834,449 条、3,124 物品 | 默认用户内随机 80/10/10、full-sort；支持验证早停 | EASE、ItemKNN、SLIM、MultiVAE、LightGCN 的独立实现与配置 | 不是时间划分；低评分被丢弃。配置说明与计时表用户数分别写 6,040／6,034，不能未经复核视为同一精确数据版本 |
| [S3 官方 LightGCN](https://github.com/kuandeng/LightGCN/tree/b067ec05f16a24e0e1efe63cd82385f0c469af20) | Gowalla、Yelp2018、Amazon-book 的预处理正例 | 提供固定 train/test 文本；全目录、屏蔽 train | 三层归一化图传播、BPR、C++ 批量评估 | 数据不匹配；所审查训练循环用 test recall 触发 early stopping，不应移植这一选型流程 |
| [S4 原作者 VAE_CF](https://github.com/dawenl/vae_cf/tree/ad103c506c8897a853968e1b98c30f31cf5e30f6) | ML-20M；≥4、用户≥5 正例，过滤后 9,990,682 条、136,677 用户、20,720 物品 | 各留 10K 验证／测试用户；随机 80% fold-in／20% 预测；训练用户目录 | Multi-VAE／DAE 的 multinomial 重建与 KL annealing | 用户留出而非我们的同用户时间任务；低评分不参与屏蔽；Recall 分母为 min(K,真值数) |
| [S5 可复现性研究框架](https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation/tree/0fb6b7f5c396f8525316ed66cf9c9fdb03a5fa9b) | 此处只抽取 SpectralCF 的 ML-1M 子实验，约226K 个五星正例 | “ours” 随机按用户 80/20，再从 train 划 90/10 验证；其他章节协议另有不同 | ItemKNN、RP3β、SLIM、EASE、iALS 等强经典基线及调参审计 | 正例仅 rating==5；表32原划分与表33重建划分也不能互换 |
| [S6 DaisyRec 附录与代码快照](https://github.com/recsys-benchmark/DaisyRec-v2.0/tree/88b0a32faa1144033e969882781a8b0354958cf6) | 附录含 ML-1M Origin／5-filter／10-filter | 附录原始运行的精确划分、候选与配置未建立 | 多维评估设计值得参考；必须核对指标实现 | 快照 NDCG 用返回列表的命中向量排序计算 IDCG；与本项目不同，附录到该代码的原始运行关联也未知 |

S1、S2、S4、S5、S6 的主要引用文件及代码定位均列在 manifest 中。S2 的质量分数在所审查配置／计时文件中**未给出**，因此没有补造。S3 的 `test_flag=part` 不应误读为 sampled negatives：所审查评估矩阵覆盖完整模型目录并屏蔽 train。

## 3. 外部原始报告值：保留口径，不进行跨项目名次比较

| 来源／结果位置 | 原始模型与数值 | 为什么不能直接对照我们的 0.101701 |
|---|---|---|
| S1 `examples/06_benchmarks/movielens.ipynb`，cell22 保存输出 | ML-100K：BPR NDCG@10 **0.362301**；LightGCN **0.419947**；BiVAE **0.471722** | 数据集、随机划分、模型间真值处理、计算硬件与我们的设置不同；历史 notebook 输出不是本地重跑 |
| S3 README 三层示例日志 | Gowalla Recall@20 **0.18236**／NDCG@20 **0.15539**；Yelp **0.06367／0.05236**；Amazon-book **0.04123／0.03189** | 数据域与 K 不同；日志中的 best/stopping 使用 test recall；没有给出三种子方差 |
| S4 notebook cell75／92 保存输出 | ML-20M：Multi-VAE NDCG@100 **0.42592**、Recall@20 **0.39535**；DAE **0.41997／0.38768** | K、数据、用户留出任务和 Recall 分母不同。括号误差为用户标准误，不是跨种子 SD |
| S5 附表33，印刷页20:41 | ML-1M 五星正例：EASE Recall@20 **0.3085**；SLIM ElasticNet **0.3069**；ItemKNN cosine **0.2929**；MF BPR **0.2616** | 正例仅五星，随机划分、不同调参预算；是 Recall@20 而非 NDCG@10 |
| S5 附表32，印刷页20:40 | 另一份 ML-1M 原划分的 EASE Recall@20 **0.0956** | 与表33同时变化了划分与独立调参；说明协议和样本构成影响巨大，不能把差值解释为算法升级 |
| S6 附表10（隔离记录） | ML-1M Origin：MostPop NDCG@10 **0.4952**；BPRMF **0.5275**；Multi-VAE **0.6538** | 原实验提交／配置未建立，且所审查快照的 IDCG 定义不同；只收录为外部报告，禁止与第二版直接排名 |

S2 另提供硬件成本表：14 CPU／256GB／TITAN V 上，原表标记 EASE **1.65**、BPR **1.73**、LightGCN **4.10**、MultiVAE **0.04 秒/epoch**。EASE 解析 fit 与神经网络 epoch 的单位本身就不同，不能据此计算相对于本机的加速比。

## 4. 两个可以复算的指标反例

[metric_diagnostic.json](metric_diagnostic.json) 由本目录的 [verify.py](verify.py) 用纯算术生成，**不是模型性能结果**：

- 用户有 10 个真实正例，Top-10 只有第一名命中。本项目 IDCG 包括 min(10,10) 个理想正例，NDCG≈0.220092；对返回命中向量自身排序构造 IDCG，会得到 1.0。该差异可在 S6 快照 `daisy/utils/metrics.py` 的 `NDCG` 函数核对，但不据此断言所有版本或附录实验均由该快照运行。
- 用户有 40 个真实正例，Top-20 命中 10 个。本项目 Recall=10/40=0.25；S4 notebook 的 `min(K,n_truth)` 分母得到 10/20=0.5。名称相同，统计量不同。

## 5. 对下一版的直接含义

第二版没有证据表明继续增加固定轮数是优先方向；有证据表明，强物品协同基线比当前图模型具有更好的效果／成本表现。外部资料进一步支持把 **ItemKNN、RP3β、SLIM 与隐式 Multi-VAE** 纳入候选，但没有证明它们在我们的协议下一定更好。优先顺序、冻结预算、拒绝测试反馈的约束、资源限制和验收门槛见 [upgrade_plan.md](upgrade_plan.md)。

许可证方面：S1/S2 为 MIT，S4 为 Apache-2.0，S5 根目录为 AGPL-3.0（内含子项目还可能不同），S3 所审查树未发现根 LICENSE。这里只摘录事实与引用，没有复制外部实现进 MIT 项目。后续应独立实现数学方法或使用许可证核对后的适配器；S2 README 对学术用途另有说明，也需在实际分发前复核。
