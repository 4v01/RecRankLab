# 第二版独立审查交付说明

实验源提交：`ab7798096d563205153f632d1029c59559cd10aa`。全部 33 次 ML-1M 最终实验均来自该干净提交，种子为 17、29、43。后续提交仅修正文档、报告生成与描述性元数据；原始运行记录没有改写。

| 模型 | NDCG@10 均值 | 种子标准差 | 每种子验证拟合次数 |
|---|---:|---:|---:|
| autorec | 0.027863 | 0.007931 | 2 |
| autorec_item | 0.046278 | 0.001501 | 4 |
| biasmf | 0.018118 | 0.000353 | 2 |
| bpr_mf | 0.084996 | 0.001543 | 2 |
| bpr_optimized | 0.086720 | 0.001162 | 4 |
| bpr_positives_only | 0.084239 | 0.000125 | 2 |
| ease | 0.101701 | 0.000000 | 4 |
| ease_selfopt | 0.101701 | 0.000000 | 4 |
| lightgcn | 0.088450 | 0.000410 | 2 |
| lightgcn_optimized | 0.091230 | 0.001111 | 4 |
| popularity | 0.058417 | 0.000000 | 1 |

EASE 相对原预算 LightGCN 的 NDCG@10 约高 14.98%，但算法与预算不同，不能当作等计算预算的因果结论。扩大预算 LightGCN 的均值为 0.091230，EASE 为 0.101701。物品侧 AutoRec 的 RMSE 从本次重跑用户侧版本的 0.942738 降至 0.870566，约下降 7.66%；其排序仍低于 Popularity。

EASE 固定网格与验证记忆搜索的结果完全相同：四次拟合中均选择 ridge=1000。当前消融没有证明自优化额外有效，不包装为 CogAlpha/XALPHA 复现。EASE 三种子是确定性复现核对，零标准差不等于零泛化不确定性。

验证：24 项单元/集成测试通过，六个真实 ML-100K 烟测通过，独立结果审计通过。远端 CI 状态见 validation.md；原始数据、划分数组和 checkpoint 均未提交。

主要局限：首版测试结果已被查看；这是复用 benchmark 的后续研究。用户内时间划分不是全局时间回测。改进版有更大的训练/搜索预算。需要独立数据或预先冻结的额外时间设置验证泛化；当前用户 bootstrap 区间只条件于本次划分和模型。

审查入口：technical_report.md、methodology.md、limitations.md、results.csv、paired_differences.csv、experiment_manifest.json 和 evidence/。复现时先检出运行记录中的实验源 SHA，再按 README 执行；审计须保留完整 Git 历史。
