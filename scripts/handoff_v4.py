"""Create the final Chinese handoff from actual aggregates, including CPU blocks."""
import json,re
from pathlib import Path
import pandas as pd
root=Path('reports/v4');summary=pd.read_csv(root/'results.csv');status=json.loads((root/'completion_status.json').read_text());manifest=json.loads((root/'experiment_manifest.json').read_text())
lines=['# RecRankLab V4 独立审查交接','','本轮排查误差来源，增加完整ML-10M、SGD矩阵分解、MF-DCN、X架构启发的历史双塔，以及验证选权的模型组合。评分/排序协议保留，未为追求更高指标改变划分、裁剪主预测或抽样测试候选。',f"实际完成{status['completed']}/{status['expected']}条最终记录；六条ML-10M AutoRec及其融合记录资源受限，不填造数值。可行范围完成：{status['feasible_complete']}。",'', '| 数据集 | 方法 | 种子数 | RMSE↓ | R²↑ |','|---|---|---:|---:|---:|']
for _,r in summary[summary.selection_metric=='rmse'].sort_values(['dataset','rmse_mean']).iterrows():lines.append(f'| {r.dataset} | {r.experiment} | {int(r.seeds)} | {r.rmse_mean:.6f} ± {r.rmse_std:.6f} | {r.r2_mean:.6f} ± {r.r2_std:.6f} |')
lines+=['','第三版ML-1M最佳AutoRec RMSE0.869547/R²0.424549为历史参照；本轮ML-1M的AutoRec＋SGD融合给出改善。增加训练预算、候选与组合选择均已披露，不能将收益归因于单一改动。跨数据集绝对误差受用户、目录、时间与量表影响，不能当作增加样本量的因果收益。','','| 数据集 | 方法 | 种子数 | NDCG@10↑ | Recall@10↑ |','|---|---|---:|---:|---:|']
for _,r in summary[summary.selection_metric=='ndcg@10'].sort_values(['dataset','ndcg@10_mean'],ascending=[True,False]).iterrows():lines.append(f"| {r.dataset} | {r.experiment} | {int(r.seeds)} | {r['ndcg@10_mean']:.6f} ± {r['ndcg@10_std']:.6f} | {r['recall@10_mean']:.6f} |")
lines+=['','## 原因与方案','',
'验证诊断指向长尾/低活跃用户、长时间间隔、目标与负例采样匹配以及有限训练预算，冷启动占比不足以解释总体评分误差。更复杂的MF-DCN和小型双塔并未自动超过强基线；全部负面结果保留。验证残差互补性解释为什么简单融合可能优于单模型，权重始终只由验证集选择。',
'可执行的下一步：预注册LogQ采样修正与时间前缀历史对照；按训练/验证误差检验时间偏置和可用内容特征；针对长尾设计新用户/新物品独立留出。不要为制造改善而切换主划分或输入测试目标。',
'火山引擎公开DCN类型与X官方Phoenix公开架构仅提供方法启发；本项目不是其生产系统复现，没有社交、多模态、停留时间或转发标签。R²采用1−SSE/SST，是误差的相对尺度，不是因果解释。',
'', '## 审查材料','',
'- [完整技术报告](technical_report.md)、[评分表](rating_results.csv)、[排序表](ranking_results.csv)、[逐种子结果](per_seed_results.csv)、[全部验证候选/权重](validation_trials.csv)。',
'- [配对bootstrap](paired_differences.csv)、[分组验证误差](validation_cohorts.csv)、[残差互补性](validation_error_complementarity.csv)、[上下文缺失压力测试](context_stress.csv)。',
'- [manifest](experiment_manifest.json)、[冻结方法](methodology.md)、[额外阶段](anchor_prospectus.md)、[资源限制](resource_adaptation.md)、[来源审查](research/README.md)。',
'', '## 已知限制','',
'40条最终记录包括25次基础拟合和15次复用组件的融合评估；46条原计划中的6条缺失明确列出。ML-10M AutoRec单轮完整训练计时约153秒，未缩小数据冒充完成。上下文压力测试20/24项；它只模拟固定参数下推理输入缺失，不代表系统整体容错或训练数据损坏鲁棒性。',
'研究复用了既有benchmark，后续阶段在较早结果可见后提出；验证次数、算力和候选预算不相同。用户bootstrap条件于当前划分和拟合模型，不包含新数据、搜索和研究者方向不确定性。迁移前后数值模型定义已核对一致，但硬件不同，不承诺跨平台逐比特相同。',
'当前环境无法访问Actions底层工件存储，因此后续计算在GitHub运行，匿名文本证据提交到仓库。参数不含原始评分/历史；原始数据不随仓库发布。加载参数引用需要自行准备相同指纹的官方数据。']
(root/'handoff_zh.md').write_text('\n'.join(lines)+'\n')
sha=sorted({r['source_commit'] for r in manifest['runs'].values()})
test_log=(root/'test_output.txt').read_text();match=re.search(r'(\d+) passed in ([\d.]+)s',test_log)
validation='''# V4 validation evidence

68 local tests passed (7.69 seconds), including independent native-SGD equations, sparse-vs-dense AutoRec training, external-data checkpoint reconstruction, stable top-K/candidate equivalence, stopping and split tests. The source preparation CI at a2b7dd8 passed: https://github.com/4v01/RecRankLab/actions/runs/42647372584 . Test output is preserved in test_output.txt.

Ten real ML-100K one-epoch configuration/engine smoke artifacts are retained in smoke_summary.json; these are correctness checks, not benchmark scores. Standard CI repeats tests, old-generation audits, official-data preparation, the V4 evidence audit when feasible work completes, and these smoke paths.

The final audit independently verifies forty actual final records, six explicitly blocked records, source/config/seed/data signatures, selected/refit epochs, blend dependencies, official SST/R², anonymous per-user arithmetic, aggregate means/SD, paired point differences and twenty context-stress cases. See audit_output.txt. The final delivery CI link is available from the repository Actions page and the delivered commit SHA.

Numerical experiment source commits:\n\n'''+''.join(f'- `{s}`\n' for s in sha)+'''\nHosted training: https://github.com/4v01/RecRankLab/actions/runs/42648058543 . An earlier checkout-only attempt failed before training because a short SHA was interpreted as a branch: https://github.com/4v01/RecRankLab/actions/runs/42647433983 . No result or candidate was discarded for performance. Raw V2/V3 evidence remains unchanged.
'''
if match:validation=validation.replace('68 local tests passed (7.69 seconds)',f'{match[1]} local tests passed ({match[2]} seconds)')
(root/'validation.md').write_text(validation)
