"""Report real V4 evidence; immutable runs and older report generations are inputs."""
import gc,json,shutil
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from recranklab.data.prepare import digest,load
from recranklab.utils.repro import provenance,seed_all
from recranklab.experiments.study_v4 import load_study_v4

ROOT=Path('reports/v4')

def expected_runs(study):
    for dataset in study['datasets']:
        for name,spec in study['experiments'].items():
            for seed in spec.get('seeds',study['seeds']):yield dataset,name,seed,'base',spec
        for name,spec in study['hybrids'].items():
            for seed in study['seeds']:yield dataset,name,seed,'hybrid',spec


def collect(study):
    evidence=ROOT/'evidence';evidence.mkdir(exist_ok=True)
    runs={};rows=[];trials=[];checks=[];fit_diagnostics=[];missing=[]
    for dataset,name,seed,kind,spec in expected_runs(study):
        directory=Path('outputs/v4')/dataset/name/str(seed);path=directory/'result.json'
        if not path.exists():missing.append([dataset,name,seed]);continue
        r=json.loads(path.read_text());key=f'{dataset}_{name}_seed{seed}';copy=evidence/f'{key}.json';shutil.copyfile(path,copy)
        files={}
        for filename in ('per_user.csv','per_user_rating.csv'):
            p=directory/filename
            if p.exists():
                target=evidence/f'{key}_{filename}';shutil.copyfile(p,target)
                files[filename]={'path':str(target.relative_to(ROOT)),'sha256':digest(target)}
        runs[key]={'kind':kind,'evidence':str(copy.relative_to(ROOT)),'evidence_sha256':digest(copy),'source_commit':r['environment']['git_commit'],'signature':r['signature'],'dataset':r['data'],'files':files}
        score=r['test_metrics'];selection=spec.get('objective',r['config'].get('selection_metric'))
        row={'dataset':dataset,'experiment':name,'seed':seed,'kind':kind,'selection_metric':selection,'validation_objective_value':r['validation_trials'][r['selected_trial']]['metrics'][selection],'source_commit':r['environment']['git_commit'],**score,'total_seconds':r['total_seconds']}
        if kind=='base':
            row.update(selected_epoch=r['selected_config'].get('epochs'),selected_trial=r['selected_trial'],final_fit_seconds=r['final_fit_seconds'],peak_rss_mib=r['peak_rss_mib'])
        else:row.update(weight_first=r['weight_first'],selected_trial=r['selected_trial'])
        rows.append(row)
        for i,t in enumerate(r['validation_trials']):
            trials.append({'dataset':dataset,'experiment':name,'seed':seed,'trial':i,'selected':i==r['selected_trial'],'selection_metric':selection,'best_epoch':t.get('best_epoch'),'stopped_epoch':t.get('stopped_epoch'),'weight_first':t.get('weight_first'),'overrides':json.dumps(t.get('overrides',{}),sort_keys=True),**t.get('metrics',{}),'fit_seconds':t.get('fit_seconds')})
            history=t.get('checkpoint_history',[])
            if history:
                fit_diagnostics.append({'dataset':dataset,'experiment':name,'seed':seed,'trial':i,'selected_candidate':i==r['selected_trial'],'best_epoch':t['best_epoch'],'first_check_epoch':history[0]['epoch'],'stopped_epoch':t['stopped_epoch'],'max_epoch':r['config']['epochs'],'best_at_first_check':t['best_epoch']==history[0]['epoch'],'best_at_max_epoch':t['best_epoch']==r['config']['epochs'],'best_validation_objective':t['metrics'][selection],'last_validation_objective':history[-1]['metrics'][selection],'training_loss_at_best':t['loss_history'][t['best_epoch']-1],'last_training_loss':t['loss_history'][-1]})
            for c in t.get('checkpoint_history',[]):checks.append({'dataset':dataset,'experiment':name,'seed':seed,'trial':i,'epoch':c['epoch'],**c['metrics']})
    frame=pd.DataFrame(rows);frame.to_csv(ROOT/'per_seed_results.csv',index=False)
    pd.DataFrame(fit_diagnostics).to_csv(ROOT/'fit_diagnostics.csv',index=False)
    pd.DataFrame(trials).to_csv(ROOT/'validation_trials.csv',index=False);pd.DataFrame(checks).to_csv(ROOT/'validation_checkpoints.csv',index=False)
    numeric=[c for c in frame.select_dtypes(include=[np.number]).columns if c not in ('seed','selected_trial')]
    aggregate=[]
    for (dataset,name),g in frame.groupby(['dataset','experiment'],sort=True):
        row={'dataset':dataset,'experiment':name,'seeds':len(g),'kind':g.kind.iloc[0],'selection_metric':g.selection_metric.iloc[0]}
        for metric in numeric:
            vals=g[metric].dropna()
            if len(vals):row.update({metric+'_mean':float(vals.mean()),metric+'_std':float(vals.std(ddof=1)) if len(vals)>1 else 0.})
        aggregate.append(row)
    summary=pd.DataFrame(aggregate);summary.to_csv(ROOT/'results.csv',index=False)
    summary[summary.selection_metric=='rmse'].to_csv(ROOT/'rating_results.csv',index=False)
    summary[summary.selection_metric=='ndcg@10'].to_csv(ROOT/'ranking_results.csv',index=False)
    blocked_set={(b['dataset'],name,seed) for b in study['resource_blocks']['final_records'] for name in b['experiments'] for seed in b['seeds']}
    blocked=[x for x in missing if tuple(x) in blocked_set];pending=[x for x in missing if tuple(x) not in blocked_set]
    status={'complete':not missing,'feasible_complete':not pending,'completed':len(rows),'expected':len(list(expected_runs(study))),'resource_blocked':blocked,'pending':pending,'missing':missing}
    (ROOT/'completion_status.json').write_text(json.dumps(status,indent=2))
    return runs,summary,status


def uncertainty(runs):
    pairs=[('sgdmf_v4','bias_baseline_v4','rmse'),('mf_dcn_v4','sgdmf_v4','rmse'),('rating_blend_v4','sgdmf_v4','rmse'),('history_tower_v4','sparse_ease_v4','ndcg@10'),('ranking_blend_v4','sparse_ease_v4','ndcg@10'),('sparse_autorec_v4','sgdmf_v4','rmse'),('anchor_blend_v4','sparse_autorec_v4','rmse'),('anchor_blend_v4','historical_autorec_v3','rmse')]
    output=[]
    for dataset in ('ml-1m','ml-10m'):
        for first,second,metric in pairs:
            arrays=[]
            for seed in (17,29,43):
                a=f'{dataset}_{first}_seed{seed}';b=f'{dataset}_{second}_seed{17 if second in ("bias_baseline_v4","sparse_ease_v4") else seed}'
                if a not in runs or (b not in runs and not (dataset=='ml-1m' and second=='historical_autorec_v3')):continue
                filename='per_user_rating.csv' if metric=='rmse' else 'per_user.csv'
                x=pd.read_csv(ROOT/runs[a]['files'][filename]['path'])
                y=pd.read_csv(Path(f'reports/v3/evidence/autorec_rmse_v3_seed{seed}_rating_users.csv') if second=='historical_autorec_v3' else ROOT/runs[b]['files'][filename]['path'])
                assert x.evaluation_row.equals(y.evaluation_row)
                if metric=='rmse':
                    np.testing.assert_array_equal(x['count'],y['count']);arrays.append((x.sse.to_numpy(),y.sse.to_numpy(),x['count'].to_numpy()))
                else:arrays.append((x[metric]-y[metric]).to_numpy())
            if len(arrays)!=3:continue
            rng=np.random.default_rng(2026)
            if metric=='rmse':
                actual=float(np.mean([np.sqrt(a.sum()/n.sum())-np.sqrt(b.sum()/n.sum()) for a,b,n in arrays]))
                samples=[]
                for _ in range(2000):
                    index=rng.integers(len(arrays[0][0]),size=len(arrays[0][0]))
                    samples.append(np.mean([np.sqrt(a[index].sum()/n[index].sum())-np.sqrt(b[index].sum()/n[index].sum()) for a,b,n in arrays]))
            else:
                difference=np.mean(arrays,axis=0);actual=float(difference.mean())
                samples=[difference[rng.integers(len(difference),size=len(difference))].mean() for _ in range(2000)]
            lo,hi=np.quantile(samples,[.025,.975]);output.append({'dataset':dataset,'comparison':first+' - '+second,'metric':metric,'mean_difference':actual,'ci_low':float(lo),'ci_high':float(hi),'bootstrap_replicates':2000,'bootstrap_seed':2026,'unit':'user, conditional on split and fitted models'})
    pd.DataFrame(output).to_csv(ROOT/'paired_differences.csv',index=False)


def diagnostics(runs):
    from recranklab.models.core import Recommender
    data_refs={};random_refs=[];cohorts=[];branch=[];diversity=[]
    prior_dependencies=ROOT/'diagnostic_model_dependencies.json'
    dependencies=json.loads(prior_dependencies.read_text()) if prior_dependencies.exists() else {}
    diagnostic_environment=provenance(require_clean=False)
    def merge_diagnostics(filename,records,keys):
        path=ROOT/filename;new=pd.DataFrame(records)
        if len(new):
            new['diagnostic_source_commit']=diagnostic_environment['git_commit'];new['diagnostic_script_sha256']=digest(__file__)
        if path.exists():
            old=pd.read_csv(path)
            if len(new):
                new_keys=set(map(tuple,new[keys].to_numpy()))
                old=old[[tuple(v) not in new_keys for v in old[keys].to_numpy()]]
            new=pd.concat([old,new],ignore_index=True)
        new.to_csv(path,index=False)
    for dataset in ('ml-1m','ml-10m'):
        parts,data=load(dataset);train,val,test=parts['train'],parts['val'],parts['test']
        data_refs[dataset]={'manifest':data,'partitions':{}}
        for stage in ('val','test'):
            y=parts[stage].rating.to_numpy(dtype=float)
            data_refs[dataset]['partitions'][stage]={'count':len(y),'mean':float(y.mean()),'sst':float(np.square(y-y.mean()).sum()),'variance':float(y.var())}
        catalog=np.sort(train.item.unique());p=val[(val.rating>=4)&val.item.isin(catalog)].groupby('user').size()
        count=train.groupby('user').size();n=len(catalog)-count.reindex(p.index).to_numpy()
        truth=p.to_numpy();discount=1/np.log2(np.arange(20)+2)
        for k in (10,20):
            prefix=np.r_[0.,np.cumsum(discount)]
            expected_ndcg=np.mean((truth/n)*prefix[np.minimum(k,n)]/prefix[np.minimum(k,truth)])
            random_refs.append({'dataset':dataset,'stage':'validation','k':k,'expected_random_recall':float(np.mean(np.minimum(k,n)/n)),'expected_random_ndcg':float(expected_ndcg),'eligible_users':len(p)})
        uc=val.user.map(train.user.value_counts()).fillna(0).to_numpy();ic=val.item.map(train.item.value_counts()).fillna(0).to_numpy()
        gap=(val.timestamp.to_numpy()-val.user.map(train.groupby('user').timestamp.max()).to_numpy())/86400.
        masks={'user_1_to_49':uc<50,'user_50_to_199':(uc>=50)&(uc<200),'user_200plus':uc>=200,'item_cold':ic==0,'item_1_to_19':(ic>=1)&(ic<20),'item_20_to_99':(ic>=20)&(ic<100),'item_100plus':ic>=100,'gap_0_to_7d':gap<=7,'gap_7_to_30d':(gap>7)&(gap<=30),'gap_30_to_180d':(gap>30)&(gap<=180),'gap_180plus_d':gap>180}
        residuals={}
        for name in ('sgdmf_v4','mf_dcn_v4','sparse_autorec_v4'):
            key=f'{dataset}_{name}_seed17'
            path=Path('outputs/v4_validation_components')/dataset/name/'17'/'model.pt'
            if key not in runs or not path.exists():continue
            meta=path.with_name('metadata.json');metadata=json.loads(meta.read_text());assert digest(path)==metadata['checkpoint_sha256']
            dependencies[str(path)]={'checkpoint_sha256':digest(path),'metadata':metadata,'diagnostic_environment':diagnostic_environment,'diagnostic_script_sha256':digest(__file__)}
            model=Recommender.load(path);prediction=model.predict_pairs(val.user.to_numpy(),val.item.to_numpy());truth_values=val.rating.to_numpy(dtype=float)
            residuals[name]=prediction-truth_values
            for label,mask in masks.items():
                if not mask.any():continue
                error=prediction[mask]-truth_values[mask];sst=np.square(truth_values[mask]-truth_values[mask].mean()).sum()
                cohorts.append({'dataset':dataset,'experiment':name,'seed':17,'partition':'validation','group':label,'count':int(mask.sum()),'rmse':float(np.sqrt(np.square(error).mean())),'mae':float(np.abs(error).mean()),'r2':float(1-np.square(error).sum()/sst) if sst>0 else None})
            if name=='mf_dcn_v4':
                import torch
                ordinary=float(np.sqrt(np.square(prediction-truth_values).mean()))
                with torch.no_grad():model.head.weight.zero_();model.head.bias.zero_()
                reduced=model.predict_pairs(val.user.to_numpy(),val.item.to_numpy())
                branch.append({'dataset':dataset,'seed':17,'partition':'validation','diagnostic':'posthoc_cross_and_deep_head_suppression_no_retraining','full_rmse':ordinary,'suppressed_rmse':float(np.sqrt(np.square(reduced-truth_values).mean())),'causal_retraining_ablation':False})
            del model,prediction;gc.collect()
        for first,second in [('sparse_autorec_v4','sgdmf_v4'),('sgdmf_v4','mf_dcn_v4')]:
            if first not in residuals or second not in residuals:continue
            a,b=residuals[first],residuals[second]
            diversity.append({'dataset':dataset,'first':first,'second':second,'seed':17,'partition':'validation','count':len(a),'first_rmse':float(np.sqrt(np.square(a).mean())),'second_rmse':float(np.sqrt(np.square(b).mean())),'residual_correlation':float(np.corrcoef(a,b)[0,1]),'mean_residual_product':float(np.mean(a*b)),'half_blend_rmse':float(np.sqrt(np.square((a+b)/2).mean())),'first_bias':float(a.mean()),'second_bias':float(b.mean())})
        del parts,train,val,test,residuals;gc.collect()
    (ROOT/'rating_reference.json').write_text(json.dumps(data_refs,indent=2))
    (ROOT/'diagnostic_model_dependencies.json').write_text(json.dumps(dependencies,indent=2))
    pd.DataFrame(random_refs).to_csv(ROOT/'random_reference.csv',index=False)
    merge_diagnostics('validation_cohorts.csv',cohorts,['dataset','experiment','seed','partition','group'])
    merge_diagnostics('dcn_branch_diagnostic.csv',branch,['dataset','seed','diagnostic'])
    merge_diagnostics('validation_error_complementarity.csv',diversity,['dataset','first','second','seed'])


def report(summary,status):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(16,8))
    for row,dataset in enumerate(('ml-1m','ml-10m')):
        for col,(metric,objective) in enumerate([('rmse','rmse'),('r2','rmse'),('ndcg@10','ndcg@10')]):
            ax=axes[row,col];g=summary[(summary.dataset==dataset)&(summary.selection_metric==objective)].sort_values(metric+'_mean')
            ax.errorbar(g[metric+'_mean'],np.arange(len(g)),xerr=g[metric+'_std'],fmt='o',capsize=3)
            ax.set_yticks(np.arange(len(g)),g.experiment.str.replace('_v4',''));ax.set_title(dataset+' '+metric);ax.grid(axis='x',alpha=.3)
    fig.tight_layout();(ROOT/'figures').mkdir(exist_ok=True);fig.savefig(ROOT/'figures/results.png',dpi=150);plt.close(fig)
    diagnosis=json.loads((ROOT/'diagnosis.json').read_text())
    lines=['# RecRankLab V4：原因排查与跨数据集实验','','本轮为同一benchmark反复探索后的研究扩展；保留既有测试协议。所有数字来自实际完成记录，外部方法不等于上游生产模型复现。',f"原计划全部完成：{status['complete']}；本次可行范围完成：{status['feasible_complete']}；实际{status['completed']}/{status['expected']}条最终记录（基础拟合＋复用组件的融合评估），资源受限{len(status['resource_blocked'])}条，待完成{len(status['pending'])}条。缺失项不填数。",'','## 原因排查','',
        '1. **任务难度与指标方向**：RMSE越低越好，NDCG/R²越高越好。完整目录排序、屏蔽全部历史的结果不能与随机采样少量候选的文献数字直接比较。random_reference.csv给出同协议验证集随机排序期望，不冒充实测模型。',
        '2. **冷启动不是总体评分误差的主要解释**：验证集冷物品评分占比ML-1M约0.017%、ML-10M约0.007%。长尾与活跃度异质性另见validation_cohorts.csv，不能仅凭总体均值定位瓶颈。',
        '3. **训练和模型受限**：历史部分最佳轮次命中训练上限，另有MF候选在首次检查轮次即最优；这提示预算/检查频率需要诊断，并不能证明全部模型都是欠拟合。V4补充逐观测SGD、偏置热启动和更细评分验证；SGD与旧Adam差异包含多个因素，不是优化器单因素结论。',
        '4. **缺少线上特征和行为标签**：MovieLens是选择性评分，不是曝光/点击日志。X的停留、多行为、社交和多模态标签不可凭空补齐；本轮只测试小型历史双塔。MF-DCN只有ID特征，不等于工业特征体系。',
        '5. **规模瓶颈**：旧实现会缓存整张评分矩阵和所有用户候选，ML-10M仅这两部分约需2.78＋5.56GiB，超过当前8GiB预算。新实现按需评分、稀疏历史、分批评估、精确稳定top-K和紧凑checkpoint，使完整数据训练可行。它解决可运行性，不保证指标提高。',
        '', '## 数据与适用边界','','| 数据集 | 评分数 | 用户 | 有评分电影 | 密度 | 训练偏置基线验证RMSE |','|---|---:|---:|---:|---:|---:|']
    for dataset,v in diagnosis['datasets'].items():
        m=v['manifest'];lines.append(f"| {dataset} | {m['n_ratings']} | {m['n_users']} | {m['n_items']} | {v['density']:.2%} | {v['validation_baseline_rmse']:.6f} |")
    lines+=['','ML-10M AutoRec完整预算及其融合共6条记录资源受限，未当作已完成；完整训练计时证据与执行迁移说明见resource_adaptation.md和train_only_timing_pilot.json。其他ML-10M方法均用完整数据。两个数据集来自不同人群、目录、时间和评分量表；跨数据集绝对数字的差异不能归因为增加样本。ML-10M主评分量表0.5–5；主指标始终未裁剪。','', '## 真实评分结果','','三种子均值±样本SD；确定性BiasBaseline仅seed17，不将重复拟合作为独立样本。基础模型按验证RMSE选型，融合额外搜索五个冻结权重。','','| 数据集 | 方法 | 种子数 | RMSE↓ | R²↑ | MAE↓ |','|---|---|---:|---:|---:|---:|']
    for _,r in summary[summary.selection_metric=='rmse'].sort_values(['dataset','rmse_mean']).iterrows():lines.append(f"| {r.dataset} | {r.experiment} | {r.seeds} | {r.rmse_mean:.6f} ± {r.rmse_std:.6f} | {r.r2_mean:.6f} ± {r.r2_std:.6f} | {r.mae_mean:.6f} |")
    lines+=['','ML-1M第三版最佳AutoRec RMSE0.869547/R²0.424549为历史描述性参照；选型网格、方法与预算不同。R²=1−SSE/SST，不是因果或机制解释；同一测试集上它是平方误差的另一尺度。','','## 真实排序结果','','完整目录，正例≥4，屏蔽全部历史；确定性SparseEASE仅seed17。','','| 数据集 | 方法 | 种子数 | NDCG@10↑ | Recall@10↑ | NDCG@20↑ |','|---|---|---:|---:|---:|---:|']
    for _,r in summary[summary.selection_metric=='ndcg@10'].sort_values(['dataset','ndcg@10_mean'],ascending=[True,False]).iterrows():lines.append(f"| {r.dataset} | {r.experiment} | {r.seeds} | {r['ndcg@10_mean']:.6f} ± {r['ndcg@10_std']:.6f} | {r['recall@10_mean']:.6f} | {r['ndcg@20_mean']:.6f} |")
    lines+=['','## 组合、诊断与不确定性','',
        'validation_error_complementarity.csv量化训练单折模型在验证集上的残差相关性与固定半权重融合误差，用于解释互补性，不以测试残差确定权重。validation_trials.csv包含全部候选、轮次与融合权重。端点权重表示退回单模型，也是结果；不因未改善删去某种方法。paired_differences.csv提供同一数据集内用户配对bootstrap；区间条件于给定划分和模型，不包含新数据、搜索和研究者方向不确定性。',
        'fit_diagnostics.csv保留每个候选的训练损失、首个检查/上限命中和验证退化，训练损失在不同目标间不可直接横比。validation_cohorts.csv按训练用户活跃度、物品支持度、时间间隔报告seed17验证表现。dcn_branch_diagnostic.csv是在选定验证模型上临时置零交互输出头的诊断，未重训，不能当作严格架构消融或测试结果。',
        '', '## 复现、局限和下一步','',
        'context_stress.csv是固定参数的推理上下文缺失压力测试：0/20/50/80%四档、seed17模型、扰动seed2026，保留原目录与全历史屏蔽。数据可用范围20/24项；ML-10M AutoRec的4项不可用。它不等同于训练数据损坏、新用户冷启动或系统整体容错；详见robustness_prospectus.md。',
        '研究来源见research/README.md；主阶段方法与预算见methodology.md和configs/v4/study.yaml；额外AutoRec对照与融合阶段见anchor_prospectus.md和configs/v4/anchor_study.yaml；manifest绑定每条结果的代码SHA、完整配置、数据版本、种子和组件依赖。旧报告完全保留。评分模型仅做真实观测评分评估，未将排序logit当评分。',
        '候选预算、验证次数、最大轮次和损失不同；融合还使用额外验证选权与组件重训预算。后续阶段发生在已有版本测试结果已知后，属于探索，不宣称未触碰测试集的确认性结论。官方数据/模型checkpoint不随仓库分发；可按对应源SHA完整重训。',
        '历史双塔的批内负例分布非均匀；原始对比目标可能学习偏好相对采样频率的比值，而非直接的相关性。这是需进一步检验的目标匹配假设，不能从当前结果断言为唯一原因。下一轮应预注册LogQ采样修正、严格时间前缀上下文与更密集的早期验证，配合相同预算对照。下一步也应由验证诊断决定：对真正命中预算边界的模型预先扩展预算；对时间间隔差异检验时间偏置；对长尾优先引入可用元数据与严格冷启动留出；组合只用训练折外或验证预测。新的确认性改进应在预先冻结的新留出集/外部数据上验证。不能为提高数字切换主测试划分、采样候选或输入测试目标。']
    (ROOT/'technical_report.md').write_text('\n'.join(lines)+'\n')


def main():
    seed_all(2026,2)
    study=load_study_v4()
    existing=ROOT/'completion_status.json'
    if existing.exists() and json.loads(existing.read_text()).get('feasible_complete'):
        blocked={(b['dataset'],name,seed) for b in study['resource_blocks']['final_records'] for name in b['experiments'] for seed in b['seeds']}
        missing=[(d,n,s) for d,n,s,_,_ in expected_runs(study) if (d,n,s) not in blocked and not (Path('outputs/v4')/d/n/str(s)/'result.json').exists()]
        if missing:raise RuntimeError('Refusing to replace a completed delivered report with incomplete local outputs; audit published evidence or restore the matching source outputs first.')
    runs,summary,status=collect(study)
    uncertainty(runs);diagnostics(runs);report(summary,status)
    smoke=[]
    for path in sorted([*Path('outputs/v4_smoke/ml-100k').glob('*/*/result.json'),*Path('outputs/v4_engine_smoke/ml-100k').glob('*/*/result.json')]):
        r=json.loads(path.read_text());smoke.append({'experiment':r['experiment'],'source_commit':r['environment']['git_commit'],'signature':r['signature'],'metrics':r['test_metrics']})
    if not smoke and (ROOT/'smoke_summary.json').exists():smoke=json.loads((ROOT/'smoke_summary.json').read_text())
    (ROOT/'smoke_summary.json').write_text(json.dumps(smoke,indent=2))
    artifacts={str(p.relative_to(ROOT)):{'sha256':digest(p)} for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='experiment_manifest.json'}
    historical={str(p):digest(p) for p in [Path('reports/experiment_manifest.json'),Path('reports/results.csv'),Path('reports/v3/experiment_manifest.json'),Path('reports/v3/results.csv')]}
    manifest={'schema_version':4,'generated_at_utc':datetime.now(timezone.utc).isoformat(),'report_environment':provenance(require_clean=False),'study':study,'status':status,'runs':runs,'artifacts':artifacts,'immutable_historical':historical}
    (ROOT/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(summary[['dataset','experiment','seeds','rmse_mean','r2_mean','ndcg@10_mean']].to_string(index=False));print(status)

if __name__=='__main__':main()
