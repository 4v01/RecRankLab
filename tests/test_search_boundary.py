"""Integration guard: adaptive proposals and selection cannot follow test feedback."""
import json
from pathlib import Path
import pandas as pd
from recranklab.experiments import run as runner
from recranklab.utils.repro import config_load


def test_adaptive_search_refit_and_test_boundary(monkeypatch, tmp_path):
    def frame(item):
        return pd.DataFrame({'user':[0],'item':[item],'rating':[5]})
    parts={'train':frame(0),'val':frame(1),'test':frame(2)}
    monkeypatch.setattr(runner,'load',lambda *args:(parts,{'n_users':1,'n_items':3,'fingerprint':'fixture'}))
    monkeypatch.setattr(runner,'provenance',lambda **kwargs:{'git_commit':'fixture','git_dirty':False})
    fits=[];test_calls=[]
    class Model:
        rating_model=False
        loss_history=[]
        def __init__(self,cfg):self.cfg=cfg
        def fit(self,rows):fits.append((self.cfg['ridge'],rows.item.tolist()));return self
        def parameter_count(self):return 0
        def save(self,path):Path(path).write_bytes(b'fixture')
    class Protocol:
        def __init__(self,history,target,*args):self.target=target.item.tolist()
        def evaluate(self,model):
            ridge=round(model.cfg['ridge'],4)
            if self.target==[1]:value={100.:.2,1000.:.1,33.3333:.05,300.:.3}[ridge]
            else:
                # Test prefers another candidate; this must not affect the search.
                test_calls.append(ridge);value=1. if ridge==1000. else 0.
                saved=json.loads((tmp_path/'ml-1m/ease_selfopt/17/selection.json').read_text())
                assert saved['config']['ridge']==300.
            return {'ndcg@10':value},pd.DataFrame({'user':[0],'ndcg@10':[value]})
    monkeypatch.setattr(runner,'create',lambda name,nu,ni,cfg:Model(cfg))
    monkeypatch.setattr(runner,'RankingProtocol',Protocol)
    cfg=config_load('configs/ease_selfopt.yaml');cfg['output_root']=str(tmp_path)
    result=runner.run_one('ease',cfg,17)
    assert [r['overrides']['ridge'] for r in result['validation_trials']]==[100.,1000.,33.33333333,300.]
    assert result['selected_config']['ridge']==300.
    assert all(rows==[0] for _,rows in fits[:-1]) and fits[-1]==(300.,[0,1])
    assert test_calls==[300.]
    runner.run_one('ease',cfg,17)
    assert test_calls==[300.]
