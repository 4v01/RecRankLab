import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recranklab.experiments import run as runner
from recranklab.utils.repro import config_load


def test_selection_refit_boundary_and_overwrite_guard(monkeypatch, tmp_path):
    # Validation favors candidate 1; test favors candidate 0. Selection must stay 1.
    def frame(items):
        return pd.DataFrame({'user':[0]*len(items),'item':items,'rating':[5]*len(items)})
    parts={'train':frame([0]),'val':frame([1]),'test':frame([2])}
    meta={'n_users':1,'n_items':3,'fingerprint':'synthetic'}
    monkeypatch.setattr(runner,'load',lambda *args:(parts,meta))
    monkeypatch.setattr(runner,'provenance',lambda **kwargs:{'git_commit':'test-source','git_dirty':False})
    fits=[]; evaluations=[]
    class Model:
        rating_model=False
        loss_history=[]
        def __init__(self,cfg): self.cfg=cfg
        def fit(self,data):
            fits.append((self.cfg['lr'],data.item.tolist())); return self
        def save(self,path): Path(path).write_bytes(b'fixture checkpoint')
        def parameter_count(self): return 0
    class Protocol:
        def __init__(self,history,target,*args):
            self.history,self.target=history,target
        def evaluate(self,model):
            evaluations.append((self.history.item.tolist(),self.target.item.tolist()))
            value=model.cfg['lr'] if self.target.item.tolist()==[1] else -model.cfg['lr']
            return {'ndcg@10':value},pd.DataFrame({'user':[0],'ndcg@10':[value]})
    monkeypatch.setattr(runner,'create',lambda name,nu,ni,cfg:Model(cfg))
    monkeypatch.setattr(runner,'RankingProtocol',Protocol)
    cfg=config_load('configs/base.yaml');cfg.update(output_root=str(tmp_path),trials=[{'lr':.1},{'lr':.2}])
    result=runner.run_one('bpr_mf',cfg,17)
    assert result['selected_trial']==1
    assert fits==[(.1,[0]),(.2,[0]),(.2,[0,1])]
    assert evaluations==[([0],[1]),([0],[1]),([0,1],[2])]
    runner.run_one('bpr_mf',cfg,17)
    assert len(fits)==3  # completed result is reused without a second test evaluation
    with pytest.raises(RuntimeError,match='overwrite'):
        runner.run_one('bpr_mf',{**cfg,'epochs':999},17)
