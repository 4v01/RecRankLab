import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import ElasticNet

from recranklab.models.core import create, Recommender, retain_neighbors
from recranklab.evaluation.metrics import RankingProtocol, rating_metrics, rating_user_errors
from recranklab.experiments.training import EarlyStopping
from recranklab.experiments import run as runner
from recranklab.utils.repro import seed_all, config_load


@pytest.fixture
def ratings():
    return pd.DataFrame([(0,0,5),(0,1,1),(1,0,4),(1,2,5),(2,1,5),(2,2,4)],columns=['user','item','rating'])


def cfg():
    return dict(seed=17,epochs=2,lr=.001,reg=0.,hidden=4,latent=2,auto_batch_size=2,
                top_k=2,shrink=1.,alpha=1.,beta=.6,slim_workers=2,
                elastic_alpha=.01,l1_ratio=.5,slim_max_iter=2000,slim_tol=1e-6)


@pytest.mark.parametrize('name',['itemknn','rp3beta','slim','multivae'])
def test_new_models_fit_only_seed_checkpoint_and_shared_population(name,ratings,tmp_path):
    seed_all(17,1);m=create(name,3,4,cfg()).fit(ratings)
    expected=m.predict_scores([0,1,2])
    assert expected.shape==(3,4) and np.isfinite(expected).all()
    assert m.inputs[0,1]==0 and m.inputs[:,3].sum()==0
    m.save(tmp_path/'model.pt')
    np.testing.assert_array_equal(expected,Recommender.load(tmp_path/'model.pt').predict_scores([0,1,2]))
    seed_all(17,1);n=create(name,3,4,cfg()).fit(ratings.copy())
    np.testing.assert_array_equal(expected,n.predict_scores([0,1,2]))
    target=pd.DataFrame([(0,2,5),(1,1,4)],columns=ratings.columns)
    got,_=RankingProtocol(ratings,target,3,4).evaluate(m)
    assert got['eligible_users']==2
    if name!='multivae':
        assert np.all(np.diag(m.coefficients.numpy())==0)
        assert np.all(m.coefficients.numpy()>=0)
    if name=='slim':assert m.solver_diagnostics['nonconverged_columns']==0


def test_itemknn_cosine_shrink_exact(ratings):
    m=create('itemknn',3,4,{**cfg(),'top_k':4}).fit(ratings)
    x=m.inputs.numpy().astype(float);g=x.T@x;n=np.sqrt(g.diagonal());d=n[:,None]*n[None,:]+1.
    expected=g/d;np.fill_diagonal(expected,0.)
    np.testing.assert_allclose(m.coefficients.numpy(),expected,atol=1e-7)


def test_rp3beta_transition_and_popularity_exact(ratings):
    m=create('rp3beta',3,4,{**cfg(),'top_k':4,'normalize_similarity':False}).fit(ratings)
    x=m.inputs.numpy().astype(float);ud=x.sum(1).clip(min=1);it=x.sum(0).clip(min=1)
    expected=(x.T/it[:,None])@(x/ud[:,None]);expected/=it[None,:]**.6;np.fill_diagonal(expected,0.)
    np.testing.assert_allclose(m.coefficients.numpy(),expected,atol=1e-7)


def test_slim_matches_independent_target_regression(ratings):
    m=create('slim',3,4,cfg()).fit(ratings)
    x=m.inputs.numpy().astype(float)
    for j in range(3):
        y=x[:,j].copy();features=x.copy();features[:,j]=0
        solver=ElasticNet(alpha=.01,l1_ratio=.5,positive=True,fit_intercept=False,
                          tol=1e-6,max_iter=2000,selection='cyclic').fit(features,y)
        np.testing.assert_allclose(m.coefficients.numpy()[:,j],solver.coef_,atol=1e-5)


def test_stable_neighbors_zero_diagonal_and_ties():
    b=np.ones((4,4));out=retain_neighbors(b,1)
    assert np.all(np.diag(out)==0)
    np.testing.assert_array_equal(out[:,3],[1,0,0,0])


def test_multivae_loss_inputs_and_deterministic_posterior(ratings):
    seed_all(17,1);m=create('multivae',3,4,cfg());m.setup(ratings)
    m.eval();a=m(m.inputs)[0];b=m(m.inputs)[0]
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert m.inputs[0,1]==0 # low rating is not reconstructed as an implicit positive


def test_centered_autorec_preserves_observation_mask_and_train_only_baseline(ratings,tmp_path):
    c={**cfg(),'auto_variant':'item','center_ratings':True,'baseline_reg':10.}
    seed_all(17,1);m=create('autorec',3,4,c).fit(ratings)
    assert m.mask[1,0] and m.train_frame.rating.iloc[1]==1
    assert m.inputs[2,0]==0 and not m.mask[2,0]
    assert m.mean==ratings.rating.mean()
    m.save(tmp_path/'centered.pt')
    np.testing.assert_array_equal(m.predict_scores([0,1,2]),Recommender.load(tmp_path/'centered.pt').predict_scores([0,1,2]))
    stats=rating_user_errors(m,ratings);score=rating_metrics(m,ratings)
    assert np.sqrt(stats.sse.sum()/stats['count'].sum())==pytest.approx(score['rmse'],abs=1e-12)


@pytest.mark.parametrize('metric,values,best_epoch',[('rmse',[.9,.8,.85,.86],2),('ndcg@10',[.1,.2,.15,.14],2)])
def test_early_stop_direction_patience_and_best_epoch(metric,values,best_epoch):
    class Model:config={'epochs':10}
    scores=iter(values);s=EarlyStopping(lambda m:{metric:next(scores)},metric,interval=1,patience=2)
    flags=[s(Model(),e) for e in range(1,5)]
    assert flags==[False,False,False,True] and s.best['epoch']==best_epoch


def test_rating_selection_refits_best_epoch_without_test_feedback(monkeypatch,tmp_path):
    def frame(i):return pd.DataFrame({'user':[0],'item':[i],'rating':[5]})
    parts={'train':frame(0),'val':frame(1),'test':frame(2)}
    monkeypatch.setattr(runner,'load',lambda *a:(parts,{'n_users':1,'n_items':3,'fingerprint':'fixture'}))
    monkeypatch.setattr(runner,'provenance',lambda **kw:{'git_commit':'fixture','git_dirty':False})
    fits=[];test_calls=[]
    class Model:
        rating_model=True
        def __init__(self,c):self.config=c;self.loss_history=[];self.epoch=0
        def fit(self,data):
            fits.append((data.item.tolist(),self.config['epochs']))
            for e in range(1,self.config['epochs']+1):
                self.epoch=e;self.loss_history.append(float(e))
                if hasattr(self,'epoch_callback') and self.epoch_callback(self,e):break
            return self
        def save(self,path):Path(path).write_bytes(b'fixture')
        def parameter_count(self):return 0
    class Protocol:
        def __init__(self,history,target,*a):self.target=target
        def evaluate(self,model):
            assert self.target.item.tolist()==[2]
            saved=json.loads((tmp_path/'ml-1m/rating_fixture/17/selection.json').read_text())
            assert saved['config']['epochs']==2 and saved['selection_metric']=='rmse'
            test_calls.append(1)
            return {'ndcg@10':0.},pd.DataFrame({'user':[0],'ndcg@10':[0.]})
    monkeypatch.setattr(runner,'create',lambda n,u,i,c:Model(c))
    monkeypatch.setattr(runner,'RankingProtocol',Protocol)
    def rating(model,target):
        if target.item.tolist()==[1]:return {'rmse':{1:.9,2:.8,3:.85,4:.86}[model.epoch]}
        assert model.epoch==2
        return {'rmse':99.} # Test cannot influence epoch selection.
    monkeypatch.setattr(runner,'rating_metrics',rating)
    monkeypatch.setattr(runner,'rating_user_errors',lambda *a:pd.DataFrame({'user':[0],'count':[1],'sse':[9801.],'sae':[99.]}))
    c=config_load('configs/base.yaml');c.update(experiment='rating_fixture',output_root=str(tmp_path),
        epochs=10,early_stopping=True,eval_interval=1,patience=2,selection_metric='rmse',trials=[{'lr':.001}])
    result=runner.run_one('autorec',c,17)
    assert fits==[([0],10),([0,1],2)] and len(test_calls)==1
    assert result['validation_trials'][0]['stopped_epoch']==4
    runner.run_one('autorec',c,17);assert len(test_calls)==1


def test_failed_trial_consumes_budget_and_cannot_win(monkeypatch,tmp_path):
    def frame(i):return pd.DataFrame({'user':[0],'item':[i],'rating':[5]})
    parts={'train':frame(0),'val':frame(1),'test':frame(2)}
    monkeypatch.setattr(runner,'load',lambda *a:(parts,{'n_users':1,'n_items':3,'fingerprint':'fixture'}))
    monkeypatch.setattr(runner,'provenance',lambda **kw:{'git_commit':'fixture','git_dirty':False})
    fits=[]
    class Model:
        rating_model=False;loss_history=[]
        def __init__(self,c):self.config=c
        def fit(self,rows):
            fits.append((self.config['lr'],rows.item.tolist()))
            if self.config['lr']==0.:raise RuntimeError('solver did not converge')
            return self
        def save(self,path):Path(path).write_bytes(b'fixture')
        def parameter_count(self):return 0
    class Protocol:
        def __init__(self,*a):pass
        def evaluate(self,model):return {'ndcg@10':.1},pd.DataFrame({'user':[0],'ndcg@10':[.1]})
    monkeypatch.setattr(runner,'create',lambda n,u,i,c:Model(c));monkeypatch.setattr(runner,'RankingProtocol',Protocol)
    c=config_load('configs/base.yaml');c.update(output_root=str(tmp_path),trials=[{'lr':0.},{'lr':.001}],continue_failed_trials=True)
    r=runner.run_one('bpr_mf',c,17)
    assert r['selected_trial']==1 and len(r['validation_trials'])==2
    assert 'error' in r['validation_trials'][0] and 'metrics' not in r['validation_trials'][0]
    assert fits==[(0.,[0]),(.001,[0]),(.001,[0,1])]


def test_slim_parallel_worker_isolation(ratings):
    a=create('slim',3,4,{**cfg(),'slim_workers':1}).fit(ratings)
    b=create('slim',3,4,{**cfg(),'slim_workers':3}).fit(ratings)
    np.testing.assert_array_equal(a.coefficients.numpy(),b.coefficients.numpy())


def test_svdpp_history_formula_low_ratings_and_zero_history(ratings,tmp_path):
    c={**cfg(),'dim':3,'implicit_reg':.001,'reg':.01,'user_batch_size':2}
    seed_all(17,1);m=create('svdpp',4,4,c);m.setup(ratings)
    assert m.history_matrix[0,1]==1 # Exposure includes the observed low rating.
    with torch.no_grad():
        context=torch.sparse.mm(m.history_graph,m.implicit.weight)
        expected=(m.implicit.weight[0]+m.implicit.weight[1])/np.sqrt(2)
        torch.testing.assert_close(context[0],expected)
        assert torch.all(context[3]==0)
    m.fit(ratings);scores=m.predict_scores([0,1,2,3]);assert np.isfinite(scores).all()
    m.save(tmp_path/'svdpp.pt')
    np.testing.assert_array_equal(scores,Recommender.load(tmp_path/'svdpp.pt').predict_scores([0,1,2,3]))
    seed_all(17,1);n=create('svdpp',4,4,c).fit(ratings)
    np.testing.assert_array_equal(scores,n.predict_scores([0,1,2,3]))


def test_svdpp_zero_implicit_factors_reduces_to_biasmf_algebra(ratings):
    c={**cfg(),'dim':3};m=create('svdpp',3,4,c);m.setup(ratings)
    with torch.no_grad():
        m.implicit.weight.zero_()
        expected=m.mean+m.ub.weight+m.ib.weight.T+m.users.weight@m.items.weight.T
        expected[:,~m.item_support]=m.mean
        torch.testing.assert_close(m.all_scores(),expected)


def test_observed_rating_r2_perfect_mean_negative_and_constant():
    class Predictions:
        item_support=np.ones(3,dtype=bool);user_support=np.ones(1,dtype=bool)
        def __init__(self,values):self.values=np.asarray(values,dtype=float)
        def predict_scores(self,users,items):return self.values[np.asarray(items)][None,:]
    target=pd.DataFrame({'user':[0,0,0],'item':[0,1,2],'rating':[1.,3.,5.]})
    assert rating_metrics(Predictions([1,3,5]),target)['r2']==1.
    assert rating_metrics(Predictions([3,3,3]),target)['r2']==0.
    assert rating_metrics(Predictions([5,5,1]),target)['r2']==pytest.approx(-3.5)
    assert rating_metrics(Predictions([3,3,3]),target.assign(rating=3.))['r2'] is None
    assert rating_metrics(Predictions([1,3,5]),target.iloc[:1])['r2'] is None
