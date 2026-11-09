import ctypes
import zipfile
import numpy as np
import pandas as pd
import pytest
import torch
from recranklab.models.core import create,Recommender
from recranklab.models.scalable import native_sgd
from recranklab.data.prepare import split_rows,split_rows_vectorized,prepare,load
from recranklab.evaluation.metrics import RankingProtocol,rating_metrics,rating_user_errors
from recranklab.utils.repro import seed_all


def frame():
    return pd.DataFrame({'user':[0,0,0,1,1,1,2,2,2], 'item':[0,1,2,1,2,3,0,3,4],
        'rating':[5.,2.,4.,4.,5.,3.,5.,1.,4.], 'timestamp':[8,3,4,3,5,9,8,3,6]})


def test_vectorized_split_matches_original_with_ties_and_short_users():
    f=pd.DataFrame([(u,i,4.,i//3,u*19+i) for u in range(8) for i in range(3+u*2)],columns=['user','item','rating','timestamp','row_id'])
    f=f.sample(frac=1,random_state=7)
    a,b=split_rows(f),split_rows_vectorized(f)
    for key in a:np.testing.assert_array_equal(a[key],b[key])
    with pytest.raises(ValueError,match='Repeated'):split_rows_vectorized(pd.concat([f,f]))
    with pytest.raises(ValueError,match='three'):split_rows_vectorized(f[f.user==0].iloc[:2])


def test_official_10m_layout_and_half_star_range(tmp_path):
    raw=tmp_path/'raw';raw.mkdir()
    with zipfile.ZipFile(raw/'ml-10m.zip','w') as archive:
        archive.writestr('ml-10M100K/ratings.dat',''.join(f'{u}::{i}::{.5 if i==1 else 4.5}::{i}\n' for u in (1,2) for i in range(1,11)))
    m=prepare('ml-10m',tmp_path);parts,_=load('ml-10m',tmp_path)
    assert m['n_ratings']==20 and len(parts['train'])==16
    assert parts['train'].rating.min()==.5


def test_native_sgd_one_step_matches_equations():
    kernel,_=native_sgd();u=np.array([0],dtype=np.int64);i=u.copy();order=u.copy();r=np.array([4.],dtype=np.float32)
    p=np.array([[.2,.3]],dtype=np.float32);q=np.array([[.4,-.1]],dtype=np.float32);bu=np.array([.1],dtype=np.float32);bi=np.array([-.2],dtype=np.float32)
    oldp,oldq=p.copy(),q.copy();e=4-(3+.1-.2+(oldp*oldq).sum())
    pi=ctypes.POINTER(ctypes.c_int64);pf=ctypes.POINTER(ctypes.c_float)
    loss=kernel(u.ctypes.data_as(pi),i.ctypes.data_as(pi),r.ctypes.data_as(pf),order.ctypes.data_as(pi),1,2,3.,p.ctypes.data_as(pf),q.ctypes.data_as(pf),bu.ctypes.data_as(pf),bi.ctypes.data_as(pf),.01,.02,.03)
    np.testing.assert_allclose(p,oldp+.01*(e*oldq-.02*oldp),atol=1e-7)
    np.testing.assert_allclose(q,oldq+.01*(e*oldp-.02*oldq),atol=1e-7)
    assert loss==pytest.approx(float(e*e),rel=1e-6)
    assert bu[0]==pytest.approx(.1+.01*(e-.03*.1))


@pytest.mark.parametrize('name',['sgdmf','mf_dcn','bias_baseline','sparse_ease','history_tower'])
def test_scalable_models_deterministic_checkpoint_and_finite(name,tmp_path):
    cfg=dict(seed=17,epochs=2,dim=4,lr=.005,reg=.02,rating_batch_size=4,user_batch_size=2,history_length=2)
    seed_all(17,1);model=create(name,4,6,cfg).fit(frame());expected=model.predict_scores([0,1,2],[1,3,5])
    assert model.cache is None or isinstance(model.cache,tuple)
    assert expected.shape==(3,3) and np.isfinite(expected).all()
    model.save(tmp_path/'m.pt');loaded=Recommender.load(tmp_path/'m.pt')
    np.testing.assert_allclose(expected,loaded.predict_scores([0,1,2],[1,3,5]),atol=1e-7)
    seed_all(17,1);second=create(name,4,6,cfg).fit(frame())
    np.testing.assert_array_equal(expected,second.predict_scores([0,1,2],[1,3,5]))
    if hasattr(model,'predict_pairs'):
        np.testing.assert_allclose(model.predict_pairs([0,1,2],[1,3,5]),np.diag(expected),atol=1e-6)
        stats=rating_user_errors(model,frame());metric=rating_metrics(model,frame())
        assert metric['rmse']==pytest.approx(np.sqrt(stats.sse.sum()/stats['count'].sum()),abs=1e-12)


def test_sparse_ease_matches_original_and_lazy_catalog():
    cfg=dict(ridge=30.)
    old=create('ease',4,6,cfg).fit(frame());new=create('sparse_ease',4,6,cfg).fit(frame())
    np.testing.assert_allclose(old.predict_scores([0,1,2]),new.predict_scores([0,1,2]),atol=1e-7)
    target=pd.DataFrame({'user':[0,1],'item':[3,4],'rating':[5.,5.]})
    protocol=RankingProtocol(frame(),target,4,6)
    assert not isinstance(protocol.candidates,dict)
    np.testing.assert_array_equal(protocol.candidates[0],[3,4])
    a,_=protocol.evaluate(old);b,_=protocol.evaluate(new);assert a==b


def test_history_excludes_sampled_target_and_masks_known_positives():
    model=create('history_tower',4,6,dict(dim=4,history_length=50));model.setup(frame())
    u=torch.tensor([0,1]);target=torch.tensor([2,1])
    original=model.encode_users(u,exclude=target)
    ids=model.history_ids.clone();model.history_ids[0,model.history_ids[0]==2]=6;model.history_ids[1,model.history_ids[1]==1]=6
    np.testing.assert_allclose(original.detach().numpy(),model.encode_users(u).detach().numpy(),atol=1e-7)
    model.history_ids=ids
    # User0 likes item1? No, rating2; user1 likes item2, so only lower-left is masked.
    np.testing.assert_array_equal(model.negative_mask(u,target),[[False,False],[True,False]])
    assert set(model.history_ids[0].tolist())=={0,2,6}


def test_partition_top_k_and_batched_evaluation_preserve_stable_protocol():
    from recranklab.evaluation.metrics import stable_top_indices
    rng=np.random.default_rng(8)
    for n in [0,3,20,21,1000]:
        scores=rng.integers(0,5,n).astype(float)
        for k in (1,10,20):
            np.testing.assert_array_equal(stable_top_indices(scores,k),np.argsort(-scores,kind='stable')[:k])
    target=pd.DataFrame({'user':[0,1,2],'item':[3,4,2],'rating':[5.,5.,4.]})
    model=create('sparse_ease',4,6,{'ridge':30.}).fit(frame())
    protocol=RankingProtocol(frame(),target,4,6)
    a,ua=protocol.evaluate(model,batch_size=1);b,ub=protocol.evaluate(model,batch_size=2)
    assert a==b;pd.testing.assert_frame_equal(ua,ub)


def test_hybrid_endpoints_and_candidate_isolation():
    from recranklab.experiments.hybrid import RatingBlend,RankingBlend
    cfg=dict(seed=17,epochs=1,dim=4,lr=.005,reg=.02)
    seed_all(17,1);first=create('sgdmf',4,6,cfg).fit(frame());second=create('mf_dcn',4,6,cfg).fit(frame())
    for weight,base in [(0.,second),(1.,first)]:
        blend=RatingBlend(first,second,weight,{'rating_min':.5})
        np.testing.assert_allclose(blend.predict_pairs([0,1],[3,4]),base.predict_pairs([0,1],[3,4]))
    ease=create('sparse_ease',4,6,{'ridge':30.}).fit(frame())
    tower=create('history_tower',4,6,{**cfg,'user_batch_size':3}).fit(frame())
    rank=RankingBlend(ease,tower,.25,frame())
    full=rank.predict_scores([0,1],rank.catalog)
    np.testing.assert_allclose(rank.predict_scores([0,1],[1,3]),full[:,[1,3]],atol=1e-7)


def test_half_star_auxiliary_clip_has_dataset_bound():
    class Predictions:
        config={'rating_min':.5};item_support=np.array([True,True]);user_support=np.array([True])
        def predict_pairs(self,u,i):return np.array([.5,5.])
    target=pd.DataFrame({'user':[0,0],'item':[0,1],'rating':[.5,5.]})
    assert rating_metrics(Predictions(),target)['rmse_clipped_aux']==0.
