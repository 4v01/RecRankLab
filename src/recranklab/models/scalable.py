"""Original bounded-memory models for the separately frozen V4 study."""
import ctypes
import hashlib
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from scipy import linalg, sparse
import torch
from torch import nn
from torch.nn import functional as F

from .core import Recommender, BiasMF


def fit_biases(frame,nu,ni,reg=10.,iterations=10):
    u=frame.user.to_numpy();i=frame.item.to_numpy();mean=float(frame.rating.mean())
    residual=frame.rating.to_numpy(dtype=float)-mean
    bu,bi=np.zeros(nu),np.zeros(ni)
    uc=np.bincount(u,minlength=nu);ic=np.bincount(i,minlength=ni)
    for _ in range(iterations):
        bu=np.bincount(u,weights=residual-bi[i],minlength=nu)/(uc+reg)
        bi=np.bincount(i,weights=residual-bu[u],minlength=ni)/(ic+reg)
    return bu.astype('float32'),bi.astype('float32')


class BiasBaseline(Recommender):
    rating_model=True
    def __init__(self,nu,ni,config):
        super().__init__(nu,ni,config)
        self.register_buffer('bu',torch.zeros(nu));self.register_buffer('bi',torch.zeros(ni))
    def fit(self,frame):
        self.setup(frame)
        bu,bi=fit_biases(frame,self.n_users,self.n_items,self.config.get('bias_shrink',10.))
        self.bu.copy_(torch.from_numpy(bu));self.bi.copy_(torch.from_numpy(bi))
        return self
    @torch.no_grad()
    def predict_scores(self,user_ids,item_ids=None):
        u=np.atleast_1d(user_ids).astype(int)
        i=np.arange(self.n_items) if item_ids is None else np.atleast_1d(item_ids).astype(int)
        return self.mean+self.bu.numpy()[u,None]+self.bi.numpy()[None,i]


class StreamingMF(BiasMF):
    """Same BiasMF equation and cold fallback, only requested Cartesian scores."""
    @torch.no_grad()
    def predict_pairs(self,user_ids,item_ids):
        u=np.asarray(user_ids);i=np.asarray(item_ids);out=np.empty(len(u),dtype=np.float64)
        for start in range(0,len(u),32768):
            uu=torch.as_tensor(u[start:start+32768],dtype=torch.long);ii=torch.as_tensor(i[start:start+32768],dtype=torch.long)
            value=self.mean+self.ub(uu).flatten()+self.ib(ii).flatten()+(self.users(uu)*self.items(ii)).sum(1)
            value[~torch.as_tensor(self.user_support)[uu]|~torch.as_tensor(self.item_support)[ii]]=self.mean
            out[start:start+len(uu)]=value.numpy()
        return out

    @torch.no_grad()
    def predict_scores(self,user_ids,item_ids=None):
        u=torch.as_tensor(np.atleast_1d(user_ids),dtype=torch.long)
        i=torch.arange(self.n_items) if item_ids is None else torch.as_tensor(np.atleast_1d(item_ids),dtype=torch.long)
        value=self.mean+self.ub(u)+self.ib(i).T+self.users(u)@self.items(i).T
        value[~torch.as_tensor(self.user_support)[u],:]=self.mean
        value[:,~torch.as_tensor(self.item_support)[i]]=self.mean
        return value.numpy()


def native_sgd():
    source=Path(__file__).with_name('sgd_kernel.cpp')
    flags=['-O3','-std=c++17','-shared','-fPIC','-ffp-contract=off']
    compiler=subprocess.check_output(['c++','--version'],text=True).splitlines()[0]
    identity=hashlib.sha256(source.read_bytes()+str(flags).encode()+compiler.encode()).hexdigest()
    root=Path(tempfile.gettempdir())/'recranklab-native'/identity;root.mkdir(parents=True,exist_ok=True)
    library=root/'sgd.so'
    if not library.exists():
        target=root/'sgd.partial.so'
        subprocess.run(['c++',*flags,str(source),'-o',str(target)],check=True,capture_output=True)
        target.replace(library)
    lib=ctypes.CDLL(str(library));kernel=lib.sgd_epoch
    pi=ctypes.POINTER(ctypes.c_int64);pf=ctypes.POINTER(ctypes.c_float)
    kernel.argtypes=[pi,pi,pf,pi,ctypes.c_int64,ctypes.c_int,ctypes.c_float,pf,pf,pf,pf,ctypes.c_float,ctypes.c_float,ctypes.c_float]
    kernel.restype=ctypes.c_double
    return kernel,{'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'compiler':compiler,'flags':flags,'execution':'single-thread sequential SGD'}


class SGDMF(StreamingMF):
    """Per-observation, warm-bias SGD with old factors used in both updates."""
    def fit(self,frame):
        self.setup(frame);kernel,self.solver_diagnostics=native_sgd()
        bu,bi=fit_biases(frame,self.n_users,self.n_items,self.config.get('bias_shrink',10.))
        with torch.no_grad():
            self.ub.weight[:,0].copy_(torch.from_numpy(bu));self.ib.weight[:,0].copy_(torch.from_numpy(bi))
        u=np.ascontiguousarray(frame.user,dtype=np.int64);i=np.ascontiguousarray(frame.item,dtype=np.int64)
        r=np.ascontiguousarray(frame.rating,dtype=np.float32)
        p,q=self.users.weight.detach().numpy(),self.items.weight.detach().numpy()
        ub,ib=self.ub.weight.detach().numpy(),self.ib.weight.detach().numpy()
        pi=ctypes.POINTER(ctypes.c_int64);pf=ctypes.POINTER(ctypes.c_float)
        rng=np.random.default_rng(self.config['seed'])
        for epoch in range(self.config['epochs']):
            order=rng.permutation(len(frame)).astype(np.int64,copy=False)
            lr=self.config['lr']*self.config.get('lr_decay',.98)**epoch
            mse=kernel(u.ctypes.data_as(pi),i.ctypes.data_as(pi),r.ctypes.data_as(pf),order.ctypes.data_as(pi),len(frame),p.shape[1],self.mean,
                p.ctypes.data_as(pf),q.ctypes.data_as(pf),ub.ctypes.data_as(pf),ib.ctypes.data_as(pf),lr,self.config['reg'],self.config.get('bias_reg',.02))
            if not np.isfinite(mse):raise ValueError('Nonfinite SGD loss')
            self.loss_history.append(float(mse))
            if self.finish_epoch():break
        return self


class MFDCN(StreamingMF):
    """MF + original vector-cross DCN + parallel MLP, adapted to rating MSE.

    Only user/item IDs; no claims to replicate a proprietary platform model.
    """
    def __init__(self,nu,ni,config):
        super().__init__(nu,ni,config)
        width=2*config.get('dim',32)
        self.cross_weights=nn.ParameterList([nn.Parameter(torch.randn(width)*.01) for _ in range(config.get('cross_layers',2))])
        self.cross_bias=nn.ParameterList([nn.Parameter(torch.zeros(width)) for _ in self.cross_weights])
        self.deep=nn.Sequential(nn.Linear(width,64),nn.ReLU(),nn.Dropout(config.get('dropout',.1)),nn.Linear(64,32),nn.ReLU())
        self.head=nn.Linear(width+32,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def pair_scores(self,u,i):
        ue,ie=self.users(u),self.items(i);x0=torch.cat([ue,ie],dim=-1);cross=x0
        for weight,bias in zip(self.cross_weights,self.cross_bias):
            cross=x0*(cross@weight).unsqueeze(-1)+bias+cross
        return self.mean+self.ub(u).flatten()+self.ib(i).flatten()+(ue*ie).sum(-1)+self.head(torch.cat([cross,self.deep(x0)],dim=-1)).flatten()
    @torch.no_grad()
    def predict_pairs(self,user_ids,item_ids):
        self.eval();u=np.asarray(user_ids);i=np.asarray(item_ids);out=np.empty(len(u),dtype=np.float64)
        for start in range(0,len(u),8192):
            uu=torch.as_tensor(u[start:start+8192],dtype=torch.long);ii=torch.as_tensor(i[start:start+8192],dtype=torch.long)
            value=self.pair_scores(uu,ii)
            value[~torch.as_tensor(self.user_support)[uu]|~torch.as_tensor(self.item_support)[ii]]=self.mean
            out[start:start+len(uu)]=value.numpy()
        return out
    def fit(self,frame):
        self.setup(frame)
        bu,bi=fit_biases(frame,self.n_users,self.n_items,self.config.get('bias_shrink',10.))
        with torch.no_grad():
            self.ub.weight[:,0].copy_(torch.from_numpy(bu));self.ib.weight[:,0].copy_(torch.from_numpy(bi))
        u=torch.tensor(frame.user.to_numpy(),dtype=torch.long);i=torch.tensor(frame.item.to_numpy(),dtype=torch.long)
        r=torch.tensor(frame.rating.to_numpy(),dtype=torch.float32)
        opt=torch.optim.Adam(self.parameters(),lr=self.config['lr'],weight_decay=self.config.get('weight_decay',1e-5))
        for _ in range(self.config['epochs']):
            self.train();total=0.
            for idx in torch.randperm(len(frame)).split(self.config.get('rating_batch_size',8192)):
                uu,ii=u[idx],i[idx];prediction=self.pair_scores(uu,ii)
                penalty=(self.users(uu).square().sum(1)+self.items(ii).square().sum(1)).mean()
                loss=F.mse_loss(prediction,r[idx])+self.config['reg']*penalty
                opt.zero_grad();loss.backward();opt.step();total+=loss.item()*len(idx)
            self.loss_history.append(total/len(frame))
            if self.finish_epoch():break
        return self
    @torch.no_grad()
    def predict_scores(self,user_ids,item_ids=None):
        self.eval();users=np.atleast_1d(user_ids).astype(int)
        items=np.arange(self.n_items) if item_ids is None else np.atleast_1d(item_ids).astype(int)
        result=np.empty((len(users),len(items)),dtype=np.float32)
        for j,user in enumerate(users):
            for start in range(0,len(items),8192):
                ii=torch.as_tensor(items[start:start+8192],dtype=torch.long);uu=torch.full_like(ii,int(user))
                values=self.pair_scores(uu,ii).numpy()
                values[~self.item_support[ii.numpy()]]=self.mean
                if not self.user_support[user]:values[:]=self.mean
                result[j,start:start+len(ii)]=values
        return result


class SparseEASE(Recommender):
    """Same EASE equation, sparse histories and one-user scoring; float64 solve."""
    def __init__(self,nu,ni,config):
        super().__init__(nu,ni,config)
        self.register_buffer('coefficients',torch.zeros(ni,ni))
    def setup(self,frame):
        super().setup(frame);positive=frame[frame.rating>=4]
        self.inputs=sparse.csr_matrix((np.ones(len(positive),dtype=np.float64),(positive.user,positive.item)),shape=(self.n_users,self.n_items))
    def fit(self,frame):
        self.setup(frame);ridge=float(self.config.get('ridge',300.))
        if not np.isfinite(ridge) or ridge<=0:raise ValueError('EASE ridge must be positive and finite')
        gram=(self.inputs.T@self.inputs).toarray();gram.flat[::self.n_items+1]+=ridge
        factor=linalg.cho_factor(gram,overwrite_a=True,check_finite=False)
        inverse=linalg.cho_solve(factor,np.eye(self.n_items),overwrite_b=True,check_finite=False)
        inverse/=-np.diag(inverse).copy()[None,:];np.fill_diagonal(inverse,0.)
        self.coefficients=torch.from_numpy(inverse.astype('float32'))
        return self
    @torch.no_grad()
    def predict_scores(self,user_ids,item_ids=None):
        users=np.atleast_1d(user_ids).astype(int)
        items=np.arange(self.n_items) if item_ids is None else np.atleast_1d(item_ids).astype(int)
        # Slice after sparse multiply avoids copying a dense n_items × n_candidates matrix per user.
        return np.asarray(self.inputs[users]@self.coefficients.numpy())[:,items]


class HistoryTower(Recommender):
    """Small history-conditioned two-tower; X-inspired architecture, not Phoenix.

    One positive per supported user per epoch; other batch items are negatives,
    except every known fit-positive is masked. Sampled target is removed from
    its own history context. No click/dwell labels are invented.
    """
    def __init__(self,nu,ni,config):
        super().__init__(nu,ni,config);d=config.get('dim',64)
        self.user_embedding=nn.Embedding(nu,d)
        self.item_embedding=nn.Embedding(ni+1,d,padding_idx=ni)
        nn.init.normal_(self.user_embedding.weight,std=.05);nn.init.normal_(self.item_embedding.weight,std=.05)
        with torch.no_grad():self.item_embedding.weight[ni].zero_()
        self.user_tower=nn.Sequential(nn.Linear(2*d,d),nn.ReLU(),nn.Linear(d,d))
        self.item_tower=nn.Sequential(nn.Linear(d,d),nn.ReLU(),nn.Linear(d,d))
    def setup(self,frame):
        super().setup(frame)
        pos=frame[frame.rating>=4].copy()
        self.positive=sparse.csr_matrix((np.ones(len(pos)),(pos.user,pos.item)),shape=(self.n_users,self.n_items))
        length=self.config.get('history_length',50)
        history=np.full((self.n_users,length),self.n_items,dtype=np.int64)
        keys=['user']+(['timestamp'] if 'timestamp' in pos else [])+['item']
        for user,group in pos.sort_values(keys,kind='stable').groupby('user',sort=False):
            ids=group.item.to_numpy()[-length:];history[int(user),:len(ids)]=ids
        self.history_ids=torch.from_numpy(history)
    def encode_users(self,users,exclude=None):
        ids=self.history_ids[users];valid=ids!=self.n_items
        if exclude is not None:valid=valid&(ids!=exclude[:,None])
        vectors=self.item_embedding(ids)*valid[:,:,None]
        pooled=vectors.sum(1)/valid.sum(1).clamp(min=1)[:,None]
        return F.normalize(self.user_tower(torch.cat([self.user_embedding(users),pooled],dim=1)),dim=1)
    def encode_items(self,items):return F.normalize(self.item_tower(self.item_embedding(items)),dim=1)
    def negative_mask(self,users,targets):
        mask=self.positive[users.numpy()][:,targets.numpy()].toarray()>0
        np.fill_diagonal(mask,False)
        return torch.from_numpy(mask)
    def fit(self,frame):
        self.setup(frame);rng=np.random.default_rng(self.config['seed']);active=np.flatnonzero(np.diff(self.positive.indptr)>0)
        optimizer=torch.optim.Adam(self.parameters(),lr=self.config['lr'],weight_decay=self.config.get('reg',1e-5))
        for _ in range(self.config['epochs']):
            self.train();total=0.;count=0
            order=rng.permutation(active)
            for start in range(0,len(order),self.config.get('user_batch_size',512)):
                ids=order[start:start+self.config.get('user_batch_size',512)]
                if len(ids)<2:continue
                target=np.array([rng.choice(self.positive.indices[self.positive.indptr[u]:self.positive.indptr[u+1]]) for u in ids])
                u=torch.from_numpy(ids);i=torch.from_numpy(target)
                user_vector=self.encode_users(u,exclude=i);item_vector=self.encode_items(i)
                logits=user_vector@item_vector.T/self.config.get('temperature',.1)
                logits=logits.masked_fill(self.negative_mask(u,i),-1e9)
                loss=F.cross_entropy(logits,torch.arange(len(ids)))
                optimizer.zero_grad();loss.backward();optimizer.step();total+=loss.item()*len(ids);count+=len(ids)
            self.loss_history.append(total/max(count,1))
            if self.finish_epoch():break
        return self
    @torch.no_grad()
    def predict_scores(self,user_ids,item_ids=None):
        self.eval()
        if self.cache is None:
            users=torch.cat([self.encode_users(u) for u in torch.arange(self.n_users).split(1024)])
            items=self.encode_items(torch.arange(self.n_items))
            self.cache=(users,items)
        u=torch.as_tensor(np.atleast_1d(user_ids),dtype=torch.long)
        i=torch.arange(self.n_items) if item_ids is None else torch.as_tensor(np.atleast_1d(item_ids),dtype=torch.long)
        return (self.cache[0][u]@self.cache[1][i].T).numpy()


from .core import AutoRec

class SparseAutoRec(AutoRec):
    """Same observed-only denoising AutoRec objective, CSR inputs/hidden cache.

    This is an infrastructure port/control, not a novel architecture. No dense
    all-user predictions are retained; pair scoring selects decoder rows.
    """
    def setup(self,frame):
        Recommender.setup(self,frame)
        if self.config.get('center_ratings',False):raise ValueError('Sparse control is plain AutoRec only')
        matrix=sparse.csr_matrix((frame.rating.to_numpy(dtype=np.float32),(frame.user,frame.item)),shape=(self.n_users,self.n_items))
        self.inputs=matrix.T.tocsr() if self.config.get('auto_variant','user')=='item' else matrix
    def fit(self,frame):
        self.setup(frame);optimizer=torch.optim.Adam(self.parameters(),lr=self.config['lr'],weight_decay=self.config['reg'])
        for _ in range(self.config['epochs']):
            self.train();losses=[]
            for ids in torch.randperm(self.inputs.shape[0]).split(self.config.get('auto_batch_size',128)):
                target=torch.from_numpy(self.inputs[ids.numpy()].toarray());mask=target!=0
                x=F.dropout(target,p=self.config.get('input_dropout',0.),training=True) if self.config.get('input_dropout',0.) else target
                if not mask.any():continue
                if self.config.get('sparse_training',False):
                    # Same dropout draw/observed MSE/Adam decay as dense training;
                    # avoid dense input matmul and decoding unobserved targets.
                    hidden=torch.sigmoid(torch.sparse.mm(x.to_sparse_coo(),self.encoder.weight.T)+self.encoder.bias)
                    observed_row,observed_col=mask.nonzero(as_tuple=True)
                    prediction=(hidden[observed_row]*self.decoder.weight[observed_col]).sum(1)+self.decoder.bias[observed_col]
                    loss=(prediction-target[observed_row,observed_col]).square().mean()
                else:
                    prediction=self(x)
                    loss=(prediction-target).square()[mask].mean()
                optimizer.zero_grad();loss.backward();optimizer.step();losses.append(loss.item())
            self.loss_history.append(float(np.mean(losses)))
            if self.finish_epoch():break
        return self
    @torch.no_grad()
    def encoded(self):
        self.eval()
        if self.cache is None:
            self.cache=torch.cat([torch.sigmoid(self.encoder(torch.from_numpy(self.inputs[start:start+128].toarray()))) for start in range(0,self.inputs.shape[0],128)])
        return self.cache
    @torch.no_grad()
    def predict_pairs(self,user_ids,item_ids):
        users=np.asarray(user_ids);items=np.asarray(item_ids);out=np.empty(len(users),dtype=float);encoded=self.encoded()
        rows,columns=(items,users) if self.config.get('auto_variant','user')=='item' else (users,items)
        for start in range(0,len(users),32768):
            rr=torch.as_tensor(rows[start:start+32768],dtype=torch.long);cc=torch.as_tensor(columns[start:start+32768],dtype=torch.long)
            value=(encoded[rr]*self.decoder.weight[cc]).sum(1)+self.decoder.bias[cc]
            cold=~self.user_support[users[start:start+32768]]|~self.item_support[items[start:start+32768]]
            value[torch.as_tensor(cold)]=self.mean;out[start:start+len(rr)]=value.numpy()
        return out
    @torch.no_grad()
    def predict_scores(self,user_ids,item_ids=None):
        users=np.atleast_1d(user_ids).astype(int);items=np.arange(self.n_items) if item_ids is None else np.atleast_1d(item_ids).astype(int)
        encoded=self.encoded()
        if self.config.get('auto_variant','user')=='item':
            result=(encoded[items]@self.decoder.weight[users].T+self.decoder.bias[users]).T
        else:result=encoded[users]@self.decoder.weight[items].T+self.decoder.bias[items]
        result[~torch.as_tensor(self.user_support)[users],:]=self.mean
        result[:,~torch.as_tensor(self.item_support)[items]]=self.mean
        return result.numpy()
