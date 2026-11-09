"""Small from-scratch CPU implementations; no external model code is copied."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import linalg
from scipy import sparse
from sklearn.linear_model import ElasticNet
from torch import nn
from torch.nn import functional as F


class NegativeSampler:
    """The constructor accepts ONLY the current fit partition, never future labels."""
    def __init__(self, train, n_users, n_items, seed, policy="all_history"):
        if policy not in ("all_history", "positives_only"):
            raise ValueError(policy)
        self.rng = np.random.default_rng(seed)
        self.catalog = np.sort(train.item.unique())
        self.blocked = np.zeros((n_users, n_items), dtype=bool)
        history = train if policy == "all_history" else train[train.rating >= 4]
        self.blocked[history.user, history.item] = True
        self.available = (~self.blocked[:, self.catalog]).any(axis=1)

    def sample(self, users):
        users = np.asarray(users, dtype=np.int64)
        if not self.available[users].all():
            raise ValueError("User has no unobserved training-supported candidates")
        result = self.rng.choice(self.catalog, len(users))
        invalid = self.blocked[users, result]
        while invalid.any():
            result[invalid] = self.rng.choice(self.catalog, invalid.sum())
            invalid = self.blocked[users, result]
        return result


class Recommender(nn.Module):
    rating_model = False

    def __init__(self, n_users, n_items, config):
        super().__init__()
        self.n_users, self.n_items, self.config = n_users, n_items, dict(config)
        self.cache = None
        self.loss_history = []

    def setup(self, train):
        self.train_frame = train[["user", "item", "rating"]].copy()
        self.mean = float(train.rating.mean())
        self.user_support = np.bincount(train.user, minlength=self.n_users) > 0
        self.item_support = np.bincount(train.item, minlength=self.n_items) > 0

    @torch.no_grad()
    def predict_scores(self, user_ids, item_ids=None):
        if self.cache is None:
            self.eval()
            self.cache = self.all_scores().detach().cpu().numpy()
        u = np.atleast_1d(user_ids).astype(int)
        i = np.arange(self.n_items) if item_ids is None else np.atleast_1d(item_ids).astype(int)
        return self.cache[np.ix_(u, i)]

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload={"model":self.config["model"],"config":self.config,
                 "n_users":self.n_users,"n_items":self.n_items,
                 "state_dict":self.state_dict(),"loss_history":self.loss_history}
        if self.config.get("compact_checkpoint",False):
            context={"mean":self.mean,"user_support":torch.as_tensor(self.user_support),
                     "item_support":torch.as_tensor(self.item_support),"sparse":{}}
            for name in ("inputs","positive"):
                value=getattr(self,name,None)
                if sparse.issparse(value):
                    value=value.tocsr()
                    context["sparse"][name]={"data":torch.from_numpy(value.data),
                        "indices":torch.from_numpy(value.indices),"indptr":torch.from_numpy(value.indptr)}
            payload["compact_context"]=context
        else:payload["training"]=torch.as_tensor(self.train_frame.to_numpy())
        if self.config["model"]=="history_tower":payload["history_ids"]=self.history_ids
        torch.save(payload,path)

    @staticmethod
    def load(path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model = create(payload["model"], payload["n_users"], payload["n_items"], payload["config"])
        if "compact_context" in payload:
            context=payload["compact_context"];model.mean=context["mean"]
            model.user_support=context["user_support"].numpy();model.item_support=context["item_support"].numpy()
            model.train_frame=pd.DataFrame(columns=["user","item","rating"])
            for name,value in context["sparse"].items():
                setattr(model,name,sparse.csr_matrix((value["data"].numpy(),value["indices"].numpy(),value["indptr"].numpy()),shape=(model.n_users,model.n_items)))
        else:
            frame = pd.DataFrame(payload["training"].numpy(), columns=["user", "item", "rating"])
            frame[["user", "item"]] = frame[["user", "item"]].astype(int)
            model.setup(frame)
        if "history_ids" in payload:model.history_ids=payload["history_ids"]
        model.load_state_dict(payload["state_dict"])
        model.loss_history = payload["loss_history"]
        return model

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def finish_epoch(self):
        """An optional observer receives current weights; only runner supplies validation."""
        self.cache = None
        callback = getattr(self, "epoch_callback", None)
        return bool(callback(self, len(self.loss_history))) if callback else False


class Popularity(Recommender):
    def setup(self, train):
        super().setup(train)
        self.counts = torch.tensor(np.bincount(train.loc[train.rating >= 4, "item"], minlength=self.n_items), dtype=torch.float32)

    def fit(self, train):
        self.setup(train)
        self.cache = None
        return self

    def all_scores(self):
        return self.counts.expand(self.n_users, -1)


class BiasMF(Recommender):
    rating_model = True

    def __init__(self, n_users, n_items, config):
        super().__init__(n_users, n_items, config)
        d = config.get("dim", 32)
        self.users = nn.Embedding(n_users, d)
        self.items = nn.Embedding(n_items, d)
        self.ub = nn.Embedding(n_users, 1)
        self.ib = nn.Embedding(n_items, 1)
        nn.init.normal_(self.users.weight, std=.05)
        nn.init.normal_(self.items.weight, std=.05)
        nn.init.zeros_(self.ub.weight)
        nn.init.zeros_(self.ib.weight)

    def fit(self, train):
        self.setup(train)
        self.cache = None
        opt = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
        rows = torch.tensor(train[["user", "item", "rating"]].to_numpy(), dtype=torch.float32)
        batch = self.config.get("rating_batch_size", 8192)
        for _ in range(self.config["epochs"]):
            losses = []
            for idx in torch.randperm(len(rows)).split(batch):
                u, i, r = rows[idx].T
                u, i = u.long(), i.long()
                ue, ie = self.users(u), self.items(i)
                pred = self.mean + self.ub(u).flatten() + self.ib(i).flatten() + (ue * ie).sum(1)
                reg = (ue.square().sum(1) + ie.square().sum(1) + self.ub(u).flatten().square() + self.ib(i).flatten().square()).mean()
                loss = F.mse_loss(pred, r) + self.config["reg"] * reg
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append((loss.item(), len(idx)))
            self.loss_history.append(sum(v*n for v,n in losses) / len(rows))
            if self.finish_epoch():
                break
        return self

    def all_scores(self):
        scores = self.mean + self.ub.weight + self.ib.weight.T + self.users.weight @ self.items.weight.T
        scores[~self.user_support, :] = self.mean
        scores[:, ~self.item_support] = self.mean
        return scores


class SVDPP(BiasMF):
    """Rating MF plus normalized implicit exposure to ALL fit-rated items.

    User batches compute history contexts once, but all observed rating targets
    contribute squared error. Separate implicit regularization is declared.
    """
    def __init__(self,n_users,n_items,config):
        super().__init__(n_users,n_items,config)
        self.implicit=nn.Embedding(n_items,config.get('dim',64))
        nn.init.normal_(self.implicit.weight,std=.05)

    @staticmethod
    def tensor_graph(matrix):
        coo=matrix.tocoo()
        indices=torch.tensor(np.stack([coo.row,coo.col]),dtype=torch.long)
        return torch.sparse_coo_tensor(indices,torch.tensor(coo.data,dtype=torch.float32),
                                      coo.shape,check_invariants=True).coalesce()

    def setup(self,train):
        super().setup(train)
        self.history_matrix=sparse.csr_matrix((np.ones(len(train),dtype=np.float32),
            (train.user.to_numpy(),train.item.to_numpy())),shape=(self.n_users,self.n_items))
        degree=np.asarray(self.history_matrix.sum(1)).ravel().clip(min=1)
        self.normalized_history=sparse.diags(1/np.sqrt(degree))@self.history_matrix
        self.history_graph=self.tensor_graph(self.normalized_history)

    def fit(self,train):
        self.setup(train);self.cache=None
        optimizer=torch.optim.Adam(self.parameters(),lr=self.config['lr'])
        user_np=train.user.to_numpy();item_np=train.item.to_numpy();rating_np=train.rating.to_numpy()
        order=np.argsort(user_np,kind='stable')
        boundaries=np.cumsum(np.bincount(user_np,minlength=self.n_users))[:-1]
        row_indices=np.split(order,boundaries)
        active=np.flatnonzero(self.user_support)
        batch=self.config.get('user_batch_size',128)
        for _ in range(self.config['epochs']):
            self.train();losses=[]
            for users in torch.tensor(active)[torch.randperm(len(active))].split(batch):
                ids=users.numpy();idx=np.concatenate([row_indices[u] for u in ids])
                lookup=np.full(self.n_users,-1,dtype=np.int64);lookup[ids]=np.arange(len(ids))
                u=torch.tensor(user_np[idx],dtype=torch.long);i=torch.tensor(item_np[idx],dtype=torch.long)
                local=torch.tensor(lookup[user_np[idx]],dtype=torch.long)
                r=torch.tensor(rating_np[idx],dtype=torch.float32)
                graph=self.tensor_graph(self.normalized_history[ids])
                context=torch.sparse.mm(graph,self.implicit.weight)
                ue,ie=self.users(u),self.items(i)
                pred=self.mean+self.ub(u).flatten()+self.ib(i).flatten()+((ue+context[local])*ie).sum(1)
                regular=(ue.square().sum(1)+ie.square().sum(1)+self.ub(u).flatten().square()+self.ib(i).flatten().square()).mean()
                raw_graph=self.tensor_graph(self.history_matrix[ids])
                history_norm=torch.sparse.mm(raw_graph,self.implicit.weight.square().sum(1,keepdim=True)).flatten()
                implicit_penalty=history_norm[local].mean()
                loss=F.mse_loss(pred,r)+self.config['reg']*regular+self.config.get('implicit_reg',.001)*implicit_penalty
                optimizer.zero_grad();loss.backward();optimizer.step()
                losses.append((float(loss.item()),len(idx)))
            self.loss_history.append(sum(v*n for v,n in losses)/len(train))
            if self.finish_epoch():break
        return self

    def all_scores(self):
        context=torch.sparse.mm(self.history_graph,self.implicit.weight)
        score=self.mean+self.ub.weight+self.ib.weight.T+(self.users.weight+context)@self.items.weight.T
        score[~self.user_support,:]=self.mean;score[:,~self.item_support]=self.mean
        return score


class AutoRec(Recommender):
    """User-based AutoRec: sigmoid hidden, linear output, observed-only SSE."""
    rating_model = True

    def __init__(self, n_users, n_items, config):
        super().__init__(n_users, n_items, config)
        width = n_users if config.get("auto_variant", "user") == "item" else n_items
        self.encoder = nn.Linear(width, config.get("hidden", 64))
        self.decoder = nn.Linear(config.get("hidden", 64), width)

    def setup(self, train):
        super().setup(train)
        self.inputs = torch.zeros(self.n_users, self.n_items)
        self.inputs[train.user.to_numpy(), train.item.to_numpy()] = torch.tensor(train.rating.to_numpy(), dtype=torch.float32)
        self.mask = self.inputs != 0
        self.rating_baseline = None
        if self.config.get("center_ratings", False):
            # Fit regularized additive biases on OBSERVED FIT ratings only.
            u, i = train.user.to_numpy(), train.item.to_numpy()
            residual = train.rating.to_numpy(dtype=float) - self.mean
            bu, bi = np.zeros(self.n_users), np.zeros(self.n_items)
            strength = self.config.get("baseline_reg", 10.)
            for _ in range(10):
                bu = np.bincount(u, weights=residual-bi[i], minlength=self.n_users) / (np.bincount(u, minlength=self.n_users)+strength)
                bi = np.bincount(i, weights=residual-bu[u], minlength=self.n_items) / (np.bincount(i, minlength=self.n_items)+strength)
            self.rating_baseline = torch.tensor(self.mean+bu[:,None]+bi[None,:], dtype=torch.float32)
            self.inputs = torch.where(self.mask, self.inputs-self.rating_baseline, 0.)
        if self.config.get("auto_variant", "user") == "item":
            self.inputs = self.inputs.T.contiguous()
            self.mask = self.mask.T.contiguous()

    def forward(self, x):
        return self.decoder(torch.sigmoid(self.encoder(x)))

    def fit(self, train):
        self.setup(train)
        self.cache = None
        opt = torch.optim.Adam(self.parameters(), lr=self.config["lr"], weight_decay=self.config["reg"])
        for _ in range(self.config["epochs"]):
            self.train()
            losses = []
            for users in torch.randperm(len(self.inputs)).split(self.config.get("auto_batch_size", 128)):
                inputs = self.inputs[users]
                # Denoising corrupts inputs only; all observed targets remain supervised.
                dropout = self.config.get("input_dropout", 0.)
                if dropout:
                    inputs = F.dropout(inputs, p=dropout, training=True)
                pred = self(inputs)
                mask = self.mask[users]
                if not mask.any():
                    continue
                loss = (pred - self.inputs[users]).square()[mask].mean()
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())
            self.loss_history.append(float(np.mean(losses)))
            if self.finish_epoch():
                break
        return self

    def all_scores(self):
        scores = torch.cat([self(x) for x in self.inputs.split(256)])
        if self.config.get("auto_variant", "user") == "item":
            scores = scores.T.contiguous()
        if self.rating_baseline is not None:
            scores = scores + self.rating_baseline
        scores[~self.user_support, :] = self.mean
        scores[:, ~self.item_support] = self.mean
        return scores


class BPRMF(Recommender):
    def __init__(self, n_users, n_items, config):
        super().__init__(n_users, n_items, config)
        self.embedding = nn.Embedding(n_users + n_items, config.get("dim", 32))
        nn.init.normal_(self.embedding.weight, std=.1)

    def representations(self):
        return self.embedding.weight[:self.n_users], self.embedding.weight[self.n_users:]

    def fit(self, train):
        self.setup(train)
        self.cache = None
        positives = train[train.rating >= 4]
        sampler = NegativeSampler(train, self.n_users, self.n_items, self.config["seed"], self.config.get("negative_policy", "all_history"))
        positives = positives[sampler.available[positives.user]]
        if positives.empty:
            raise ValueError("No trainable positives")
        rows = torch.tensor(positives[["user", "item"]].to_numpy(), dtype=torch.long)
        opt = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
        for _ in range(self.config["epochs"]):
            losses = []
            for idx in torch.randperm(len(rows)).split(self.config.get("ranking_batch_size", 32768)):
                u, i = rows[idx].T
                j = torch.from_numpy(sampler.sample(u.numpy()))
                users, items = self.representations()
                diff = (users[u] * (items[i] - items[j])).sum(1)
                # Regularize initial, not propagated embeddings.
                reg = (self.embedding(u).square().sum(1) + self.embedding(i + self.n_users).square().sum(1) + self.embedding(j + self.n_users).square().sum(1)).mean() / 2
                loss = F.softplus(-diff).mean() + self.config["reg"] * reg
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append((loss.item(), len(idx)))
            self.loss_history.append(sum(v*n for v,n in losses) / len(rows))
            if self.finish_epoch():
                break
        return self

    def all_scores(self):
        u, i = self.representations()
        return u @ i.T


class LightGCN(BPRMF):
    def setup(self, train):
        super().setup(train)
        pos = train[train.rating >= 4]
        u = torch.tensor(pos.user.to_numpy(), dtype=torch.long)
        i = torch.tensor(pos.item.to_numpy(), dtype=torch.long) + self.n_users
        row, col = torch.cat([u, i]), torch.cat([i, u])
        size = self.n_users + self.n_items
        degree = torch.bincount(row, minlength=size).float().clamp(min=1)
        weight = (degree[row] * degree[col]).rsqrt()
        self.graph = torch.sparse_coo_tensor(torch.stack([row, col]), weight, (size, size), check_invariants=True).coalesce()

    def representations(self):
        x = self.embedding.weight
        layers = [x]
        for _ in range(self.config.get("layers", 2)):
            x = torch.sparse.mm(self.graph, x)
            layers.append(x)
        x = torch.stack(layers).mean(0)
        return x[:self.n_users], x[self.n_users:]


class EASE(Recommender):
    """EASE (Steck, WWW 2019): closed-form ridge, exactly zero diagonal.

    X contains fit positives only. Solve in float64, then store B in float32.
    No labels, history or statistics from validation/test enter this fit.
    """
    def __init__(self, n_users, n_items, config):
        super().__init__(n_users, n_items, config)
        self.register_buffer("coefficients", torch.zeros(n_items, n_items))

    def setup(self, train):
        super().setup(train)
        self.inputs = torch.zeros(self.n_users, self.n_items)
        pos = train[train.rating >= 4]
        self.inputs[pos.user.to_numpy(), pos.item.to_numpy()] = 1.

    def fit(self, train):
        self.setup(train)
        self.cache = None
        ridge = float(self.config.get("ridge", 1000.))
        if not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("EASE ridge must be positive and finite")
        x = self.inputs.numpy().astype(np.float64)
        gram = x.T @ x
        gram.flat[::self.n_items + 1] += ridge
        inverse = linalg.cho_solve(linalg.cho_factor(gram, check_finite=False),
                                 np.eye(self.n_items), check_finite=False)
        coefficients = -inverse / np.diag(inverse)[None, :]
        np.fill_diagonal(coefficients, 0.)
        self.coefficients = torch.from_numpy(coefficients.astype(np.float32))
        return self

    def all_scores(self):
        return self.inputs @ self.coefficients


def retain_neighbors(coefficients, top_k):
    """Stable source-neighbor selection per target column, never keep diagonal."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    result = np.array(coefficients, copy=True)
    np.fill_diagonal(result, 0.)
    if top_k < len(result):
        order = np.argsort(-result, axis=0, kind="stable")
        result[order[top_k:], np.arange(len(result))[None, :]] = 0.
    return result


class ItemKNN(EASE):
    """Positive-only cosine item similarities with shrinkage and top-K sparsity."""
    def fit(self, train):
        self.setup(train)
        self.cache = None
        x = self.inputs.numpy().astype(np.float64)
        gram = x.T @ x
        norm = np.sqrt(np.diag(gram))
        denominator = norm[:,None]*norm[None,:] + self.config.get("shrink", 10.)
        similarity = np.divide(gram, denominator, out=np.zeros_like(gram), where=denominator>0)
        self.coefficients = torch.tensor(retain_neighbors(similarity, self.config.get("top_k",200)), dtype=torch.float32)
        return self


class RP3Beta(EASE):
    """User→item→user→item walk, alpha powers, target popularity^-beta.

    Keep top-K source neighbors for each target. Optional row-L1 normalization
    applies after sparsification; this precise variant is declared in the study.
    """
    def fit(self, train):
        self.setup(train)
        self.cache = None
        x = self.inputs.numpy().astype(np.float64)
        ud, item_degree = x.sum(1).clip(min=1), x.sum(0).clip(min=1)
        alpha, beta = self.config.get("alpha",1.), self.config.get("beta",.6)
        if alpha <= 0 or beta < 0:
            raise ValueError("Invalid walk powers")
        transition = (x.T/item_degree[:,None])**alpha @ (x/ud[:,None])**alpha
        transition /= item_degree[None,:]**beta
        transition = retain_neighbors(transition, self.config.get("top_k",200))
        if self.config.get("normalize_similarity",True):
            transition /= np.abs(transition).sum(1,keepdims=True).clip(min=1e-12)
        self.coefficients = torch.tensor(transition, dtype=torch.float32)
        return self


class SLIM(EASE):
    """Independent positive ElasticNet regressions; zero self coefficient.

    Each worker owns its CSC copy, so zeroing a target column cannot race with
    another worker. Nonconvergence is an explicit failed trial, not a score.
    """
    def fit(self, train):
        from concurrent.futures import ThreadPoolExecutor
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        self.setup(train)
        self.cache = None
        x = sparse.csc_matrix(self.inputs.numpy().astype(np.float64))
        workers = int(self.config.get("slim_workers", 2))
        if workers < 1:
            raise ValueError("slim_workers must be positive")
        def solve(indices):
            local = x.copy()
            fitted=[]; failures=0; maximum_ratio=0.
            estimator=ElasticNet(alpha=self.config.get("elastic_alpha",.001),
                                 l1_ratio=self.config.get("l1_ratio",.1),positive=True,
                                 fit_intercept=False,copy_X=False,selection="cyclic",
                                 max_iter=self.config.get("slim_max_iter",1000),
                                 tol=self.config.get("slim_tol",.001))
            for target in indices:
                lo,hi=local.indptr[target:target+2]
                values=local.data[lo:hi].copy()
                if not len(values):
                    continue
                y=np.asarray(local[:,target].toarray()).ravel()
                local.data[lo:hi]=0.
                try:
                    estimator.fit(local,y)
                finally:
                    local.data[lo:hi]=values
                coef=estimator.coef_.copy();coef[target]=0.
                tolerance=estimator.tol*np.mean(y*y)
                ratio=float(estimator.dual_gap_)/max(tolerance,1e-15)
                maximum_ratio=max(maximum_ratio,ratio)
                failures+=int(ratio>1.001)
                fitted.append((int(target),coef))
            return fitted,failures,maximum_ratio
        coefficients=np.zeros((self.n_items,self.n_items),dtype=np.float32)
        failures=0;max_ratio=0.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",ConvergenceWarning)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for fitted,failed,ratio in pool.map(solve,np.array_split(np.arange(self.n_items),workers)):
                    failures+=failed;max_ratio=max(max_ratio,ratio)
                    for item,coef in fitted:
                        coefficients[:,item]=coef
        self.solver_diagnostics={"nonconverged_columns":failures,"max_dual_gap_over_tolerance":max_ratio,
                                 "max_iter":self.config.get("slim_max_iter",1000)}
        if failures:
            raise RuntimeError(f"SLIM failed convergence on {failures} columns")
        self.coefficients=torch.from_numpy(coefficients)
        return self


class MultiVAE(Recommender):
    """Binary-fit-positive multinomial VAE; posterior mean at evaluation."""
    def __init__(self,n_users,n_items,config):
        super().__init__(n_users,n_items,config)
        hidden,latent=config.get("hidden",200),config.get("latent",128)
        self.encoder=nn.Sequential(nn.Linear(n_items,hidden),nn.Tanh(),nn.Linear(hidden,2*latent))
        self.decoder=nn.Sequential(nn.Linear(latent,hidden),nn.Tanh(),nn.Linear(hidden,n_items))
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight);nn.init.normal_(layer.bias,std=.001)

    def setup(self,train):
        super().setup(train)
        self.inputs=torch.zeros(self.n_users,self.n_items)
        positive=train[train.rating>=4]
        self.inputs[positive.user.to_numpy(),positive.item.to_numpy()]=1.

    def forward(self,x):
        normalized=F.normalize(x,p=2,dim=1)
        corrupted=F.dropout(normalized,p=self.config.get("input_dropout",.5),training=self.training)
        mu,logvar=self.encoder(corrupted).chunk(2,dim=1)
        logvar=logvar.clamp(-10,10)
        z=mu+torch.randn_like(mu)*torch.exp(.5*logvar) if self.training else mu
        return self.decoder(z),mu,logvar

    def fit(self,train):
        self.setup(train);self.cache=None
        optimizer=torch.optim.Adam(self.parameters(),lr=self.config["lr"],weight_decay=self.config.get("reg",0.))
        supported=torch.where(self.inputs.sum(1)>0)[0]
        if not len(supported):raise ValueError("No positive histories for MultiVAE")
        batch=self.config.get("auto_batch_size",256)
        steps_per_epoch=int(np.ceil(len(supported)/batch))
        anneal_steps=max(1,self.config.get("anneal_epochs",20)*steps_per_epoch)
        updates=0
        for _ in range(self.config["epochs"]):
            self.train();losses=[]
            for ids in supported[torch.randperm(len(supported))].split(batch):
                x=self.inputs[ids]
                logits,mu,logvar=self(x)
                reconstruction=-(F.log_softmax(logits,dim=1)*x).sum(1).mean()
                kl=-.5*(1+logvar-mu.square()-logvar.exp()).sum(1).mean()
                anneal=self.config.get("anneal_cap",.2)*min(1.,updates/anneal_steps)
                loss=reconstruction+anneal*kl
                optimizer.zero_grad();loss.backward();optimizer.step()
                updates+=1;losses.append((float(loss.item()),len(ids)))
            self.loss_history.append(sum(v*n for v,n in losses)/len(supported))
            if self.finish_epoch():break
        return self

    def all_scores(self):
        self.eval()
        return torch.cat([self(x)[0] for x in self.inputs.split(256)])


MODELS = {"popularity": Popularity, "biasmf": BiasMF, "autorec": AutoRec, "bpr_mf": BPRMF, "lightgcn": LightGCN, "ease": EASE,
          "itemknn":ItemKNN,"rp3beta":RP3Beta,"slim":SLIM,"multivae":MultiVAE,"svdpp":SVDPP}


def create(name, n_users, n_items, config):
    return MODELS[name](n_users, n_items, {**config, "model": name})
