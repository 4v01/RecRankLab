"""One evaluator for every model; full supported catalog, deterministic ties."""
import hashlib

import numpy as np
import pandas as pd


def stable_top_indices(scores,k):
    if len(scores)<=k:return np.argsort(-scores,kind="stable")
    threshold=np.partition(scores,len(scores)-k)[len(scores)-k]
    selected=np.flatnonzero(scores>=threshold)
    return selected[np.argsort(-scores[selected],kind="stable")[:k]]


class _LazyCandidates:
    """Same full catalog and mask, materialized for only the requested user."""
    def __init__(self,catalog,history,users):self.catalog,self.history,self.users=catalog,history,users
    def __getitem__(self,user):
        return self.catalog[~np.isin(self.catalog,list(self.history[int(user)]))]
    def __iter__(self):return iter(self.users)
    def __len__(self):return len(self.users)


class RankingProtocol:
    def __init__(self, history, target, n_users, n_items):
        self.catalog = np.sort(history.item.unique())
        self.history = {int(u): set(g.item) for u, g in history.groupby("user")}
        supported = set(self.catalog)
        self.targets = {}
        positives = target[target.rating >= 4]
        cold = prior = 0
        for u, g in positives.groupby("user"):
            raw = set(g.item)
            cold += len(raw - supported)
            prior += len(raw & supported & self.history.get(u, set()))
            valid = raw & supported - self.history.get(u, set())
            if valid and u in self.history:
                self.targets[int(u)] = valid
        self.users = np.array(sorted(self.targets), dtype=np.int64)
        self.candidates = _LazyCandidates(self.catalog, self.history, self.users)
        self.audit = {"eligible_users": len(self.users), "total_users": n_users,
                      "excluded_users": n_users - len(self.users),
                      "users_without_heldout_positives": n_users - positives.user.nunique(),
                      "cold_start_positive_targets": cold, "prior_positive_targets": prior,
                      "heldout_positive_targets": len(positives), "catalog_size": len(self.catalog),
                      "unsupported_catalog_items": n_items - len(self.catalog),
                      "eligible_user_sha256": hashlib.sha256(self.users.astype('<i8').tobytes()).hexdigest(),
                      "candidate_catalog_sha256": hashlib.sha256(self.catalog.astype('<i8').tobytes()).hexdigest()}

    def evaluate(self, model, ks=(10, 20), batch_size=1):
        rows = []
        recommended = {k: set() for k in ks}
        if batch_size<1:raise ValueError("Invalid evaluation batch size")
        def scored_users():
            for start in range(0,len(self.users),batch_size):
                batch=self.users[start:start+batch_size]
                if batch_size==1:
                    u=int(batch[0]);candidates=self.candidates[u]
                    yield u,candidates,model.predict_scores([u],candidates)[0]
                else:
                    matrix=model.predict_scores(batch,self.catalog)
                    for j,u in enumerate(batch):
                        keep=~np.isin(self.catalog,list(self.history[int(u)]))
                        yield int(u),self.catalog[keep],matrix[j,keep]
        for u,candidates,scores in scored_users():
            if not np.isfinite(scores).all():
                raise ValueError("Nonfinite ranking scores")
            # Exact stable top-K, including all boundary ties before ID tie-break.
            ranking=candidates[stable_top_indices(scores,max(ks))]
            truth = self.targets[int(u)]
            hits = np.array([i in truth for i in ranking], dtype=float)
            row = {"user": int(u)}
            for k in ks:
                recommended[k].update(ranking[:k].tolist())
                row[f"recall@{k}"] = float(hits[:k].sum() / len(truth))
                dcg = (hits[:k] / np.log2(np.arange(len(hits[:k])) + 2)).sum()
                ideal = (1 / np.log2(np.arange(min(k, len(truth))) + 2)).sum()
                row[f"ndcg@{k}"] = float(dcg / ideal)
            rows.append(row)
        frame = pd.DataFrame(rows, columns=["user"] + [f"{m}@{k}" for k in ks for m in ("recall", "ndcg")])
        metrics = {c: float(frame[c].mean()) if len(frame) else None for c in frame.columns if c != "user"}
        metrics.update({f"coverage@{k}": len(v) / len(self.catalog) for k, v in recommended.items()})
        return {**metrics, **self.audit}, frame


def rating_metrics(model, target):
    if not len(target):
        return {"rmse": None, "mae": None, "r2":None, "rating_count": 0}
    if hasattr(model,"predict_pairs"):
        prediction=np.asarray(model.predict_pairs(target.user.to_numpy(),target.item.to_numpy()),dtype=float)
        truth=target.rating.to_numpy(dtype=float)
    else:
        predictions = np.array([model.predict_scores([u], g.item.to_numpy())[0] for u, g in target.groupby("user")], dtype=object)
        prediction = np.concatenate([np.asarray(x, dtype=float) for x in predictions])
        truth = np.concatenate([g.rating.to_numpy() for _, g in target.groupby("user")])
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite rating predictions")
    error = prediction - truth
    clipped_error=np.clip(prediction,getattr(model,"config",{}).get("rating_min",1.),5.)-truth
    sst=float(np.square(truth-truth.mean()).sum())
    r2=float(1-np.square(error).sum()/sst) if len(truth)>=2 and sst>0 else None
    return {"rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(abs(error))),
            "r2":r2,
            "rmse_clipped_aux":float(np.sqrt(np.mean(clipped_error**2))),
            "mae_clipped_aux":float(np.mean(np.abs(clipped_error))),
            "rating_count": len(truth), "cold_item_ratings": int((~model.item_support[target.item]).sum()),
            "cold_user_ratings": int((~model.user_support[target.user]).sum())}


def rating_user_errors(model,target):
    """Observed-only sufficient statistics; never export ratings/prediction pairs."""
    if hasattr(model,"predict_pairs"):
        error=model.predict_pairs(target.user.to_numpy(),target.item.to_numpy())-target.rating.to_numpy()
        if not np.isfinite(error).all():raise ValueError('Nonfinite rating predictions')
        stats=pd.DataFrame({'user':target.user.to_numpy(),'count':np.ones(len(target),dtype=int),'sse':np.square(error),'sae':np.abs(error)})
        return stats.groupby('user',sort=True,as_index=False).sum()
    rows=[]
    for user,group in target.groupby('user',sort=True):
        prediction=model.predict_scores([user],group.item.to_numpy())[0]
        error=prediction-group.rating.to_numpy()
        if not np.isfinite(error).all():raise ValueError('Nonfinite rating predictions')
        rows.append({'user':int(user),'count':len(error),'sse':float(np.square(error).sum()),'sae':float(np.abs(error).sum())})
    return pd.DataFrame(rows)


def paired_bootstrap(differences, seed=2026, n_boot=2000):
    """Resample users, after averaging each user's paired differences across seeds."""
    values = np.asarray(differences, dtype=float)
    if not len(values):
        raise ValueError("Empty paired comparison")
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(len(values), size=len(values))].mean() for _ in range(n_boot)]
    lo, hi = np.quantile(means, [.025, .975])
    return {"mean_difference": float(values.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "bootstrap_seed": seed, "bootstrap_replicates": n_boot, "unit": "user, seed-averaged"}
