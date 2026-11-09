"""One evaluator for every model; full supported catalog, deterministic ties."""
import hashlib

import numpy as np
import pandas as pd


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
        self.candidates = {int(u): self.catalog[~np.isin(self.catalog, list(self.history[int(u)]))]
                           for u in self.users}
        self.audit = {"eligible_users": len(self.users), "total_users": n_users,
                      "excluded_users": n_users - len(self.users),
                      "users_without_heldout_positives": n_users - positives.user.nunique(),
                      "cold_start_positive_targets": cold, "prior_positive_targets": prior,
                      "heldout_positive_targets": len(positives), "catalog_size": len(self.catalog),
                      "unsupported_catalog_items": n_items - len(self.catalog),
                      "eligible_user_sha256": hashlib.sha256(self.users.astype('<i8').tobytes()).hexdigest(),
                      "candidate_catalog_sha256": hashlib.sha256(self.catalog.astype('<i8').tobytes()).hexdigest()}

    def evaluate(self, model, ks=(10, 20)):
        rows = []
        recommended = {k: set() for k in ks}
        for u in self.users:
            candidates = self.candidates[int(u)]
            scores = model.predict_scores([u], candidates)[0]
            if not np.isfinite(scores).all():
                raise ValueError("Nonfinite ranking scores")
            # Stable sort preserves ascending item ID for equal scores; no random tie jitter.
            ranking = candidates[np.argsort(-scores, kind="stable")[:max(ks)]]
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
        return {"rmse": None, "mae": None, "rating_count": 0}
    predictions = np.array([model.predict_scores([u], g.item.to_numpy())[0] for u, g in target.groupby("user")], dtype=object)
    prediction = np.concatenate([np.asarray(x, dtype=float) for x in predictions])
    truth = np.concatenate([g.rating.to_numpy() for _, g in target.groupby("user")])
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite rating predictions")
    error = prediction - truth
    clipped_error=np.clip(prediction,1.,5.)-truth
    return {"rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(abs(error))),
            "rmse_clipped_aux":float(np.sqrt(np.mean(clipped_error**2))),
            "mae_clipped_aux":float(np.mean(np.abs(clipped_error))),
            "rating_count": len(truth), "cold_item_ratings": int((~model.item_support[target.item]).sum()),
            "cold_user_ratings": int((~model.user_support[target.user]).sum())}


def rating_user_errors(model,target):
    """Observed-only sufficient statistics; never export ratings/prediction pairs."""
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
