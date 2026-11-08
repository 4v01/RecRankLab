"""Small from-scratch CPU implementations; no external model code is copied."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import linalg
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
        torch.save({"model": self.config["model"], "config": self.config,
                    "n_users": self.n_users, "n_items": self.n_items,
                    "training": torch.as_tensor(self.train_frame.to_numpy()),
                    "state_dict": self.state_dict(), "loss_history": self.loss_history}, path)

    @staticmethod
    def load(path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model = create(payload["model"], payload["n_users"], payload["n_items"], payload["config"])
        frame = pd.DataFrame(payload["training"].numpy(), columns=["user", "item", "rating"])
        frame[["user", "item"]] = frame[["user", "item"]].astype(int)
        model.setup(frame)
        model.load_state_dict(payload["state_dict"])
        model.loss_history = payload["loss_history"]
        return model

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


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
        return self

    def all_scores(self):
        scores = self.mean + self.ub.weight + self.ib.weight.T + self.users.weight @ self.items.weight.T
        scores[~self.user_support, :] = self.mean
        scores[:, ~self.item_support] = self.mean
        return scores


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
        return self

    def all_scores(self):
        scores = torch.cat([self(x) for x in self.inputs.split(256)])
        if self.config.get("auto_variant", "user") == "item":
            scores = scores.T.contiguous()
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


MODELS = {"popularity": Popularity, "biasmf": BiasMF, "autorec": AutoRec, "bpr_mf": BPRMF, "lightgcn": LightGCN, "ease": EASE}


def create(name, n_users, n_items, config):
    return MODELS[name](n_users, n_items, {**config, "model": name})
