"""Official MovieLens ingestion; deterministic user-relative temporal holdout."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import yaml

URL = "https://files.grouplens.org/datasets/movielens/{}.zip"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_rows(frame):
    """Tie order is timestamp, original item ID, source row ID. Reject duplicates."""
    if frame.duplicated(["user", "item"]).any():
        raise ValueError("Repeated user-item ratings are not supported")
    parts = {k: [] for k in ("train", "val", "test")}
    for _, g in frame.groupby("user", sort=True):
        g = g.sort_values(["timestamp", "item", "row_id"], kind="stable")
        if len(g) < 3:
            raise ValueError("Each user needs at least three observations")
        a = min(max(1, int(.8 * len(g))), len(g) - 2)
        b = min(max(a + 1, int(.9 * len(g))), len(g) - 1)
        for key, rows in zip(parts, (g.iloc[:a], g.iloc[a:b], g.iloc[b:])):
            parts[key].append(rows.row_id.to_numpy(dtype=np.int64))
    return {k: np.concatenate(v) for k, v in parts.items()}


def prepare(dataset, root="data"):
    if dataset not in ("ml-1m", "ml-100k"):
        raise ValueError(dataset)
    raw = Path(root) / "raw" / f"{dataset}.zip"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        temp = raw.with_suffix(".partial")
        with urllib.request.urlopen(URL.format(dataset), timeout=120) as response:
            temp.write_bytes(response.read())
        temp.replace(raw)
    with zipfile.ZipFile(raw) as archive:
        member = f"{dataset}/" + ("ratings.dat" if dataset == "ml-1m" else "u.data")
        ratings_bytes = archive.read(member)
    frame = pd.read_csv(io.BytesIO(ratings_bytes), sep="::" if dataset == "ml-1m" else "\t",
                        engine="python", names=["user", "item", "rating", "timestamp"])
    if frame.isna().any().any() or not frame.rating.between(1, 5).all():
        raise ValueError("Invalid ratings")
    # Structural ID remapping only. No held-out rating statistics enter model fitting.
    users = np.sort(frame.user.unique())
    items = np.sort(frame.item.unique())
    frame["user"] = np.searchsorted(users, frame.user)
    frame["item"] = np.searchsorted(items, frame.item)
    frame["row_id"] = np.arange(len(frame))
    parts = split_rows(frame)
    out = Path(root) / "processed" / dataset
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "ratings.csv", index=False)
    np.savez(out / "splits.npz", **parts)
    np.savez(out / "id_maps.npz", original_users=users, original_items=items)
    manifest = {"dataset": dataset, "source_url": URL.format(dataset),
                "archive_sha256": digest(raw), "ratings_member_sha256": hashlib.sha256(ratings_bytes).hexdigest(),
                "split_policy": "user temporal floor(.8n)/floor(.9n); ties timestamp,item,row_id",
                "n_users": len(users), "n_items": len(items), "n_ratings": len(frame),
                "split_counts": {k: len(v) for k, v in parts.items()},
                "split_index_sha256": {k: hashlib.sha256(v.astype('<i8').tobytes()).hexdigest() for k, v in parts.items()},
                "files": {p.name: digest(p) for p in out.iterdir() if p.name != "manifest.json"}}
    manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load(dataset, root="data"):
    out = Path(root) / "processed" / dataset
    manifest = json.loads((out / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if digest(out / name) != expected:
            raise ValueError(f"Data integrity failure: {name}")
    frame = pd.read_csv(out / "ratings.csv")
    with np.load(out / "splits.npz") as parts:
        views = {k: frame.iloc[parts[k]].copy() for k in ("train", "val", "test")}
    return views, manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["ml-1m", "ml-100k"], default="ml-1m")
    p.add_argument("--config", default="configs/base.yaml")
    a = p.parse_args()
    config = yaml.safe_load(Path(a.config).read_text())
    print(json.dumps(prepare(a.dataset, config.get("data_root", "data")), indent=2))


if __name__ == "__main__":
    main()
