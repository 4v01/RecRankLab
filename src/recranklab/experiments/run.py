"""Predeclared finite tuning -> train+validation refit -> one final test per seed."""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import resource
import time

import pandas as pd

from recranklab.data.prepare import load, digest
from recranklab.evaluation.metrics import RankingProtocol, rating_metrics
from recranklab.models.core import create, MODELS
from recranklab.utils.repro import config_load, fingerprint, provenance, seed_all
from recranklab.experiments.search import propose, hypothesis


def run_one(model_name, config, seed, allow_dirty=False):
    config = {**config, "model": model_name}
    started = time.perf_counter()
    env = provenance(require_clean=not allow_dirty)
    parts, data_manifest = load(config["dataset"], config["data_root"])
    tag = config.get("experiment", model_name)
    out = Path(config["output_root"]) / config["dataset"] / tag / str(seed)
    signature = fingerprint({"config": config, "seed": seed, "model": model_name,
                             "commit": env["git_commit"], "data": data_manifest["fingerprint"]})
    if (out / "result.json").exists():
        saved = json.loads((out / "result.json").read_text())
        if saved["signature"] != signature:
            raise RuntimeError(f"Refusing to overwrite a final test: {out}. Use a new output root.")
        print(f"Already complete, reusing {out}", flush=True)
        return saved
    out.mkdir(parents=True, exist_ok=True)
    nu, ni = data_manifest["n_users"], data_manifest["n_items"]
    validation = RankingProtocol(parts["train"], parts["val"], nu, ni)
    budget = 1 if model_name == "popularity" else config.get("search_budget", len(config["trials"]))
    records = []
    for trial_id in range(budget):
        overrides = {} if model_name == "popularity" else propose(config, records)
        cfg = {**config, **overrides, "seed": seed, "model": model_name}
        seed_all(seed, config["threads"])
        begin = time.perf_counter()
        model = create(model_name, nu, ni, cfg).fit(parts["train"])
        fit_seconds = time.perf_counter() - begin
        scores, _ = validation.evaluate(model)
        if model.rating_model:
            scores.update(rating_metrics(model, parts["val"]))
        record = {"trial_id": trial_id, "overrides": overrides, "metrics": scores,
                  "fit_seconds": fit_seconds, "loss_history": model.loss_history,
                  "hypothesis": hypothesis(config, records, overrides)}
        records.append(record)
        (out / "validation.json").write_text(json.dumps(records, indent=2))
        print(f"{tag} seed={seed} trial={trial_id} val_ndcg@10={scores['ndcg@10']:.6f} fit={fit_seconds:.1f}s", flush=True)
        del model
        gc.collect()
    # Tie: first candidate, fixed before test. No test-dependent early stopping.
    chosen = max(range(len(records)), key=lambda x: records[x]["metrics"]["ndcg@10"])
    cfg = {**config, **records[chosen]["overrides"], "seed": seed, "model": model_name}
    selection = {"selected_trial": chosen, "config": cfg, "selection_metric": "ndcg@10",
                 "git_commit": env["git_commit"], "data_fingerprint": data_manifest["fingerprint"]}
    (out / "selection.json").write_text(json.dumps(selection, indent=2))
    # Final test protocol is instantiated only after selection is persisted.
    final_train = pd.concat([parts["train"], parts["val"]], ignore_index=True)
    seed_all(seed, config["threads"])
    begin = time.perf_counter()
    model = create(model_name, nu, ni, cfg).fit(final_train)
    fit_seconds = time.perf_counter() - begin
    model.save(out / "model.pt")
    evaluation_start = time.perf_counter()
    test = RankingProtocol(final_train, parts["test"], nu, ni)
    metrics, users = test.evaluate(model)
    if model.rating_model:
        metrics.update(rating_metrics(model, parts["test"]))
    evaluation_seconds = time.perf_counter() - evaluation_start
    # Export anonymous aligned evaluation rows, not raw user IDs or MovieLens ratings.
    users = users.drop(columns="user").rename_axis("evaluation_row").reset_index()
    users.to_csv(out / "per_user.csv", index=False)
    result = {"schema_version": 1, "signature": signature, "dataset": config["dataset"],
              "experiment": tag, "model": model_name, "seed": seed,
              "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "config": config, "config_sha256": fingerprint(config), "selected_config": cfg,
              "selected_trial": chosen, "validation_trials": records,
              "test_metrics": metrics, "data": data_manifest, "environment": env,
              "trainable_parameters": model.parameter_count(), "final_fit_seconds": fit_seconds,
              "evaluation_seconds": evaluation_seconds, "total_seconds": time.perf_counter() - started,
              "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "peak_rss_scope": "process lifetime high-water mark, including earlier seeds",
              "final_loss_history": model.loss_history,
              "checkpoint_sha256": digest(out / "model.pt"), "per_user_sha256": digest(out / "per_user.csv")}
    (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f"FINAL {tag} seed={seed}: {json.dumps(metrics)}", flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODELS), required=True)
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--dataset", choices=["ml-1m", "ml-100k"])
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--smoke", action="store_true", help="Separate 1-epoch 100K smoke; never headline")
    p.add_argument("--allow-dirty", action="store_true", help="Development only, recorded in provenance")
    a = p.parse_args()
    cfg = config_load(a.config)
    if a.dataset:
        cfg["dataset"] = a.dataset
    if a.smoke:
        cfg.update(dataset="ml-100k", epochs=1, trials=[cfg["trials"][0]], output_root="outputs/smoke")
    for seed in a.seeds or cfg["seeds"]:
        run_one(a.model, cfg, seed, a.allow_dirty)


if __name__ == "__main__":
    main()
