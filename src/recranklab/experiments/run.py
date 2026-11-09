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
from recranklab.evaluation.metrics import RankingProtocol, rating_metrics, rating_user_errors
from recranklab.models.core import create, MODELS
from recranklab.utils.repro import config_load, fingerprint, provenance, seed_all
from recranklab.experiments.search import propose, hypothesis
from recranklab.experiments.training import EarlyStopping


def run_one(model_name, config, seed, allow_dirty=False):
    config = {**config, "model": model_name}
    started = time.perf_counter()
    seed_all(seed, config["threads"])
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
    metric=config.get('selection_metric','ndcg@10')
    if metric not in ('rmse','ndcg@10'):raise ValueError('Unsupported selection metric')
    budget = 1 if model_name == "popularity" else config.get("search_budget", len(config["trials"]))
    records = []
    for trial_id in range(budget):
        overrides = {} if model_name == "popularity" else propose(config, records)
        cfg = {**config, **overrides, "seed": seed, "model": model_name}
        seed_all(seed, config["threads"])
        begin = time.perf_counter()
        model = create(model_name, nu, ni, cfg)
        if metric=='rmse' and not model.rating_model:raise ValueError('RMSE selection needs a rating model')
        def evaluate_validation(current):
            if metric=='rmse':return rating_metrics(current,parts['val'])
            values,_=validation.evaluate(current)
            if current.rating_model:values.update(rating_metrics(current,parts['val']))
            return values
        stopper=None
        if config.get('early_stopping',False):
            stopper=EarlyStopping(evaluate_validation,metric,config.get('eval_interval',5),
                                  config.get('patience',3),config.get('min_delta',0.))
            model.epoch_callback=stopper
        try:
            model.fit(parts["train"])
        except (RuntimeError,ValueError) as error:
            if not config.get('continue_failed_trials',False):raise
            records.append({'trial_id':trial_id,'overrides':overrides,'error':str(error),
                            'fit_seconds':time.perf_counter()-begin})
            (out/'validation.json').write_text(json.dumps(records,indent=2))
            print(f'FAILED {tag} seed={seed} trial={trial_id}: {error}',flush=True)
            del model;gc.collect();continue
        fit_seconds = time.perf_counter() - begin
        scores=stopper.best['metrics'] if stopper is not None else evaluate_validation(model)
        record = {"trial_id": trial_id, "overrides": overrides, "metrics": scores,
                  "fit_seconds": fit_seconds, "loss_history": model.loss_history,
                  "hypothesis": hypothesis(config, records, overrides)}
        records.append(record)
        if stopper is not None:
            record.update(best_epoch=stopper.best['epoch'],stopped_epoch=len(model.loss_history),
                          checkpoint_history=stopper.history)
            del model.epoch_callback
        if hasattr(model,'solver_diagnostics'):record['solver_diagnostics']=model.solver_diagnostics
        (out / "validation.json").write_text(json.dumps(records, indent=2))
        print(f"{tag} seed={seed} trial={trial_id} validation_{metric}={scores[metric]:.6f} fit={fit_seconds:.1f}s", flush=True)
        del model
        gc.collect()
    # Tie: first candidate, fixed before test. No test-dependent early stopping.
    valid=[i for i,r in enumerate(records) if 'error' not in r]
    if not valid:raise RuntimeError(f'All validation candidates failed: {tag}')
    choose=min if metric=='rmse' else max
    chosen = choose(valid, key=lambda x: records[x]["metrics"][metric])
    cfg = {**config, **records[chosen]["overrides"], "seed": seed, "model": model_name}
    if 'best_epoch' in records[chosen]:cfg['epochs']=records[chosen]['best_epoch']
    selection = {"selected_trial": chosen, "config": cfg, "selection_metric": metric,
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
        errors=rating_user_errors(model,parts['test']).drop(columns='user').rename_axis('evaluation_row').reset_index()
        errors.to_csv(out/'per_user_rating.csv',index=False)
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
    if model.rating_model:result['per_user_rating_sha256']=digest(out/'per_user_rating.csv')
    if hasattr(model,'solver_diagnostics'):result['solver_diagnostics']=model.solver_diagnostics
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
