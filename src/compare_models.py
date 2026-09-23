"""Run a preregistered comparison without changing the production model."""
import argparse
from datetime import datetime, timezone
import gc
from importlib import metadata
import json
import os
from pathlib import Path
import random
import subprocess
import time

import joblib
import numpy as np
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
import torch

from .comparison_data import SPLITS, batches, collate, load_study_config, prepare_inputs, save_json, study_arrays
from .comparison_metrics import evaluate_logits, fit_operating_points
from .comparison_models import ContractClassifier, FeatureClassifier
from .experiment_v3 import calculate_metrics, read_jsonl
from .preprocessing_v3 import ROOT, digest, file_digest


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def initialize(config, run_id):
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run-id must be a directory name")
    result = ROOT / "results/comparison" / run_id
    model_dir = ROOT / "models/comparison" / run_id
    result.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = result / "protocol.json"
    source_files = [*Path(__file__).parent.glob("comparison_*.py"), Path(__file__)]
    sources = {p.name: file_digest(p) for p in source_files}
    if path.exists():
        protocol = json.loads(path.read_text(encoding="utf-8"))
        if protocol["config"] != config:
            raise ValueError("Refusing to change a running experiment's configuration")
        for name, expected in protocol["source_sha256"].items():
            if file_digest(Path(__file__).with_name(name)) != expected:
                raise ValueError(f"Study source changed after freezing: {name}. Use a new run.")
    else:
        protocol = {"run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(), "config": config, "source_sha256": sources, "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip(), "environment": {name: metadata.version(name) for name in ("torch", "numpy", "scipy", "scikit-learn", "transformers")}, "production_manifest_sha256": file_digest(ROOT / "models/active_model.json"), "test_previously_observed": True, "pretrained_fine_tuning": False, "framework": "PyTorch for all newly trained neural variants, no reuse of V3 weights"}
        protocol["protocol_id"] = digest(protocol)
        save_json(path, protocol)
    return result, model_dir


def predict_neural(model, arrays, config):
    output = np.empty(len(arrays["labels"]), dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for indices in batches(np.diff(arrays["offsets"]), config):
            x, lengths, _ = collate(arrays, indices)
            output[indices] = model(x, lengths).numpy()
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite model predictions")
    return output


def fit_one(model, train_data, val_data, weights, config, seed, result_dir, model_path, feature_mode=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-6)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    best, stale, history = float("inf"), 0, []
    started = time.perf_counter()
    labels = train_data["labels"]
    length_array = np.ones(len(labels), dtype=int) if feature_mode else np.diff(train_data["offsets"])
    for epoch in range(config["epochs"]):
        model.train()
        total_loss, visited = 0.0, 0
        for indices in batches(length_array, config, seed + epoch):
            if feature_mode:
                x = torch.from_numpy(train_data["features"][indices])
                y = torch.as_tensor(labels[indices], dtype=torch.float32)
                logits = model(x)
            else:
                x, lengths, y = collate(train_data, indices)
                logits = model(x, lengths)
            loss = (loss_fn(logits, y) * torch.as_tensor(weights[indices])).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(indices)
            visited += len(indices)
        if visited != len(labels):
            raise ValueError("Training did not visit every contract")
        logits = predict_features(model, val_data["features"]) if feature_mode else predict_neural(model, val_data, config)
        metrics = calculate_metrics(val_data["labels"], expit(logits), 0.5)
        value = metrics["log_loss"]
        row = {"epoch": epoch + 1, "loss": total_loss / visited, "validation_log_loss": value, "validation_roc_auc": metrics["roc_auc"], "validation_f1_macro": metrics["f1_macro"], "learning_rate": optimizer.param_groups[0]["lr"], "contracts_seen": visited, "elapsed_seconds": time.perf_counter() - started}
        history.append(row)
        print(json.dumps({"variant": result_dir.parent.name, "seed": seed, **row}), flush=True)
        if value < best - 1e-8:
            best, stale = value, 0
            torch.save(model.state_dict(), model_path)
        else:
            stale += 1
        scheduler.step(value)
        save_json(result_dir / "history.json", history)
        if stale >= config["early_stopping_patience"]:
            break
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    logits = predict_features(model, val_data["features"]) if feature_mode else predict_neural(model, val_data, config)
    np.save(result_dir / "validation_logits.npy", logits)
    metrics = calculate_metrics(val_data["labels"], expit(logits), 0.5)
    return {"seed": seed, "parameters": sum(p.numel() for p in model.parameters()), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), "epochs_executed": len(history), "best_epoch": int(np.argmin([row["validation_log_loss"] for row in history])) + 1, "training_seconds": time.perf_counter() - started, "validation": metrics, "model_path": str(model_path.relative_to(ROOT)).replace("\\", "/"), "model_sha256": file_digest(model_path), "status": "complete"}


def predict_features(model, features):
    model.eval()
    with torch.inference_mode():
        return np.concatenate([model(torch.from_numpy(np.asarray(features[start:start + 128], dtype=np.float32))).numpy() for start in range(0, len(features), 128)]).astype(np.float64)


def train_scratch(config, result, models, cache, selected_variants=None):
    vocabulary_size = json.loads((cache / "manifest.json").read_text())["vocabulary_size"]
    weights = np.load(cache / "training_weights.npy")
    for mode in config["input_modes"]:
        train_data = study_arrays(config, cache, "train", mode)
        validation_data = study_arrays(config, cache, "validation", mode)
        for architecture in config["architectures"]:
            variant = architecture + "_" + mode
            if selected_variants and variant not in selected_variants:
                continue
            for seed in config["seeds"]:
                result_dir = result / variant / f"seed-{seed}"
                model_dir = models / variant
                result_dir.mkdir(parents=True, exist_ok=True)
                model_dir.mkdir(parents=True, exist_ok=True)
                if (result_dir / "run.json").exists():
                    continue
                seed_everything(seed)
                model = ContractClassifier(architecture, vocabulary_size, config)
                print(f"START {variant} seed {seed}, parameters {sum(p.numel() for p in model.parameters())}", flush=True)
                summary = fit_one(model, train_data, validation_data, weights, config, seed, result_dir, model_dir / f"seed-{seed}.pt")
                save_json(result_dir / "run.json", {"variant": variant, "architecture": architecture, "input_mode": mode, **summary})
                del model
                gc.collect()


def train_baselines(config, result, models, cache):
    train_rows = read_jsonl(ROOT / config["dataset_dir"] / "train.jsonl.gz")
    val_rows = read_jsonl(ROOT / config["dataset_dir"] / "validation.jsonl.gz")
    weights = np.load(cache / "training_weights.npy")
    for mode in config["input_modes"]:
        variant = "tfidf_" + mode
        destination = result / variant
        if (destination / "run.json").exists():
            continue
        started = time.perf_counter()
        def text(row):
            return " ".join(row["tokens"] if mode == "full" else row["tokens"][:512])
        vectorizer = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)
        X = vectorizer.fit_transform([text(row) for row in train_rows])
        clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=config["seeds"][0])
        clf.fit(X, [r["label"] for r in train_rows], sample_weight=weights)
        logits = clf.decision_function(vectorizer.transform([text(row) for row in val_rows]))
        model_path = models / (variant + ".pkl")
        joblib.dump({"vectorizer": vectorizer, "classifier": clf}, model_path)
        destination.mkdir(exist_ok=True)
        np.save(destination / "validation_logits.npy", logits)
        save_json(destination / "run.json", {"variant": variant, "architecture": "tfidf", "input_mode": mode, "seed": config["seeds"][0], "deterministic_single_fit": True, "parameters": int(clf.coef_.size + clf.intercept_.size), "trainable_parameters": int(clf.coef_.size + clf.intercept_.size), "training_seconds": time.perf_counter() - started, "validation": calculate_metrics([r["label"] for r in val_rows], expit(logits), 0.5), "model_path": str(model_path.relative_to(ROOT)).replace("\\", "/"), "model_sha256": file_digest(model_path), "status": "complete"})
        print(f"DONE baseline {variant}", flush=True)


def train_pretrained_heads(config, result, models, cache):
    feature_root = ROOT / ".cache-comparison/codet5"
    feature_manifest = json.loads((feature_root / "manifest.json").read_text())
    weights = np.load(cache / "training_weights.npy")
    for mode in config["pretrained"]["input_modes"]:
        variant = "codet5_frozen_" + mode
        X_train = np.load(feature_root / f"train_{mode}.npy")
        X_val = np.load(feature_root / f"validation_{mode}.npy")
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)
        model_dir = models / variant
        model_dir.mkdir(exist_ok=True)
        joblib.dump(scaler, model_dir / "scaler.pkl")
        train_data = {"features": X_train, "labels": study_arrays(config, cache, "train", mode)["labels"]}
        val_data = {"features": X_val, "labels": study_arrays(config, cache, "validation", mode)["labels"]}
        for seed in config["seeds"]:
            result_dir = result / variant / f"seed-{seed}"
            if (result_dir / "run.json").exists():
                continue
            result_dir.mkdir(parents=True, exist_ok=True)
            seed_everything(seed)
            model = FeatureClassifier(X_train.shape[1], config)
            summary = fit_one(model, train_data, val_data, weights, config, seed, result_dir, model_dir / f"seed-{seed}.pt", feature_mode=True)
            summary["head_parameters"] = summary["parameters"]
            summary["parameters"] += feature_manifest["encoder_parameters"]
            save_json(result_dir / "run.json", {"variant": variant, "architecture": "codet5_frozen", "input_mode": mode, "encoder_revision": feature_manifest["revision"], "encoder_frozen": True, "feature_manifest_sha256": file_digest(feature_root / "manifest.json"), **summary})


def expected_variants(config):
    return [a + "_" + m for a in config["architectures"] for m in config["input_modes"]] + ["codet5_frozen_" + m for m in config["pretrained"]["input_modes"]] + ["tfidf_" + m for m in config["input_modes"]]


def freeze_selection(config, result):
    candidates = []
    labels = study_arrays(config, ROOT / ".cache-comparison/inputs", "validation", "full")["labels"]
    for variant in expected_variants(config):
        paths = [result / variant / "run.json"] if variant.startswith("tfidf_") else [result / variant / f"seed-{seed}/run.json" for seed in config["seeds"]]
        if not all(path.exists() for path in paths):
            raise ValueError(f"Training incomplete: {variant}")
        runs = [json.loads(path.read_text()) for path in paths]
        recalls = []
        for path in paths:
            fpr, recall, _ = roc_curve(labels, np.load(path.parent / "validation_logits.npy"))
            recalls.append(float(recall[fpr <= config["target_fpr"]].max()))
        candidates.append({"variant": variant, "validation_recall_at_fpr_mean": float(np.mean(recalls)), "validation_f1_macro_mean": float(np.mean([run["validation"]["f1_macro"] for run in runs])), "validation_auc_mean": float(np.mean([run["validation"]["roc_auc"] for run in runs])), "validation_log_loss_mean": float(np.mean([run["validation"]["log_loss"] for run in runs])), "runs": [str(path.relative_to(result)).replace("\\", "/") for path in paths]})
    candidates.sort(key=lambda row: (-row["validation_recall_at_fpr_mean"], -row["validation_f1_macro_mean"], row["validation_log_loss_mean"]))
    frozen = {"selected_variant": candidates[0]["variant"], "criterion": config["selection"], "candidates": candidates, "frozen_utc": datetime.now(timezone.utc).isoformat(), "test_used_for_this_selection": False, "historical_test_exposure": True, "production_unchanged": True}
    path = result / "selection.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["candidates"] != candidates:
            raise ValueError("Candidate results changed after selection")
        return previous
    save_json(path, frozen)
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-id", default="comparison-v1-20260922")
    parser.add_argument("--phase", choices=("scratch", "baseline", "pretrained", "select"), required=True)
    parser.add_argument("--variants", nargs="+")
    args = parser.parse_args()
    config = load_study_config(args.config)
    torch.set_num_threads(config["cpu_threads"])
    torch.set_num_interop_threads(1)
    result, models = initialize(config, args.run_id)
    cache = ROOT / ".cache-comparison/inputs"
    prepare_inputs(config, cache)
    if args.phase == "scratch":
        train_scratch(config, result, models, cache, args.variants)
    elif args.phase == "baseline":
        train_baselines(config, result, models, cache)
    elif args.phase == "pretrained":
        train_pretrained_heads(config, result, models, cache)
    else:
        print(json.dumps(freeze_selection(config, result), indent=2))


if __name__ == "__main__":
    main()
