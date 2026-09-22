"""Reproducible multi-seed experiment; freeze selection/calibration before test."""
import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import subprocess

import joblib
import numpy as np
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .build_dataset_v3 import write_json
from .experiment_v3 import calculate_metrics, fit_calibration, grouped_metrics, load_arrays, make_dataset, predict_logits, read_jsonl, training_weights, verify_dataset
from .preprocessing_v3 import ROOT, VERSION, digest, file_digest, load_config


def train(config, run_id=None):
    import tensorflow as tf
    from .model_v3 import build_model

    tf.config.threading.set_intra_op_parallelism_threads(config["cpu_threads"])
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    directory = ROOT / config["dataset_dir"]
    dataset_manifest = verify_dataset(directory)
    if dataset_manifest["config"] != config:
        raise ValueError("Experiment config differs from the frozen dataset config.")
    vocabulary = json.loads((directory / "vocabulary.json").read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = run_id or f"v3-{dataset_manifest['dataset_id'][:8]}-{stamp}"
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run-id must be a directory name, not a path")
    model_dir = ROOT / config["models_dir"] / run_id
    result_dir = ROOT / config["results_dir"] / run_id
    model_dir.mkdir(parents=True, exist_ok=False)
    result_dir.mkdir(parents=True, exist_ok=False)
    source_hashes = {p.name: file_digest(p) for p in Path(__file__).parent.glob("*v3.py")}
    experiment = {"run_id": run_id, "started_utc": stamp, "dataset_id": dataset_manifest["dataset_id"], "config": config, "source_sha256": source_hashes, "environment": {name: metadata.version(name) for name in ["tensorflow", "keras", "numpy", "scikit-learn", "scipy"]}, "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "deterministic_operations": True, "selection_split": "validation", "calibration_split": "calibration", "test_used_for_selection": False, "status": "training"}
    write_json(result_dir / "experiment.json", experiment)
    train_arrays = load_arrays(directory, "train")
    validation_arrays = load_arrays(directory, "validation")
    train_records = read_jsonl(directory / "train.jsonl.gz")
    validation_records = read_jsonl(directory / "validation.jsonl.gz")
    weights = training_weights(train_records)
    seed_results = []

    class Progress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            values = {k: float(v) for k, v in (logs or {}).items()}
            print(json.dumps({"seed": seed, "epoch": epoch + 1, **values}), flush=True)

    for seed in config["seeds"]:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        model = build_model(config, len(vocabulary))
        checkpoint = model_dir / f"seed-{seed}.keras"
        print(f"Training seed={seed}, parameters={model.count_params()}, complete contracts={len(train_arrays['labels'])}", flush=True)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=config["early_stopping_patience"], restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
            tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_loss", save_best_only=True),
            tf.keras.callbacks.CSVLogger(result_dir / f"seed-{seed}-epochs.csv"),
            Progress(),
        ]
        history = model.fit(make_dataset(train_arrays, config, weights=weights, training=True, seed=seed), validation_data=make_dataset(validation_arrays, config), epochs=config["epochs"], callbacks=callbacks, verbose=0)
        write_json(result_dir / f"seed-{seed}-history.json", {k: [float(v) for v in values] for k, values in history.history.items()})
        # Reload the exact checkpoint that will be deployed/evaluated.
        model = tf.keras.models.load_model(checkpoint, compile=False)
        logits = predict_logits(model, validation_arrays, config)
        scores = calculate_metrics(validation_arrays["labels"], expit(logits), 0.5)
        seed_results.append({"seed": seed, "model": str(checkpoint.relative_to(ROOT)).replace("\\", "/"), "best_epoch": int(np.argmin(history.history["val_loss"])) + 1, "epochs_executed": len(history.history["loss"]), "parameters": model.count_params(), "validation": scores})
        write_json(result_dir / "seed_results.json", seed_results)
    selected = min(seed_results, key=lambda run: run["validation"]["log_loss"])
    model = tf.keras.models.load_model(ROOT / selected["model"], compile=False)

    print("Training full-code TF-IDF logistic baseline (training partition only)", flush=True)
    vectorizer = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)
    train_text = [" ".join(r["tokens"]) for r in train_records]
    baseline_x = vectorizer.fit_transform(train_text)
    baseline = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=config["split_seed"])
    baseline.fit(baseline_x, train_arrays["labels"], sample_weight=weights)
    joblib.dump({"vectorizer": vectorizer, "classifier": baseline}, model_dir / "tfidf_baseline.pkl")
    del baseline_x, train_text

    calibration_arrays = load_arrays(directory, "calibration")
    calibration_records = read_jsonl(directory / "calibration.jsonl.gz")
    calibration = fit_calibration(calibration_arrays["labels"], predict_logits(model, calibration_arrays, config), config["target_recall"])
    baseline_calibration_logits = baseline.decision_function(vectorizer.transform([" ".join(r["tokens"]) for r in calibration_records]))
    baseline_calibration = fit_calibration(calibration_arrays["labels"], baseline_calibration_logits, config["target_recall"])
    frozen = {"run_id": run_id, "dataset_id": dataset_manifest["dataset_id"], "preprocessing_version": VERSION, "config": config, "selected": selected, "model_path": selected["model"], "model_sha256": file_digest(ROOT / selected["model"]), "vocabulary_path": str((directory / "vocabulary.json").relative_to(ROOT)).replace("\\", "/"), "vocabulary_sha256": file_digest(directory / "vocabulary.json"), "calibration": calibration, "label_scope": "prediction_of_source_annotations_not_security_certification", "calibration_path": str((result_dir / "calibration.json").relative_to(ROOT)).replace("\\", "/"), "results_path": str(result_dir.relative_to(ROOT)).replace("\\", "/"), "dataset_path": str(directory.relative_to(ROOT)).replace("\\", "/")}
    write_json(result_dir / "calibration.json", {"neural": calibration, "baseline": baseline_calibration})
    write_json(model_dir / "bundle.json", frozen)
    print("Selection and calibration frozen. Evaluating test and source holdout.", flush=True)
    evaluated = {}
    for split in ("test", "source_holdout"):
        arrays = load_arrays(directory, split)
        records = read_jsonl(directory / (split + ".jsonl.gz"))
        if not records:
            evaluated[split] = {"samples": 0, "status": "unavailable"}
            continue
        logits = predict_logits(model, arrays, config)
        probabilities = expit(logits / calibration["temperature"])
        baseline_probabilities = expit(baseline.decision_function(vectorizer.transform([" ".join(r["tokens"]) for r in records])) / baseline_calibration["temperature"])
        evaluated[split] = {"neural_raw": calculate_metrics(arrays["labels"], expit(logits), 0.5), "neural_calibrated": calculate_metrics(arrays["labels"], probabilities, calibration["threshold"]), "tfidf_baseline": calculate_metrics(arrays["labels"], baseline_probabilities, baseline_calibration["threshold"]), "by_group": grouped_metrics(records, probabilities, calibration["threshold"])}
        write_json(result_dir / f"{split}_predictions.json", [{"sample_id": record["sample_id"], "group_id": record["group_id"], "true_label": int(label), "probability_vulnerable": float(probability), "predicted_label": int(probability >= calibration["threshold"])} for record, label, probability in zip(records, arrays["labels"], probabilities)])
    write_json(result_dir / "evaluation.json", evaluated)
    experiment.update(status="complete", selected_seed=selected["seed"], validation_f1_mean=float(np.mean([r["validation"]["f1_macro"] for r in seed_results])), validation_f1_std=float(np.std([r["validation"]["f1_macro"] for r in seed_results])), completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(result_dir / "experiment.json", experiment)
    # Activation is last: the app never points to a partially trained bundle.
    active_path = ROOT / "models/active_model.json"
    temporary = active_path.with_suffix(".json.tmp")
    write_json(temporary, frozen)
    temporary.replace(active_path)
    print(json.dumps({"run_id": run_id, "selected_seed": selected["seed"], "evaluation": {split: value.get("neural_calibrated", {}).get("accuracy") for split, value in evaluated.items()}, "active_model": str(active_path)}, indent=2), flush=True)
    return frozen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    train(load_config(args.config), args.run_id)
