"""Compare the actual CNN-BiLSTM, TF-IDF/XGBoost and a TCN trained from scratch.

Each model is selected on validation before calibration/test is read. Historic
experiments and the production manifest are never rewritten. Missing results
remain missing, including in the publication report.
"""

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.artifacts import read_jsonl, verify_dataset
from src.data.batching import training_weights
from src.evaluation.comparison_metrics import bootstrap_delta, evaluate_logits, fit_operating_points
from src.evaluation.metrics import calculate_metrics, grouped_metrics
from src.preprocessing import ROOT, digest, file_digest
from src.provenance import source_path

SPLITS = ("validation", "calibration", "test", "source_holdout")
NAMES = {
    "cnn_bilstm": "CNN–BiLSTM actif",
    "xgboost": "TF-IDF + XGBoost",
    "tcn": "TCN depuis zéro",
    "codebert": "CodeBERT fine-tuné (archive)",
}


def save_json(path, value):
    """A power interruption must not leave a partial completion marker."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config["threshold_fit_split"] != "calibration" or config["production_activation"]:
        raise ValueError("This benchmark requires separate calibration and cannot activate a model")
    return config


def initialize(config, run_id):
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run-id must be a single directory name")
    verify = verify_dataset(ROOT / config["dataset_dir"])
    if verify["dataset_id"] != config["dataset_id"]:
        raise ValueError("Unexpected benchmark dataset")
    result, models = ROOT / "results/benchmark" / run_id, ROOT / "models/benchmark" / run_id
    result.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    path = result / "protocol.json"
    source_paths = [
        source_path(name)
        for name in (
            "run_model_benchmark.py",
            "comparison_metrics.py",
            "metrics.py",
            "artifacts.py",
            "batching.py",
            "datasets.py",
            "preprocessing.py",
            "model.py",
            "provenance.py",
        )
    ]
    if "tcn" in config["models"]:
        source_paths.append(source_path("benchmark_tcn.py"))
    if "codebert" in config["models"]:
        source_paths.append(source_path("benchmark_codebert.py"))
    sources = {p.name: file_digest(p) for p in source_paths}
    active = file_digest(ROOT / config["cnn_bilstm"]["manifest"])
    if path.exists():
        protocol = json.loads(path.read_text(encoding="utf-8"))
        if (
            protocol["config"] != config
            or protocol["sources_sha256"] != sources
            or protocol["active_manifest_sha256"] != active
        ):
            raise ValueError("Frozen experiment changed; choose a new run-id")
    else:
        protocol = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "sources_sha256": sources,
            "active_manifest_sha256": active,
            "environment": {
                p: metadata.version(p)
                for p in (
                    "numpy",
                    "scipy",
                    "scikit-learn",
                    "xgboost",
                    "torch",
                    "transformers",
                    "tensorflow",
                    "keras",
                )
            },
            "test_previously_observed": True,
            "production_activation": False,
        }
        protocol["protocol_id"] = digest(protocol)
        save_json(path, protocol)
    return result, models


def records(config, split):
    return read_jsonl(ROOT / config["dataset_dir"] / (split + ".jsonl.gz"))


def texts(rows):
    return [" ".join(r["tokens"]) for r in rows]


def vectorizer(config):
    settings = config["tfidf"]
    if settings["fit_split"] != "train":
        raise ValueError("TF-IDF can only fit on training contracts")
    return TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        ngram_range=tuple(settings["ngram_range"]),
        max_features=settings["max_features"],
        min_df=settings["min_df"],
        sublinear_tf=settings["sublinear_tf"],
        dtype=np.float32,
    )


def select_seed(runs):
    """One checkpoint per family: no test/calibration metric enters selection."""
    return min(runs, key=lambda r: (r["validation"]["log_loss"], r["seed"]))


def import_baselines(config, result, source):
    """Reuse completed computations, preserving files and original provenance."""
    source = Path(source).resolve()
    if source == result.resolve():
        raise ValueError("Baseline source and destination must differ")
    protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    keys = (
        "dataset_id",
        "dataset_dir",
        "seeds",
        "selection",
        "threshold_fit_split",
        "target_recall",
        "target_fpr",
        "test_role",
        "cnn_bilstm",
        "tfidf",
        "xgboost",
        "production_activation",
    )
    if any(protocol["config"][key] != config[key] for key in keys):
        raise ValueError("Imported baselines do not match the current protocol")
    if protocol["active_manifest_sha256"] != file_digest(ROOT / config["cnn_bilstm"]["manifest"]):
        raise ValueError("Production CNN changed since baseline computation")
    for filename, expected in protocol["sources_sha256"].items():
        snapshot = source / "source_snapshot" / filename
        if not snapshot.exists() or file_digest(snapshot) != expected:
            raise ValueError("Original baseline source snapshot is missing or changed")
    checksums = {}
    for name in ("cnn_bilstm", "xgboost"):
        evaluation = json.loads((source / name / "evaluation.json").read_text(encoding="utf-8"))
        selected = evaluation["selected"]
        if file_digest(ROOT / selected["model_path"]) != selected["model_sha256"]:
            raise ValueError("Imported model changed")
        if name == "xgboost":
            selection = json.loads((source / name / "selection.json").read_text(encoding="utf-8"))
            for run in selection["runs"]:
                for artifact in ("model", "vectorizer"):
                    if file_digest(ROOT / run[artifact + "_path"]) != run[artifact + "_sha256"]:
                        raise ValueError("Imported XGBoost artifact changed")
        for path in sorted((source / name).rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            destination = result / relative
            expected = file_digest(path)
            if destination.exists() and file_digest(destination) != expected:
                raise ValueError("Refusing to replace different baseline results")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copyfile(path, destination)
            checksums[relative.as_posix()] = expected
    provenance = {
        "source_run": source.relative_to(ROOT).as_posix(),
        "source_protocol_sha256": file_digest(source / "protocol.json"),
        "files_sha256": checksums,
        "weights_duplicated": False,
        "reason": "Third model changed to TCN; CNN and XGBoost settings, scores and weights unchanged.",
    }
    save_json(result / "imported_baselines.json", provenance)
    return provenance


def train_xgboost(config, result, models):
    from xgboost import XGBClassifier

    selection_path = result / "xgboost" / "selection.json"
    if selection_path.exists():
        return json.loads(selection_path.read_text(encoding="utf-8"))
    train_rows, val_rows = records(config, "train"), records(config, "validation")
    extractor = vectorizer(config)
    started = time.perf_counter()
    train_x = extractor.fit_transform(texts(train_rows))
    val_x = extractor.transform(texts(val_rows))
    model_root = models / "xgboost"
    model_root.mkdir(parents=True, exist_ok=True)
    feature_path = model_root / "tfidf.pkl"
    joblib.dump(extractor, feature_path)
    fit_seconds = time.perf_counter() - started
    labels = np.asarray([r["label"] for r in train_rows])
    val_labels = np.asarray([r["label"] for r in val_rows])
    weights = training_weights(train_rows)
    runs = []
    for seed in config["seeds"]:
        run_dir = result / "xgboost" / f"seed-{seed}"
        marker = run_dir / "training.json"
        if marker.exists():
            runs.append(json.loads(marker.read_text(encoding="utf-8")))
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        model = XGBClassifier(**config["xgboost"], random_state=seed, n_jobs=config["cpu_threads"])
        started = time.perf_counter()
        model.fit(
            train_x, labels, sample_weight=weights, eval_set=[(val_x, val_labels)], verbose=False
        )
        elapsed = time.perf_counter() - started
        # XGBoost uses trees through best_iteration automatically with this API.
        logits = model.predict(val_x, output_margin=True).astype(np.float64)
        path = model_root / f"seed-{seed}.ubj"
        model.save_model(path)
        np.save(run_dir / "validation_logits.npy", logits)
        run = {
            "architecture": "xgboost",
            "seed": seed,
            "best_iteration": int(model.best_iteration),
            "model_path": path.relative_to(ROOT).as_posix(),
            "model_sha256": file_digest(path),
            "vectorizer_path": feature_path.relative_to(ROOT).as_posix(),
            "vectorizer_sha256": file_digest(feature_path),
            "training_seconds": elapsed,
            "tfidf_fit_seconds_shared": fit_seconds,
            "validation": calculate_metrics(val_labels, expit(logits), 0.5),
        }
        save_json(marker, run)
        save_json(run_dir / "history.json", model.evals_result())
        runs.append(run)
        print(
            json.dumps(
                {
                    "model": "xgboost",
                    "seed": seed,
                    "best_iteration": model.best_iteration,
                    "training_seconds": elapsed,
                }
            ),
            flush=True,
        )
    selected = select_seed(runs)
    selection = {
        "criterion": config["selection"],
        "fit_split": "validation",
        "selected": selected,
        "runs": runs,
    }
    save_json(selection_path, selection)
    return selection


def predict_xgboost(run, rows):
    from xgboost import XGBClassifier

    for key in ("model", "vectorizer"):
        if file_digest(ROOT / run[key + "_path"]) != run[key + "_sha256"]:
            raise ValueError("XGBoost artifact changed")
    extractor = joblib.load(ROOT / run["vectorizer_path"])
    model = XGBClassifier()
    model.load_model(ROOT / run["model_path"])
    return model.predict(extractor.transform(texts(rows)), output_margin=True).astype(np.float64)


def evaluate_model(config, result, name, selected, predict):
    marker = result / name / "evaluation.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    calibration_rows = records(config, "calibration")
    calibration_logits = predict(calibration_rows, "calibration")
    calibration = fit_operating_points(
        [r["label"] for r in calibration_rows], calibration_logits, config
    )
    save_json(result / name / "calibration.json", calibration)
    output = {"model": name, "selected": selected, "calibration": calibration, "splits": {}}
    for split in ("test", "source_holdout"):
        rows = records(config, split)
        started = time.perf_counter()
        logits = predict(rows, split)
        inference_seconds = time.perf_counter() - started
        if len(logits) != len(rows) or not np.isfinite(logits).all():
            raise ValueError("Missing/nonfinite contract predictions")
        np.save(result / name / (split + "_logits.npy"), logits)
        save_json(result / name / (split + "_sample_ids.json"), [r["sample_id"] for r in rows])
        measured = evaluate_logits([r["label"] for r in rows], logits, calibration)
        measured["by_annotation"] = grouped_metrics(
            rows, expit(logits / calibration["temperature"]), calibration["threshold"]
        )
        measured["batch_inference_seconds"] = inference_seconds
        measured["timing_note"] = (
            "batch_timing_includes_model_load_and_preprocessing_not_per_contract_latency"
            if name == "xgboost"
            else "batch_timing_excludes_model_load_not_per_contract_latency"
        )
        output["splits"][split] = measured
    save_json(marker, output)
    return output


def evaluate_cnn(config, result):
    """Reevaluate the existing checkpoint, without retraining or activating it."""
    marker = result / "cnn_bilstm" / "evaluation.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    import tensorflow as tf

    from src.data.artifacts import load_arrays
    from src.training import model as registered_layers  # noqa: F401
    from src.training.datasets import predict_logits

    bundle = json.loads((ROOT / config["cnn_bilstm"]["manifest"]).read_text(encoding="utf-8"))
    if (
        bundle["dataset_id"] != config["dataset_id"]
        or bundle["selected"]["seed"] != config["cnn_bilstm"]["selected_seed"]
    ):
        raise ValueError("CNN reference changed")
    if file_digest(ROOT / bundle["model_path"]) != bundle["model_sha256"]:
        raise ValueError("CNN weights changed")
    tf.config.threading.set_intra_op_parallelism_threads(config["cpu_threads"])
    tf.config.threading.set_inter_op_parallelism_threads(1)
    model = tf.keras.models.load_model(ROOT / bundle["model_path"], compile=False)
    selected = {
        "model_path": bundle["model_path"],
        "model_sha256": bundle["model_sha256"],
        "seed": bundle["selected"]["seed"],
        "validation": bundle["selected"]["validation"],
        "strategy": config["cnn_bilstm"]["strategy"],
        "training_run": bundle["run_id"],
    }
    save_json(
        result / "cnn_bilstm" / "selection.json",
        {"selected": selected, "fit_split": "validation", "historically_selected": True},
    )

    def predict(rows, split):
        return predict_logits(
            model, load_arrays(ROOT / config["dataset_dir"], split), bundle["config"]
        )

    evaluation = evaluate_model(config, result, "cnn_bilstm", selected, predict)
    tf.keras.backend.clear_session()
    return evaluation


def codebert_stage(config, result, models, pilot):
    import torch

    from src.experiments.benchmark.benchmark_codebert import (
        load_encoder,
        predict_contracts,
        tokenize_contract,
        train_codebert,
    )

    engine_path = source_path("benchmark_codebert.py")
    source_marker = result / "codebert" / "source.json"
    source = {"sha256": file_digest(engine_path), "revision": config["codebert"]["revision"]}
    if source_marker.exists() and json.loads(source_marker.read_text(encoding="utf-8")) != source:
        raise ValueError("CodeBERT engine changed; choose a new run-id")
    save_json(source_marker, source)
    torch.set_num_threads(config["cpu_threads"])
    train_rows, validation_rows = records(config, "train"), records(config, "validation")
    weights = training_weights(train_rows)
    if pilot:
        report = train_codebert(
            config,
            train_rows,
            validation_rows,
            weights,
            result / "codebert",
            models / "codebert",
            config["seeds"][0],
            pilot=True,
        )
        save_json(result / "codebert" / "pilot.json", report)
        return
    runs = []
    for seed in config["seeds"]:
        run_dir, model_dir = (
            result / "codebert" / f"seed-{seed}",
            models / "codebert" / f"seed-{seed}",
        )
        marker = run_dir / "training.json"
        if marker.exists():
            runs.append(json.loads(marker.read_text(encoding="utf-8")))
        else:
            run = train_codebert(
                config, train_rows, validation_rows, weights, run_dir, model_dir, seed, pilot=False
            )
            save_json(marker, run)
            runs.append(run)
    selected = select_seed(runs)
    save_json(
        result / "codebert" / "selection.json",
        {
            "selected": selected,
            "runs": runs,
            "fit_split": "validation",
            "criterion": config["selection"],
        },
    )
    model, tokenizer = load_encoder(config["codebert"])
    model.load_state_dict(
        torch.load(ROOT / selected["model_path"], map_location="cpu", weights_only=True)
    )

    def predict(rows, split):
        encoded = [
            tokenize_contract(r["tokens"], tokenizer, config["codebert"]["chunk_subtokens"])
            for r in rows
        ]
        return predict_contracts(model, encoded, config["codebert"]["chunk_batch_size"])

    evaluate_model(config, result, "codebert", selected, predict)


def tcn_stage(config, result, models, pilot=False):
    import torch

    from src.data.artifacts import load_arrays
    from src.experiments.benchmark.benchmark_tcn import load_tcn, predict_tcn, train_tcn

    if "tcn" not in config["models"]:
        raise ValueError("TCN is not part of this study")
    torch.set_num_threads(config["cpu_threads"])
    directory = ROOT / config["dataset_dir"]
    vocabulary_size = len(json.loads((directory / "vocabulary.json").read_text(encoding="utf-8")))
    train_rows = records(config, "train")
    train_arrays, validation_arrays = (
        load_arrays(directory, "train"),
        load_arrays(directory, "validation"),
    )
    if not np.array_equal(train_arrays["labels"], [r["label"] for r in train_rows]):
        raise ValueError("Training labels and encoded contracts differ")
    weights = training_weights(train_rows)
    result_dir, model_dir = result / "tcn", models / "tcn"
    if pilot:
        report = train_tcn(
            config,
            train_arrays,
            validation_arrays,
            weights,
            result_dir,
            model_dir,
            config["seeds"][0],
            vocabulary_size,
            pilot=True,
        )
        save_json(result_dir / "pilot.json", report)
        return report
    runs = []
    for seed in config["seeds"]:
        marker = result_dir / f"seed-{seed}" / "training.json"
        if marker.exists():
            run = json.loads(marker.read_text(encoding="utf-8"))
        else:
            run = train_tcn(
                config,
                train_arrays,
                validation_arrays,
                weights,
                result_dir / f"seed-{seed}",
                model_dir / f"seed-{seed}",
                seed,
                vocabulary_size,
            )
            save_json(marker, run)
        if file_digest(ROOT / run["model_path"]) != run["model_sha256"]:
            raise ValueError("TCN checkpoint changed")
        runs.append(run)
    selected = select_seed(runs)
    save_json(
        result_dir / "selection.json",
        {
            "selected": selected,
            "runs": runs,
            "fit_split": "validation",
            "criterion": config["selection"],
        },
    )
    model = load_tcn(config["tcn"], vocabulary_size, ROOT / selected["model_path"])
    return evaluate_model(
        config,
        result,
        "tcn",
        selected,
        lambda rows, split: predict_tcn(model, load_arrays(directory, split), config["tcn"]),
    )


def paired_comparisons(config, result, evaluations):
    """Cluster bootstrap conditional on validation-selected checkpoints."""
    path = result / "paired_comparisons.json"
    fingerprint = digest(
        {name: file_digest(result / name / "evaluation.json") for name in evaluations}
    )
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["evaluations_id"] != fingerprint:
            raise ValueError("Paired comparison inputs changed")
        return saved
    rows = records(config, "test")
    labels, groups = [r["label"] for r in rows], [r["group_id"] for r in rows]
    sample_ids = [r["sample_id"] for r in rows]
    probabilities, thresholds = {}, {}
    for name, evaluation in evaluations.items():
        saved_ids = json.loads((result / name / "test_sample_ids.json").read_text(encoding="utf-8"))
        if saved_ids != sample_ids:
            raise ValueError("Paired models do not have the same test order")
        logits = np.load(result / name / "test_logits.npy", allow_pickle=False)
        calibration = evaluation["calibration"]
        probabilities[name] = expit(logits / calibration["temperature"])
        thresholds[name] = calibration["threshold"]
    pairs = []
    names = config["models"]
    for index, right in enumerate(names):
        for left in names[index + 1 :]:
            differences = bootstrap_delta(
                labels,
                groups,
                [probabilities[left]],
                [thresholds[left]],
                [probabilities[right]],
                [thresholds[right]],
                config.get("bootstrap_repetitions", 1000),
                config.get("bootstrap_seed", 42),
            )
            pairs.append({"left": left, "right": right, **differences})
    output = {
        "evaluations_id": fingerprint,
        "repetitions": config.get("bootstrap_repetitions", 1000),
        "seed": config.get("bootstrap_seed", 42),
        "sampling_unit": "group_id",
        "interpretation": "left minus right; selected checkpoints fixed; exploratory; no multiple-comparison correction",
        "pairs": pairs,
    }
    save_json(path, output)
    return output


def write_report(config, result):
    evaluations = {}
    for name in config["models"]:
        path = result / name / "evaluation.json"
        if path.exists():
            evaluations[name] = json.loads(path.read_text(encoding="utf-8"))
    complete = len(evaluations) == len(config["models"])
    summary = {
        "complete": complete,
        "test_role": config["test_role"],
        "models": {},
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    title = "# " + ", ".join(NAMES[name] for name in config["models"])
    lines = [
        title,
        "",
        "**État : "
        + (
            "trois modèles évalués."
            if complete
            else "comparaison incomplète ; aucun classement final."
        )
        + "**",
        "",
        "Les résultats proviennent des mêmes partitions SMART BUG. Le test a déjà été consulté ; cette étude reste exploratoire. Aucun score publié d'un autre corpus n'entre dans ce tableau.",
        "",
        "Un checkpoint par famille est sélectionné sur validation. Le CNN est le checkpoint TensorFlow actif, pas le réseau PyTorch de l'ancienne étude. Les nouveaux modèles ont trois graines ; les métriques principales concernent leur checkpoint sélectionné, pas une moyenne mélangée au CNN actif.",
        "",
        "| Modèle | État | F1 macro | Précision positive | Rappel positif | FPR | PR-AUC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name in config["models"]:
        if name not in evaluations:
            pilot_path = result / name / "pilot.json"
            status = (
                "Pilote terminé ; entraînement complet à lancer"
                if pilot_path.exists()
                else "Non évalué"
            )
            lines.append(f"| {NAMES[name]} | {status} | — | — | — | — | — |")
            summary["models"][name] = {"status": status, "metrics": None}
            continue
        ev = evaluations[name]
        point = ev["splits"]["test"]["calibrated"]
        numbers = [
            point[k]
            for k in (
                "f1_macro",
                "precision_vulnerable",
                "recall_vulnerable",
                "false_positive_rate",
                "average_precision",
            )
        ]
        lines.append(
            "| " + NAMES[name] + " | Évalué | " + " | ".join(f"{v:.4f}" for v in numbers) + " |"
        )
        summary["models"][name] = {
            "status": "evaluated",
            "metrics": point,
            "selected": ev["selected"],
        }
    lines += [
        "",
        "Les températures et les seuils sont ajustés exclusivement sur calibration, avec une cible de rappel de 90 %. La cible ne garantit pas 90 % sur le test. Le seuil alternatif visant FPR ≤ 10 % sur calibration figure dans les JSON d'évaluation.",
        "",
        "XGBoost utilise TF-IDF appris uniquement sur train (1–2 grammes, 20 000 caractéristiques max.). Le TCN est entraîné depuis zéro sur tous les tokens encodés, avec des convolutions causales dilatées résiduelles puis une moyenne et un maximum masqués. La perte porte sur le contrat entier.",
        "",
        "## Variabilité des nouveaux modèles",
        "",
        "Les scores ci-dessous sont ceux de validation avant calibration ; ils documentent la sélection, sans utiliser le test pour choisir une graine.",
        "",
        "| Modèle | Graine | Log-loss validation | F1 macro validation à 0,5 |",
        "|---|---:|---:|---:|",
    ]
    for name in config["models"]:
        if name == "cnn_bilstm":
            continue
        selection_path = result / name / "selection.json"
        if selection_path.exists():
            for run in json.loads(selection_path.read_text(encoding="utf-8"))["runs"]:
                lines.append(
                    f"| {NAMES[name]} | {run['seed']} | {run['validation']['log_loss']:.4f} | {run['validation']['f1_macro']:.4f} |"
                )
    if complete:
        paired = paired_comparisons(config, result, evaluations)
        summary["paired_comparisons"] = paired
        lines += [
            "",
            "## Différences sur les mêmes contrats",
            "",
            "Bootstrap apparié par groupe (1 000 rééchantillonnages), conditionné aux checkpoints sélectionnés. Ces intervalles ne mesurent pas toute la variabilité d'un nouvel entraînement et ne sont pas corrigés pour les comparaisons multiples.",
            "",
            "| Différence (gauche − droite) | F1 macro, points | IC 95 % du F1, points |",
            "|---|---:|---:|",
        ]
        for pair in paired["pairs"]:
            delta = pair["f1_macro"]
            low, high = (100 * value for value in delta["ci95"])
            lines.append(
                f"| {NAMES[pair['left']]} − {NAMES[pair['right']]} | {100 * delta['difference']:+.2f} | [{low:+.2f}, {high:+.2f}] |"
            )
    lines += [
        "",
        "## Limites",
        "",
        "Les cibles sont des annotations historiques, en partie issues d'analyseurs statiques. La classe 0 ne certifie pas l'absence de vulnérabilité. La source réservée contient seulement six négatifs : son taux de faux positifs est trop fragile pour une conclusion générale.",
        "",
        "CNN–BiLSTM et TCN conservent l'ordre dans leurs représentations ; TF-IDF le réduit aux n-grammes. Les identifiants sont normalisés pour les trois modèles. Le TCN voit toute la séquence, mais chaque position a une portée de 2 045 tokens encodés ; le pooling global ne lui donne pas une capacité illimitée à modéliser les relations éloignées. Les budgets d'entraînement et les tailles de modèles diffèrent et sont archivés explicitement.",
        "",
        "Les durées d'inférence par lot ne sont pas des latences unitaires : XGBoost inclut le rechargement de ses artefacts, CNN et TCN chargent leurs poids avant le chronométrage. Ces chiffres ne servent pas à comparer la vitesse des modèles. Les expériences historiques et le pilote CodeBERT restent archivés et ne constituent plus la comparaison principale du mémoire.",
        "",
        "## Références",
        "",
        "- Bai, Kolter et Koltun (2018), [TCN et modélisation de séquences](https://arxiv.org/abs/1803.01271).",
        "- Gopali et al. (2022), [TCN pour les vulnérabilités de smart contracts](https://doi.org/10.1109/COMPSAC54236.2022.00197) ; notre entrée Solidity est une adaptation du travail sur opcodes.",
        "- Chen et Guestrin (2016), [XGBoost](https://arxiv.org/abs/1603.02754).",
        "",
    ]
    save_json(result / "summary.json", summary)
    (result / "report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/benchmark_models.json")
    parser.add_argument("--run-id", default="cnn-xgboost-tcn-20261005")
    parser.add_argument(
        "--stage",
        choices=(
            "prepare",
            "import-baselines",
            "cnn",
            "xgboost",
            "tcn-pilot",
            "tcn",
            "codebert-pilot",
            "codebert",
            "report",
            "all",
        ),
        default="prepare",
    )
    parser.add_argument("--baseline-run-id", default="cnn-xgboost-codebert-20261005-final")
    args = parser.parse_args()
    config = load_config(args.config)
    result, models = initialize(config, args.run_id)
    if args.stage == "import-baselines":
        if Path(args.baseline_run_id).name != args.baseline_run_id or args.baseline_run_id in (
            ".",
            "..",
        ):
            raise ValueError("baseline-run-id must be a single directory name")
        import_baselines(config, result, ROOT / "results/benchmark" / args.baseline_run_id)
    if args.stage in ("cnn", "all"):
        evaluate_cnn(config, result)
        write_report(config, result)
    if args.stage in ("xgboost", "all"):
        selection = train_xgboost(config, result, models)
        evaluate_model(
            config,
            result,
            "xgboost",
            selection["selected"],
            lambda rows, split: predict_xgboost(selection["selected"], rows),
        )
        write_report(config, result)
    if args.stage in ("tcn-pilot", "tcn") or (args.stage == "all" and "tcn" in config["models"]):
        tcn_stage(config, result, models, pilot=args.stage == "tcn-pilot")
    if args.stage in ("codebert-pilot", "codebert") or (
        args.stage == "all" and "codebert" in config["models"]
    ):
        if "codebert" not in config["models"]:
            raise ValueError(
                "CodeBERT is archived; use its frozen configuration in a new experiment"
            )
        codebert_stage(config, result, models, pilot=args.stage == "codebert-pilot")
    summary = write_report(config, result)
    print(
        json.dumps({"report": str(result / "report.md"), "complete": summary["complete"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
