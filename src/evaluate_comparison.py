"""Evaluate frozen candidates, calibrate outside test, and write the comparison."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import time

import joblib
import numpy as np
from scipy.special import expit
import torch

from .compare_models import freeze_selection, initialize, predict_features, predict_neural
from .comparison_data import collate, load_study_config, prepare_inputs, save_json, study_arrays
from .comparison_metrics import bootstrap_delta, evaluate_logits, fit_operating_points
from .comparison_models import ContractClassifier, FeatureClassifier
from .experiment_v3 import read_jsonl
from .preprocessing_v3 import ROOT, file_digest


class Predictor:
    def __init__(self, run, config, cache):
        self.run, self.config, self.cache = run, config, cache
        self.mode = run["input_mode"]
        self.architecture = run["architecture"]
        self.path = ROOT / run["model_path"]
        if file_digest(self.path) != run["model_sha256"]:
            raise ValueError("Checkpoint integrity failure")
        if self.architecture == "tfidf":
            self.model = joblib.load(self.path)
        elif self.architecture == "codet5_frozen":
            feature_root = ROOT / ".cache-comparison/codet5"
            if file_digest(feature_root / "manifest.json") != run["feature_manifest_sha256"]:
                raise ValueError("Pretrained feature provenance changed")
            dimension = json.loads((feature_root / "manifest.json").read_text())["feature_dimension"]
            self.model = FeatureClassifier(dimension, config)
            self.scaler = joblib.load(self.path.parent / "scaler.pkl")
        else:
            vocabulary_size = json.loads((cache / "manifest.json").read_text())["vocabulary_size"]
            self.model = ContractClassifier(self.architecture, vocabulary_size, config)
        if self.architecture != "tfidf":
            self.model.load_state_dict(torch.load(self.path, map_location="cpu", weights_only=True))
            self.model.eval()

    def predict(self, split):
        if self.architecture == "tfidf":
            rows = read_jsonl(ROOT / self.config["dataset_dir"] / (split + ".jsonl.gz"))
            text = [" ".join(row["tokens"] if self.mode == "full" else row["tokens"][:512]) for row in rows]
            return self.model["classifier"].decision_function(self.model["vectorizer"].transform(text))
        if self.architecture == "codet5_frozen":
            root = ROOT / ".cache-comparison/codet5"
            metadata = json.loads((root / (split + "_extraction.json")).read_text())
            name = f"{split}_{self.mode}.npy"
            if file_digest(root / name) != metadata["sha256"][name]:
                raise ValueError("Pretrained feature corruption")
            features = self.scaler.transform(np.load(root / name)).astype(np.float32)
            return predict_features(self.model, features)
        return predict_neural(self.model, study_arrays(self.config, self.cache, split, self.mode), self.config)

    def latency(self):
        # Start from the same normalized lexical input for every architecture;
        # include each model's encoding/tokenization and all windows. Exclude
        # shared Solidity normalization, disk I/O, model load and cold startup.
        from .preprocessing_v3 import encode
        rows = read_jsonl(ROOT / self.config["dataset_dir"] / "validation.jsonl.gz")
        rng = np.random.default_rng(self.config["bootstrap_seed"])
        indices = rng.choice(len(rows), size=min(self.config["latency_samples"], len(rows)), replace=False)
        vocabulary = json.loads((ROOT / self.config["dataset_dir"] / "vocabulary.json").read_text())
        encoder = None
        if self.architecture == "codet5_frozen":
            from .comparison_pretrained import FrozenCodeEncoder
            encoder = FrozenCodeEncoder(self.config)
        def once(row):
            tokens = row["tokens"] if self.mode == "full" else row["tokens"][:512]
            if self.architecture == "tfidf":
                return self.model["classifier"].decision_function(self.model["vectorizer"].transform([" ".join(tokens)]))
            if encoder is not None:
                features, _, _ = encoder.encode_rows([row], [self.mode])
                return predict_features(self.model, self.scaler.transform(features[:, 0]).astype(np.float32))
            ids = torch.tensor([encode(tokens, vocabulary)], dtype=torch.long)
            with torch.inference_mode():
                return self.model(ids, torch.tensor([ids.shape[1]]))
        for index in indices[:3]:
            once(rows[index])
        elapsed = []
        for index in indices:
            started = time.perf_counter()
            once(rows[index])
            elapsed.append((time.perf_counter() - started) * 1000)
        return {"median_ms": float(np.median(elapsed)), "p95_ms": float(np.percentile(elapsed, 95)), "mean_ms": float(np.mean(elapsed)), "samples": len(indices), "sample_indices": indices.tolist(), "milliseconds": elapsed, "scope": "single contract, from normalized lexical tokens through encoding and logits; excludes shared Solidity normalization, I/O, loading, calibration; pretrained includes frozen encoder", "threads": torch.get_num_threads(), "representative_seed": self.run["seed"]}


def evaluate_run(run_path, config, cache, input_manifest, do_latency=False):
    run = json.loads(run_path.read_text())
    destination = run_path.parent
    predictor = Predictor(run, config, cache)
    fit_path = destination / "calibration.json"
    if fit_path.exists():
        calibration = json.loads(fit_path.read_text())
        if calibration["model_sha256"] != run["model_sha256"]:
            raise ValueError("Calibration model mismatch")
    else:
        logits = predictor.predict("calibration")
        labels = study_arrays(config, cache, "calibration", run["input_mode"])["labels"]
        calibration = {**fit_operating_points(labels, logits, config), "model_sha256": run["model_sha256"], "frozen_utc": datetime.now(timezone.utc).isoformat()}
        save_json(fit_path, calibration)
        np.save(destination / "calibration_logits.npy", logits)
    evaluation_path = destination / "evaluation.json"
    if not evaluation_path.exists():
        result = {"variant": run["variant"], "seed": run["seed"], "model_sha256": run["model_sha256"], "calibration_sha256": file_digest(fit_path), "evaluated_utc": datetime.now(timezone.utc).isoformat(), "splits": {}}
        for split in ("test", "source_holdout"):
            logits = predictor.predict(split)
            records = json.loads((cache / (split + "_metadata.json")).read_text())
            labels = np.array([row["label"] for row in records])
            probabilities = expit(logits / calibration["temperature"])
            np.savez_compressed(destination / (split + "_predictions.npz"), logits=logits, probabilities=probabilities, labels=labels, sample_ids=np.asarray([row["sample_id"] for row in records]))
            clean = np.ones(len(labels), dtype=bool)
            clean[input_manifest["cross_split_truncated_collisions"][split]["affected_indices"]] = False
            metrics = {"all": evaluate_logits(labels, logits, calibration), "collision_free": {"samples": int(clean.sum()), **evaluate_logits(labels[clean], logits[clean], calibration)}, "by_source": {}}
            sources = np.asarray([row["source"] for row in records])
            for source in sorted(set(sources)):
                mask = sources == source
                metrics["by_source"][source] = {"samples": int(mask.sum()), **evaluate_logits(labels[mask], logits[mask], calibration)}
            result["splits"][split] = metrics
        save_json(evaluation_path, result)
    latency_path = destination.parent / "latency.json" if run["architecture"] != "tfidf" else destination / "latency.json"
    if do_latency and not latency_path.exists():
        save_json(latency_path, predictor.latency())
    print(f"EVALUATED {run['variant']} seed {run['seed']}", flush=True)


def aggregate(config, result, selection, cache):
    rows, predictions = [], {}
    for candidate in selection["candidates"]:
        variant = candidate["variant"]
        runs = [json.loads((result / path).read_text()) for path in candidate["runs"]]
        evaluations = [json.loads((result / path).with_name("evaluation.json").read_text()) for path in candidate["runs"]]
        latency = json.loads((result / variant / "latency.json").read_text())
        row = {"variant": variant, "runs": len(runs), "parameters": runs[0]["parameters"], "trainable_parameters": runs[0]["trainable_parameters"], "training_seconds_mean": float(np.mean([run["training_seconds"] for run in runs])), "latency_median_ms": latency["median_ms"], "latency_p95_ms": latency["p95_ms"], **{key: value for key, value in candidate.items() if key.startswith("validation_")}}
        epoch_durations = []
        for relative in candidate["runs"]:
            history_path = (result / relative).with_name("history.json")
            if history_path.exists():
                history = json.loads(history_path.read_text())
                epoch_durations.extend(np.diff([0.0] + [entry["elapsed_seconds"] for entry in history]).tolist())
        row["epoch_wall_seconds_median"] = float(np.median(epoch_durations)) if epoch_durations else None
        row["epoch_wall_seconds_maximum"] = float(max(epoch_durations)) if epoch_durations else None
        row["timing_has_large_pause"] = bool(epoch_durations and max(epoch_durations) > 5 * np.median(epoch_durations))
        for subset in ("all", "collision_free"):
            for point in ("raw", "calibrated", "fpr_operating_point"):
                for metric in ("f1_macro", "recall_vulnerable", "false_positive_rate", "precision_vulnerable", "roc_auc", "brier_score", "ece_10_bins", "log_loss"):
                    values = [entry["splits"]["test"][subset][point][metric] for entry in evaluations]
                    prefix = f"test_{subset}_{point}_{metric}"
                    row[prefix + "_mean"] = float(np.mean(values))
                    row[prefix + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        row["test_roc_recall_at_fpr_0_10_mean"] = float(np.mean([entry["splits"]["test"]["all"]["roc_recall_at_fpr_0_10"] for entry in evaluations]))
        rows.append(row)
        probabilities, thresholds = [], []
        for path in candidate["runs"]:
            directory = (result / path).parent
            with np.load(directory / "test_predictions.npz", allow_pickle=False) as data:
                probabilities.append(data["probabilities"])
            thresholds.append(json.loads((directory / "calibration.json").read_text())["threshold"])
        predictions[variant] = (probabilities, thresholds)
    save_json(result / "summary.json", rows)
    with (result / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    records = json.loads((cache / "test_metadata.json").read_text())
    labels = np.asarray([row["label"] for row in records])
    groups = np.asarray([row["group_id"] for row in records])
    pairs = [(a + "_full", a + "_first512") for a in config["architectures"] + ["codet5_frozen", "tfidf"]]
    pairs += [("cnn_bilstm_full", "cnn_full"), ("cnn_bilstm_full", "bilstm_full"), ("codet5_frozen_full", "cnn_bilstm_full"), (selection["selected_variant"], "tfidf_full")]
    comparisons = []
    for left, right in pairs:
        p_a, t_a = predictions[left]
        p_b, t_b = predictions[right]
        comparisons.append({"left": left, "right": right, "interpretation": "left minus right; exploratory paired group bootstrap, seeds fixed, no multiple-comparison correction", **bootstrap_delta(labels, groups, p_a, t_a, p_b, t_b, config["bootstrap_repetitions"], config["bootstrap_seed"])})
    save_json(result / "paired_comparisons.json", comparisons)
    return rows, comparisons


def write_report(config, result, selection, rows, comparisons, inputs):
    selected = selection["selected_variant"]
    lines = ["# Comparaison exploratoire des modèles", "", f"Exécution : `{result.name}`. Variante retenue **sur validation** : **{selected}**. Aucun remplacement automatique de V3.", "", "## Résultats sur le test V3 déjà consulté", "", "Moyennes sur trois graines pour les réseaux ; une seule estimation déterministe pour TF-IDF. Seuils ajustés exclusivement sur la calibration. Ces résultats ne constituent pas une confirmation sur un nouveau jeu indépendant.", "", "| Variante | F1 macro | Rappel | Faux positifs | Rappel au seuil FPR cible 10 % | FPR réellement obtenu | ECE | Latence médiane (ms) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        prefix = "test_all_calibrated_"
        point = "test_all_fpr_operating_point_"
        lines.append(f"| {row['variant']} | {row[prefix+'f1_macro_mean']:.4f} ± {row[prefix+'f1_macro_std']:.4f} | {row[prefix+'recall_vulnerable_mean']:.4f} | {row[prefix+'false_positive_rate_mean']:.4f} | {row[point+'recall_vulnerable_mean']:.4f} | {row[point+'false_positive_rate_mean']:.4f} | {row[prefix+'ece_10_bins_mean']:.4f} | {row['latency_median_ms']:.2f} |")
    lines += ["", "La cible de 10 % est une contrainte sur la calibration ; le FPR test peut s'en écarter. La courbe ROC test contient aussi un rappel descriptif à 10 %, enregistré dans le CSV, qui n'est pas un seuil déployable choisi sur le test.", "", "## Ablations et incertitude", "", "Différences gauche moins droite de F1 macro, avec intervalle bootstrap à 95 % par groupe de contrats. Ces intervalles conditionnent sur les graines exécutées ; ils ne couvrent pas toute l'incertitude d'entraînement.", "", "| Comparaison | Différence F1 | IC 95 % |", "|---|---:|---:|"]
    for comparison in comparisons:
        score = comparison["f1_macro"]
        lines.append(f"| {comparison['left']} − {comparison['right']} | {score['difference']:+.4f} | [{score['ci95'][0]:+.4f}, {score['ci95'][1]:+.4f}] |")
    collision = inputs["cross_split_truncated_collisions"]["test"]["affected_samples"]
    lines += ["", "## Protocole et limites", "", f"- Test : {inputs['splits']['test']['samples']} contrats ; {collision} présentent un préfixe de 512 tokens identique à celui d'un exemple d'entraînement, validation ou calibration. Les scores du sous-ensemble commun sans ces collisions figurent dans `summary.csv` et chaque `evaluation.json`.", "- Les partitions, le vocabulaire et les poids d'échantillons proviennent de V3. Le vocabulaire et les scalers sont ajustés sur l'entraînement uniquement. Tous les contrats de la partition train sont vus à chaque époque.", f"- CNN, BiLSTM et CNN-BiLSTM sont entraînés de zéro sous PyTorch, avec embeddings {config['embedding_dim']}, fenêtres CNN {config['window_size']}, filtres {config['convolution_filters']}, noyau {config['convolution_kernel']}, unités LSTM {config['lstm_units']} par direction, dense {config['dense_units']}, dropout {config['dropout']}, Adam lr {config['learning_rate']}, maximum {config['epochs']} époques, arrêt anticipé {config['early_stopping_patience']} sur log-loss validation, gradient clip 1.0. Batches ≤ {config['batch_size']} contrats et budget {config['batch_token_budget']} positions remplies, sauf contrat individuel plus long.", "- CNN : convolution par fenêtre, moyenne/max par fenêtre puis sur toutes les fenêtres. BiLSTM : récurrence bidirectionnelle sur tous les tokens. CNN-BiLSTM : récurrence sur les représentations des fenêtres CNN. Les variantes complètes conservent tous les tokens ; les variantes tronquées gardent 512 tokens lexicaux avant encodage sans perte.", "- CodeT5-small : encodeur préentraîné figé, sans fine-tuning ; tous les sous-tokens sont répartis en fenêtres de 256 positions spéciales incluses. Moyenne pondérée et maximum global des états, standardisation sur train, tête dense entraînée. La normalisation des identifiants V3 diffère du code naturel vu au préentraînement. Ce résultat ne mesure donc pas le potentiel d'un CodeT5 finement ajusté au Solidity.", "- TF-IDF 1–2 grammes, 20 000 caractéristiques maximum, min_df=2, TF sous-linéaire et régression logistique C=1, liblinear. C'est une référence classique ; les réseaux ne sont pas présumés meilleurs.", "- Budget comparable en passages sur les données et hyperparamètres communs, pas en FLOPs ni temps mural. Les architectures ont des nombres de paramètres différents. Les trois graines CodeT5 ne réentraînent que la tête.", "- Température et deux seuils appris sur calibration : rappel cible 90 %, et FPR cible 10 %. Le choix de variante privilégie le rappel sur la ROC validation à FPR ≤ 10 %, puis F1 macro et log-loss validation.", "- La latence inclut l'encodage propre à chaque modèle et, pour CodeT5, l'encodeur préentraîné. Elle exclut chargement des poids, disque et normalisation Solidity commune. Mesure séquentielle sur les mêmes contrats de validation, après échauffement.", "- Labels issus des sources historiques, parfois faibles ; le holdout de source contient seulement six négatifs. Ses taux de faux positifs sont très incertains. Les sous-groupes et matrices de confusion sont disponibles dans les JSON.", "- Une évaluation finale sur des contrats indépendants, avec labels vérifiés et séparation par familles/projets, reste nécessaire avant d'affirmer une supériorité générale ou de modifier le modèle actif.", "", "## Fichiers reproductibles", "", "- `config/comparison_v1.json` : hyperparamètres et décisions avant entraînement.", "- `src/compare_models.py` : entraînements, sélection, checkpoints et historiques.", "- `src/comparison_pretrained.py` : téléchargement à révision figée et extraction complète des représentations CodeT5.", "- `src/evaluate_comparison.py` : calibration, test, latences et rapport.", "- `protocol.json`, `selection.json`, `summary.csv`, `paired_comparisons.json` : provenance et résultats détaillés.", ""]
    lines += ["## Interprétation des temps", "", "Les durées d'entraînement sont des temps muraux, qui incluent les pauses et la suspension éventuelle de la machine. Une longue interruption a été observée pendant l'exécution ; ces totaux ne mesurent donc pas seuls l'efficacité des architectures. Le CSV fournit aussi les durées médiane et maximale par époque et signale les pauses supérieures à cinq fois la médiane. Les latences sont mesurées séparément après l'entraînement.", "", "Pour CodeT5, le temps d'entraînement de la tête exclut l'extraction préalable de l'encodeur : les durées d'extraction par partition sont conservées dans `pretrained_extraction.json`. Les entrées complètes et tronquées partagent les fenêtres identiques pendant cette extraction ; ce coût partagé ne peut pas être attribué séparément à chaque variante.", "", "![Comparaison des architectures, du rappel, des faux positifs et des latences](comparison.png)", ""]
    (result / "report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-id", default="comparison-v1-20260922")
    args = parser.parse_args()
    config = load_study_config(args.config)
    torch.set_num_threads(config["cpu_threads"])
    torch.set_num_interop_threads(1)
    result, _ = initialize(config, args.run_id)
    cache = ROOT / ".cache-comparison/inputs"
    inputs = prepare_inputs(config, cache)
    save_json(result / "input_audit.json", inputs)
    selection = freeze_selection(config, result)
    # No training/extraction workers should run concurrently with this phase.
    for candidate in selection["candidates"]:
        for index, relative in enumerate(candidate["runs"]):
            evaluate_run(result / relative, config, cache, inputs, do_latency=index == 0)
    rows, comparisons = aggregate(config, result, selection, cache)
    from .plot_comparison import plot_comparison
    plot_comparison(rows, result)
    write_report(config, result, selection, rows, comparisons, inputs)
    feature_root = ROOT / ".cache-comparison/codet5"
    save_json(result / "pretrained_manifest.json", json.loads((feature_root / "manifest.json").read_text()))
    save_json(result / "pretrained_extraction.json", {split: json.loads((feature_root / (split + "_extraction.json")).read_text()) for split in ("train", "validation", "calibration", "test", "source_holdout")})
    protocol = json.loads((result / "protocol.json").read_text())
    if file_digest(ROOT / "models/active_model.json") != protocol["production_manifest_sha256"]:
        raise ValueError("Production manifest changed during comparison")
    save_json(result / "completed.json", {"completed_utc": datetime.now(timezone.utc).isoformat(), "selected_variant": selection["selected_variant"], "production_manifest_unchanged": True, "evaluator_sha256": file_digest(Path(__file__)), "summary_sha256": file_digest(result / "summary.csv")})
    print(f"COMPLETED: {result / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
