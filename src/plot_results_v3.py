"""Export figures from the active V3 experiment without training or changing it."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .preprocessing_v3 import ROOT, file_digest


def export_figures(manifest_path=ROOT / "models/active_model.json"):
    manifest_path = Path(manifest_path)
    bundle = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = ROOT / bundle["results_path"]
    evaluation_path = result / "evaluation.json"
    history_path = result / f"seed-{bundle['selected']['seed']}-history.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))["test"]
    history = json.loads(history_path.read_text(encoding="utf-8"))
    active = evaluation["neural_calibrated"]
    output = result / "plots"
    output.mkdir(parents=True, exist_ok=True)
    files = []
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    note = (f"V3 actif · graine {bundle['selected']['seed']} · "
            f"{bundle['run_id']}")

    def save(figure, name, caption):
        figure.text(.5, .015, caption, ha="center", fontsize=8)
        figure.tight_layout(rect=(0, .055, 1, 1))
        for extension in ("png", "svg"):
            filename = f"{name}.{extension}"
            figure.savefig(output / filename, dpi=150, facecolor="white")
            files.append(filename)
        plt.close(figure)

    for metric, title, ylabel in [
        ("accuracy", "Exactitude pendant l'apprentissage", "Exactitude"),
        ("loss", "Perte pendant l'apprentissage", "Perte binaire")]:
        figure, axis = plt.subplots(figsize=(9, 5))
        epochs = np.arange(1, len(history[metric]) + 1)
        for key, label, color in [(metric, "Entraînement", "#2471A3"),
                                  ("val_" + metric, "Validation", "#B9770E")]:
            axis.plot(epochs, history[key], marker="o", label=label, color=color)
        axis.axvline(bundle["selected"]["best_epoch"], color="#555555", linestyle="--",
                     label="Checkpoint retenu sur la perte de validation")
        axis.set(title=title + " — V3", xlabel="Époque", ylabel=ylabel, xticks=epochs)
        if metric == "accuracy":
            axis.set_ylim(0, 1)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
        save(figure, metric, note + ("\nPerte d'entraînement pondérée ; perte de validation non pondérée."
                                     if metric == "loss" else ""))

    figure, axis = plt.subplots(figsize=(8, 6))
    matrix = np.asarray(active["confusion_matrix"])
    axis.imshow(matrix, cmap="Blues", vmin=0)
    labels = ["0 : absence d'annotation", "1 : vulnérabilité annotée"]
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=labels, yticklabels=labels,
             xlabel="Prédiction", ylabel="Label réel", title="Matrice de confusion — test V3 actif")
    for (i, j), value in np.ndenumerate(matrix):
        axis.text(j, i, str(value), ha="center", va="center", fontsize=20,
                  color="white" if value > matrix.max() / 2 else "black")
    save(figure, "confusion_matrix", note + f"\n{active['samples']} contrats · seuil calibré {bundle['calibration']['threshold']:.4f}")

    figure, axis = plt.subplots(figsize=(10, 5.5))
    metrics = ["accuracy", "f1_macro", "recall_vulnerable"]
    positions = np.arange(len(metrics))
    for offset, key, label, color in [(-.25, "neural_raw", "CNN-BiLSTM brut (seuil 0,5)", "#7F8C8D"),
            (0, "neural_calibrated", "CNN-BiLSTM actif calibré", "#2471A3"),
            (.25, "tfidf_baseline", "TF-IDF calibré", "#B9770E")]:
        values = [evaluation[key][metric] for metric in metrics]
        bars = axis.bar(positions + offset, values, .24, label=label, color=color)
        axis.bar_label(bars, labels=[f"{100*v:.2f} %" for v in values], fontsize=8, padding=3)
    axis.set(xticks=positions, xticklabels=["Exactitude", "F1 macro", "Rappel positif"],
             ylim=(0, 1.08), title="Performances sur le test — expérience V3 active")
    axis.legend(loc="lower left", fontsize=8)
    axis.grid(axis="y", alpha=.2)
    axis.set_axisbelow(True)
    save(figure, "global_metrics", note + "\nLes résultats de la comparaison PyTorch sont présentés dans results/comparison.")

    figure, axis = plt.subplots(figsize=(9, 5.5))
    positions = np.arange(3)
    for offset, key, label, color in [(-.18, "no_annotated_vulnerability", "Absence de vulnérabilité annotée", "#2471A3"),
            (.18, "vulnerability_annotated", "Vulnérabilité annotée", "#B9770E")]:
        values = [active["classification_report"][key][metric] for metric in ["precision", "recall", "f1-score"]]
        bars = axis.bar(positions + offset, values, .35, label=label, color=color)
        axis.bar_label(bars, labels=[f"{100*v:.2f} %" for v in values], fontsize=9, padding=3)
    axis.set(xticks=positions, xticklabels=["Précision", "Rappel", "F1"], ylim=(0, 1.08),
             title="Métriques par classe — test V3 actif")
    axis.legend(loc="lower left", fontsize=8)
    axis.grid(axis="y", alpha=.2)
    axis.set_axisbelow(True)
    save(figure, "class_metrics", note + "\nSeuil actif choisi sur la calibration ; la classe 0 n'est pas une certification de sécurité.")

    sources = [manifest_path, evaluation_path, history_path]
    provenance = {"run_id": bundle["run_id"], "dataset_id": bundle["dataset_id"],
                  "selected_seed": bundle["selected"]["seed"],
                  "source_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): file_digest(p) for p in sources},
                  "generator_sha256": file_digest(__file__),
                  "files": {name: file_digest(output / name) for name in files}}
    (output / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "models/active_model.json")
    export_figures(parser.parse_args().manifest)
