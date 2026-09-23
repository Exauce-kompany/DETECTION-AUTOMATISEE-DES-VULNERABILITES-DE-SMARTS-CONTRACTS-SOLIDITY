"""Standalone figures for the completed exploratory study."""
from pathlib import Path
import numpy as np


def plot_comparison(rows, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination = Path(destination)
    indexed = {row["variant"]: row for row in rows}
    architectures = ["cnn", "bilstm", "cnn_bilstm", "codet5_frozen", "tfidf"]
    labels = ["CNN", "BiLSTM", "CNN–BiLSTM", "CodeT5 figé", "TF-IDF"]
    positions = np.arange(len(architectures))
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    metrics = [("test_all_calibrated_f1_macro", "F1 macro · seuil calibré", False), ("test_all_fpr_operating_point_recall_vulnerable", "Rappel · seuil FPR cible 10 %", False), ("test_all_fpr_operating_point_false_positive_rate", "FPR réellement obtenu sur le test", False), ("latency_median_ms", "Latence médiane par contrat (ms, échelle logarithmique)", True)]
    for axis, (metric, title, logarithmic) in zip(axes.flat, metrics):
        for mode, offset, color, label in [("full", -.19, "#2471A3", "Code complet"), ("first512", .19, "#D68910", "512 premiers tokens")]:
            entries = [indexed[architecture + "_" + mode] for architecture in architectures]
            values = [entry[metric if logarithmic else metric + "_mean"] for entry in entries]
            deviations = None if logarithmic else [entry[metric + "_std"] for entry in entries]
            axis.bar(positions + offset, values, .36, yerr=deviations, capsize=3, color=color, label=label)
        axis.set_title(title, fontsize=11, loc="left")
        axis.set_xticks(positions, labels, fontsize=9)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
        if logarithmic:
            axis.set_yscale("log")
        else:
            axis.set_ylim(0, 1)
        if "false_positive" in metric:
            axis.axhline(.1, color="#444444", linewidth=1, linestyle="--")
    axes[0, 0].legend(frameon=False, fontsize=9)
    figure.suptitle("Détection de vulnérabilités Solidity · comparaison exploratoire", fontsize=16)
    figure.supxlabel("Test V3 déjà consulté · barres d'erreur : écart-type entre graines · CodeT5 : encodeur figé + tête entraînée", fontsize=9)
    for extension in ("png", "svg"):
        figure.savefig(destination / ("comparison." + extension), dpi=160, facecolor="white")
    plt.close(figure)
