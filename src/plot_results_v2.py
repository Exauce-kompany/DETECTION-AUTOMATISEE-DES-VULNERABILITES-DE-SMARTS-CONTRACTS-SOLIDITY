from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_V1 = PROJECT_ROOT / "results"
RESULTS_V2 = PROJECT_ROOT / "results" / "v2"

PLOTS_DIR = RESULTS_V2 / "plots"

HISTORY_V2_FILE = RESULTS_V2 / "training_history_v2.json"
METRICS_V2_FILE = RESULTS_V2 / "evaluation_metrics_v2.json"
REPORT_V2_FILE = RESULTS_V2 / "classification_report_v2.json"
CONFUSION_V2_FILE = RESULTS_V2 / "confusion_matrix_v2.json"

METRICS_V1_FILE = RESULTS_V1 / "evaluation_metrics.json"
CONFUSION_V1_FILE = RESULTS_V1 / "confusion_matrix.json"


# ============================================================
# UTILITAIRES
# ============================================================

def load_json(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def prepare_output_directory():

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Dossier des graphiques : "
        f"{PLOTS_DIR}"
    )


# ============================================================
# CHARGEMENT
# ============================================================

def load_results():

    print()
    print("Chargement des résultats V2...")

    history_v2 = load_json(
        HISTORY_V2_FILE
    )

    metrics_v2 = load_json(
        METRICS_V2_FILE
    )

    report_v2 = load_json(
        REPORT_V2_FILE
    )

    confusion_v2 = load_json(
        CONFUSION_V2_FILE
    )

    print("Résultats V2 chargés.")

    print()
    print("Chargement des résultats V1...")

    metrics_v1 = load_json(
        METRICS_V1_FILE
    )

    confusion_v1 = load_json(
        CONFUSION_V1_FILE
    )

    print("Résultats V1 chargés.")

    return (
        history_v2,
        metrics_v2,
        report_v2,
        confusion_v2,
        metrics_v1,
        confusion_v1,
    )


# ============================================================
# ACCURACY V2
# ============================================================

def plot_accuracy_v2(history):

    train_accuracy = history["accuracy"]
    val_accuracy = history["val_accuracy"]

    epochs = range(
        1,
        len(train_accuracy) + 1
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        epochs,
        train_accuracy,
        marker="o",
        label="Entraînement"
    )

    plt.plot(
        epochs,
        val_accuracy,
        marker="o",
        label="Validation"
    )

    plt.title(
        "Évolution de l'accuracy - Modèle V2"
    )

    plt.xlabel(
        "Époque"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.xticks(
        list(epochs)
    )

    plt.ylim(
        0,
        1
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        PLOTS_DIR
        / "accuracy_v2.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Graphique sauvegardé : "
        f"{output_path}"
    )


# ============================================================
# LOSS V2
# ============================================================

def plot_loss_v2(history):

    train_loss = history["loss"]
    val_loss = history["val_loss"]

    epochs = range(
        1,
        len(train_loss) + 1
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        epochs,
        train_loss,
        marker="o",
        label="Entraînement"
    )

    plt.plot(
        epochs,
        val_loss,
        marker="o",
        label="Validation"
    )

    plt.title(
        "Évolution de la loss - Modèle V2"
    )

    plt.xlabel(
        "Époque"
    )

    plt.ylabel(
        "Loss"
    )

    plt.xticks(
        list(epochs)
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        PLOTS_DIR
        / "loss_v2.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Graphique sauvegardé : "
        f"{output_path}"
    )


# ============================================================
# METRIQUES GLOBALES V2
# ============================================================

def plot_global_metrics_v2(metrics):

    names = [
        "Accuracy",
        "Precision\nmacro",
        "Recall\nmacro",
        "F1 macro",
        "Precision\nweighted",
        "Recall\nweighted",
        "F1 weighted",
    ]

    values = [
        metrics["accuracy"],
        metrics["precision_macro"],
        metrics["recall_macro"],
        metrics["f1_macro"],
        metrics["precision_weighted"],
        metrics["recall_weighted"],
        metrics["f1_weighted"],
    ]

    x = np.arange(
        len(names)
    )

    plt.figure(
        figsize=(12, 7)
    )

    bars = plt.bar(
        x,
        values
    )

    plt.title(
        "Métriques globales - Modèle V2"
    )

    plt.ylabel(
        "Score"
    )

    plt.xticks(
        x,
        names
    )

    plt.ylim(
        0,
        1
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    for bar, value in zip(
        bars,
        values
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    output_path = (
        PLOTS_DIR
        / "global_metrics_v2.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Graphique sauvegardé : "
        f"{output_path}"
    )


# ============================================================
# METRIQUES PAR CLASSE V2
# ============================================================

def plot_class_metrics_v2(report):

    class_names = [
        "non_vulnerable",
        "vulnerable",
    ]

    precision = [
        report[name]["precision"]
        for name in class_names
    ]

    recall = [
        report[name]["recall"]
        for name in class_names
    ]

    f1_score = [
        report[name]["f1-score"]
        for name in class_names
    ]

    x = np.arange(
        len(class_names)
    )

    width = 0.25

    plt.figure(
        figsize=(10, 6)
    )

    bars_precision = plt.bar(
        x - width,
        precision,
        width,
        label="Precision"
    )

    bars_recall = plt.bar(
        x,
        recall,
        width,
        label="Recall"
    )

    bars_f1 = plt.bar(
        x + width,
        f1_score,
        width,
        label="F1-score"
    )

    plt.title(
        "Performance par classe - Modèle V2"
    )

    plt.ylabel(
        "Score"
    )

    plt.xticks(
        x,
        class_names
    )

    plt.ylim(
        0,
        1
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.legend()

    for bars in [
        bars_precision,
        bars_recall,
        bars_f1,
    ]:

        for bar in bars:

            value = bar.get_height()

            plt.text(
                bar.get_x()
                + bar.get_width() / 2,
                value + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    plt.tight_layout()

    output_path = (
        PLOTS_DIR
        / "class_metrics_v2.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Graphique sauvegardé : "
        f"{output_path}"
    )


# ============================================================
# MATRICE DE CONFUSION V2
# ============================================================

def plot_confusion_matrix_v2(confusion_data):

    labels = confusion_data[
        "labels"
    ]

    matrix = np.array(
        confusion_data[
            "matrix"
        ]
    )

    if matrix.shape != (
        2,
        2
    ):

        raise ValueError(
            f"Matrice attendue 2x2, "
            f"trouvée {matrix.shape}"
        )

    print()
    print(
        "Matrice de confusion V2 :"
    )

    print(
        matrix
    )

    plt.figure(
        figsize=(8, 7)
    )

    image = plt.imshow(
        matrix,
        interpolation="nearest"
    )

    plt.title(
        "Matrice de confusion - Modèle V2"
    )

    plt.xlabel(
        "Classe prédite"
    )

    plt.ylabel(
        "Classe réelle"
    )

    plt.xticks(
        range(2),
        labels
    )

    plt.yticks(
        range(2),
        labels
    )

    plt.colorbar(
        image
    )

    for i in range(2):

        for j in range(2):

            plt.text(
                j,
                i,
                str(
                    int(
                        matrix[i][j]
                    )
                ),
                ha="center",
                va="center",
                fontsize=14
            )

    plt.tight_layout()

    output_path = (
        PLOTS_DIR
        / "confusion_matrix_v2.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Graphique sauvegardé : "
        f"{output_path}"
    )


# ============================================================
# COMPARAISON V1 / V2 DES METRIQUES
# ============================================================

def plot_v1_vs_v2_metrics(
    metrics_v1,
    metrics_v2,
):

    metric_names = [
        "Accuracy",
        "Precision\nmacro",
        "Recall\nmacro",
        "F1 macro",
    ]

    v1_values = [
        metrics_v1["accuracy"],
        metrics_v1["precision_macro"],
        metrics_v1["recall_macro"],
        metrics_v1["f1_macro"],
    ]

    v2_values = [
        metrics_v2["accuracy"],
        metrics_v2["precision_macro"],
        metrics_v2["recall_macro"],
        metrics_v2["f1_macro"],
    ]

    x = np.arange(
        len(metric_names)
    )

    width = 0.35

    plt.figure(
        figsize=(11, 7)
    )

    bars_v1 = plt.bar(
        x - width / 2,
        v1_values,
        width,
        label="V1"
    )

    bars_v2 = plt.bar(
        x + width / 2,
        v2_values,
        width,
        label="V2"
    )

    plt.title(
        "Comparaison des performances V1 vs V2"
    )

    plt.ylabel(
        "Score"
    )

    plt.xticks(
        x,
        metric_names
    )

    plt.ylim(
        0,
        1
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.legend()

    for bars in [
        bars_v1,
        bars_v2,
    ]:

        for bar in bars:

            value = bar.get_height()

            plt.text(
                bar.get_x()
                + bar.get_width() / 2,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    plt.tight_layout()

    output_path = (
        PLOTS_DIR
        / "v1_vs_v2_metrics.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Graphique sauvegardé : "
        f"{output_path}"
    )


# ============================================================
# COMPARAISON DES ERREURS V1 / V2
# ============================================================

def plot_v1_vs_v2_errors(
    confusion_v1,
    confusion_v2,
):

    matrix_v1 = np.array(
        confusion_v1["matrix"]
    )

    matrix_v2 = np.array(
        confusion_v2["matrix"]
    )

    fp_v1 = int(
        matrix_v1[0][1]
    )

    fn_v1 = int(
        matrix_v1[1][0]
    )

    fp_v2 = int(
        matrix_v2[0][1]
    )

    fn_v2 = int(
        matrix_v2[1][0]
    )

    labels = [
        "Faux positifs",
        "Faux négatifs",
    ]

    v1_values = [
        fp_v1,
        fn_v1,
    ]

    v2_values = [
        fp_v2,
        fn_v2,
    ]

    x = np.arange(
        len(labels)
    )

    width = 0.35

    plt.figure(
        figsize=(9, 6)
    )

    bars_v1 = plt.bar(
        x - width / 2,
        v1_values,
        width,
        label="V1"
    )

    bars_v2 = plt.bar(
        x + width / 2,
        v2_values,
        width,
        label="V2"
    )

    plt.title(
        "Réduction des erreurs : V1 vs V2"
    )

    plt.ylabel(
        "Nombre d'erreurs"
    )

    plt.xticks(
        x,
        labels
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.legend()

    for bars in [
        bars_v1,
        bars_v2,
    ]:

        for bar in bars:

            value = int(
                bar.get_height()
            )

            plt.text(
                bar.get_x()
                + bar.get_width() / 2,
                value + 3,
                str(value),
                ha="center",
                va="bottom"
            )

    plt.tight_layout()

    output_path = (
        PLOTS_DIR
        / "v1_vs_v2_errors.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Graphique sauvegardé : "
        f"{output_path}"
    )


# ============================================================
# RESUME COMPARATIF
# ============================================================

def save_comparison_summary(
    metrics_v1,
    metrics_v2,
    confusion_v1,
    confusion_v2,
):

    matrix_v1 = np.array(
        confusion_v1["matrix"]
    )

    matrix_v2 = np.array(
        confusion_v2["matrix"]
    )

    summary = {
        "v1": {
            "accuracy":
                metrics_v1["accuracy"],

            "precision_macro":
                metrics_v1["precision_macro"],

            "recall_macro":
                metrics_v1["recall_macro"],

            "f1_macro":
                metrics_v1["f1_macro"],

            "false_positive":
                int(
                    matrix_v1[0][1]
                ),

            "false_negative":
                int(
                    matrix_v1[1][0]
                ),
        },

        "v2": {
            "accuracy":
                metrics_v2["accuracy"],

            "precision_macro":
                metrics_v2["precision_macro"],

            "recall_macro":
                metrics_v2["recall_macro"],

            "f1_macro":
                metrics_v2["f1_macro"],

            "false_positive":
                int(
                    matrix_v2[0][1]
                ),

            "false_negative":
                int(
                    matrix_v2[1][0]
                ),
        },

        "improvements": {
            "accuracy":
                metrics_v2["accuracy"]
                - metrics_v1["accuracy"],

            "precision_macro":
                metrics_v2["precision_macro"]
                - metrics_v1["precision_macro"],

            "recall_macro":
                metrics_v2["recall_macro"]
                - metrics_v1["recall_macro"],

            "f1_macro":
                metrics_v2["f1_macro"]
                - metrics_v1["f1_macro"],

            "false_positive_reduction":
                int(
                    matrix_v1[0][1]
                    - matrix_v2[0][1]
                ),

            "false_negative_reduction":
                int(
                    matrix_v1[1][0]
                    - matrix_v2[1][0]
                ),
        }
    }

    output_path = (
        RESULTS_V2
        / "v1_vs_v2_summary.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print(
        f"Résumé comparatif sauvegardé : "
        f"{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GENERATION DES GRAPHIQUES V2")
    print("ET COMPARAISON V1 / V2")
    print("=" * 70)

    prepare_output_directory()

    (
        history_v2,
        metrics_v2,
        report_v2,
        confusion_v2,
        metrics_v1,
        confusion_v1,
    ) = load_results()

    print()
    print("=" * 70)
    print("GENERATION")
    print("=" * 70)

    plot_accuracy_v2(
        history_v2
    )

    plot_loss_v2(
        history_v2
    )

    plot_global_metrics_v2(
        metrics_v2
    )

    plot_class_metrics_v2(
        report_v2
    )

    plot_confusion_matrix_v2(
        confusion_v2
    )

    plot_v1_vs_v2_metrics(
        metrics_v1,
        metrics_v2,
    )

    plot_v1_vs_v2_errors(
        confusion_v1,
        confusion_v2,
    )

    save_comparison_summary(
        metrics_v1,
        metrics_v2,
        confusion_v1,
        confusion_v2,
    )

    print()
    print("=" * 70)
    print("RESULTATS COMPARATIFS")
    print("=" * 70)

    accuracy_gain = (
        metrics_v2["accuracy"]
        - metrics_v1["accuracy"]
    )

    f1_gain = (
        metrics_v2["f1_macro"]
        - metrics_v1["f1_macro"]
    )

    matrix_v1 = np.array(
        confusion_v1["matrix"]
    )

    matrix_v2 = np.array(
        confusion_v2["matrix"]
    )

    print(
        f"Accuracy V1 : "
        f"{metrics_v1['accuracy']:.4%}"
    )

    print(
        f"Accuracy V2 : "
        f"{metrics_v2['accuracy']:.4%}"
    )

    print(
        f"Gain accuracy : "
        f"{accuracy_gain:+.4%}"
    )

    print()

    print(
        f"F1 macro V1 : "
        f"{metrics_v1['f1_macro']:.4%}"
    )

    print(
        f"F1 macro V2 : "
        f"{metrics_v2['f1_macro']:.4%}"
    )

    print(
        f"Gain F1 macro : "
        f"{f1_gain:+.4%}"
    )

    print()

    print(
        "Faux positifs : "
        f"{matrix_v1[0][1]} "
        f"-> {matrix_v2[0][1]}"
    )

    print(
        "Faux négatifs : "
        f"{matrix_v1[1][0]} "
        f"-> {matrix_v2[1][0]}"
    )

    print()
    print("=" * 70)
    print("GRAPHIQUES V2 GENERES AVEC SUCCES")
    print("=" * 70)

    print(
        PLOTS_DIR
    )


if __name__ == "__main__":
    main()