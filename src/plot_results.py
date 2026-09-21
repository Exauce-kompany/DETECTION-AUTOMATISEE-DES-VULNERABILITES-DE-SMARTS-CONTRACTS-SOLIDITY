from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

HISTORY_FILE = RESULTS_DIR / "training_history.json"
METRICS_FILE = RESULTS_DIR / "evaluation_metrics.json"
REPORT_FILE = RESULTS_DIR / "classification_report.json"
CONFUSION_FILE = RESULTS_DIR / "confusion_matrix.json"


# ============================================================
# CHARGEMENT DES RESULTATS
# ============================================================

def load_results():

    print("Chargement de l'historique...")

    if not HISTORY_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {HISTORY_FILE}"
        )

    with open(
        HISTORY_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        history = json.load(file)

    print("Historique chargé.")

    print("Chargement des métriques...")

    if not METRICS_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {METRICS_FILE}"
        )

    with open(
        METRICS_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        metrics = json.load(file)

    print("Métriques chargées.")

    print("Chargement du rapport de classification...")

    if not REPORT_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {REPORT_FILE}"
        )

    with open(
        REPORT_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        report = json.load(file)

    print("Rapport chargé.")

    print("Chargement de la matrice de confusion...")

    if not CONFUSION_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {CONFUSION_FILE}"
        )

    with open(
        CONFUSION_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        confusion_data = json.load(file)

    print("Matrice de confusion chargée.")

    labels = confusion_data["labels"]

    confusion_matrix = np.array(
        confusion_data["matrix"]
    )

    return (
        history,
        metrics,
        report,
        labels,
        confusion_matrix,
    )


# ============================================================
# PREPARATION DU DOSSIER
# ============================================================

def prepare_output_directory():

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print(
        f"Dossier des graphiques : {PLOTS_DIR}"
    )


# ============================================================
# 1. GRAPHIQUE ACCURACY
# ============================================================

def plot_accuracy(history):

    print()
    print("Génération du graphique accuracy...")

    if "accuracy" not in history:
        raise ValueError(
            "La clé 'accuracy' est absente de l'historique."
        )

    if "val_accuracy" not in history:
        raise ValueError(
            "La clé 'val_accuracy' est absente de l'historique."
        )

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
        "Évolution de l'accuracy"
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
        / "accuracy.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Graphique sauvegardé : {output_path}"
    )


# ============================================================
# 2. GRAPHIQUE LOSS
# ============================================================

def plot_loss(history):

    print()
    print("Génération du graphique loss...")

    if "loss" not in history:
        raise ValueError(
            "La clé 'loss' est absente de l'historique."
        )

    if "val_loss" not in history:
        raise ValueError(
            "La clé 'val_loss' est absente de l'historique."
        )

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
        "Évolution de la fonction de perte"
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
        / "loss.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Graphique sauvegardé : {output_path}"
    )


# ============================================================
# 3. METRIQUES GLOBALES
# ============================================================

def plot_global_metrics(metrics):

    print()
    print("Génération du graphique des métriques globales...")

    metric_names = [
        "Accuracy",
        "Precision\nmacro",
        "Recall\nmacro",
        "F1-score\nmacro",
        "Precision\nweighted",
        "Recall\nweighted",
        "F1-score\nweighted",
    ]

    metric_values = [
        metrics["accuracy"],
        metrics["precision_macro"],
        metrics["recall_macro"],
        metrics["f1_macro"],
        metrics["precision_weighted"],
        metrics["recall_weighted"],
        metrics["f1_weighted"],
    ]

    x = np.arange(
        len(metric_names)
    )

    plt.figure(
        figsize=(12, 7)
    )

    bars = plt.bar(
        x,
        metric_values
    )

    plt.title(
        "Métriques globales du modèle"
    )

    plt.xlabel(
        "Métriques"
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

    for bar, value in zip(
        bars,
        metric_values
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    output_path = (
        PLOTS_DIR
        / "global_metrics.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Graphique sauvegardé : {output_path}"
    )


# ============================================================
# 4. METRIQUES PAR CLASSE
# ============================================================

def plot_class_metrics(report):

    print()
    print("Génération du graphique des métriques par classe...")

    class_names = [
        "non_vulnerable",
        "vulnerable"
    ]

    precision = []
    recall = []
    f1_scores = []

    for class_name in class_names:

        if class_name not in report:
            raise ValueError(
                f"Classe absente du rapport : {class_name}"
            )

        precision.append(
            report[class_name]["precision"]
        )

        recall.append(
            report[class_name]["recall"]
        )

        f1_scores.append(
            report[class_name]["f1-score"]
        )

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
        f1_scores,
        width,
        label="F1-score"
    )

    plt.title(
        "Performance du modèle par classe"
    )

    plt.xlabel(
        "Classe"
    )

    plt.ylabel(
        "Score"
    )

    plt.xticks(
        x,
        class_names,
        rotation=15
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

    # --------------------------------------------------------
    # Valeurs sur les barres
    # --------------------------------------------------------

    for bars in [
        bars_precision,
        bars_recall,
        bars_f1
    ]:

        for bar in bars:

            height = bar.get_height()

            plt.text(
                bar.get_x()
                + bar.get_width() / 2,
                height + 0.02,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    plt.tight_layout()

    output_path = (
        PLOTS_DIR
        / "class_metrics.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Graphique sauvegardé : {output_path}"
    )


# ============================================================
# 5. MATRICE DE CONFUSION
# ============================================================

def plot_confusion_matrix(
    confusion_matrix,
    labels
):

    print()
    print("Génération de la matrice de confusion...")

    # --------------------------------------------------------
    # Vérification de la matrice
    # --------------------------------------------------------

    if confusion_matrix.ndim != 2:
        raise ValueError(
            "La matrice de confusion doit être 2D."
        )

    rows, columns = (
        confusion_matrix.shape
    )

    if rows != columns:
        raise ValueError(
            "La matrice de confusion doit être carrée."
        )

    if rows != len(labels):
        raise ValueError(
            "Le nombre de labels ne correspond pas "
            "à la taille de la matrice."
        )

    # --------------------------------------------------------
    # Notre problème est binaire
    # --------------------------------------------------------

    if rows != 2:

        raise ValueError(
            f"Le projet actuel attend 2 classes, "
            f"mais la matrice contient {rows} classes."
        )

    print(
        f"Matrice de confusion vérifiée : "
        f"{rows} x {columns}"
    )

    print()

    print(
        "Classes utilisées :"
    )

    for index, label in enumerate(labels):

        print(
            f"{index} -> {label}"
        )

    print()

    print(
        "Matrice :"
    )

    print(
        confusion_matrix
    )

    # --------------------------------------------------------
    # Création du graphique
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 7)
    )

    image = plt.imshow(
        confusion_matrix,
        interpolation="nearest"
    )

    plt.title(
        "Matrice de confusion"
    )

    plt.xlabel(
        "Classe prédite"
    )

    plt.ylabel(
        "Classe réelle"
    )

    plt.xticks(
        range(len(labels)),
        labels,
        rotation=15
    )

    plt.yticks(
        range(len(labels)),
        labels
    )

    plt.colorbar(
        image
    )

    # --------------------------------------------------------
    # Valeurs dans les cellules
    # --------------------------------------------------------

    threshold = (
        confusion_matrix.max()
        / 2.0
    )

    for i in range(rows):

        for j in range(columns):

            value = int(
                confusion_matrix[i, j]
            )

            plt.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                fontsize=14
            )

    plt.tight_layout()

    output_path = (
        PLOTS_DIR
        / "confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Graphique sauvegardé : {output_path}"
    )


# ============================================================
# RESUME DES RESULTATS
# ============================================================

def create_summary(
    metrics,
    confusion_matrix
):

    print()
    print("=" * 60)
    print("RESUME DES RESULTATS")
    print("=" * 60)

    tn = int(
        confusion_matrix[0][0]
    )

    fp = int(
        confusion_matrix[0][1]
    )

    fn = int(
        confusion_matrix[1][0]
    )

    tp = int(
        confusion_matrix[1][1]
    )

    total = (
        tn
        + fp
        + fn
        + tp
    )

    summary = {

        "total_test_samples":
            total,

        "true_negative":
            tn,

        "false_positive":
            fp,

        "false_negative":
            fn,

        "true_positive":
            tp,

        "accuracy":
            metrics["accuracy"],

        "precision_macro":
            metrics["precision_macro"],

        "recall_macro":
            metrics["recall_macro"],

        "f1_macro":
            metrics["f1_macro"],

        "precision_weighted":
            metrics["precision_weighted"],

        "recall_weighted":
            metrics["recall_weighted"],

        "f1_weighted":
            metrics["f1_weighted"],
    }

    summary_path = (
        RESULTS_DIR
        / "evaluation_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print(
        f"Résumé sauvegardé : {summary_path}"
    )


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("=" * 60)
    print("GÉNÉRATION DES GRAPHIQUES")
    print("DÉTECTION DES VULNÉRABILITÉS")
    print("DES SMART CONTRACTS")
    print("=" * 60)

    print()

    prepare_output_directory()

    (
        history,
        metrics,
        report,
        labels,
        confusion_matrix,
    ) = load_results()

    print()
    print("=" * 60)
    print("GENERATION DES GRAPHIQUES")
    print("=" * 60)

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    plot_accuracy(
        history
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    plot_loss(
        history
    )

    # --------------------------------------------------------
    # Métriques globales
    # --------------------------------------------------------

    plot_global_metrics(
        metrics
    )

    # --------------------------------------------------------
    # Métriques par classe
    # --------------------------------------------------------

    plot_class_metrics(
        report
    )

    # --------------------------------------------------------
    # Matrice de confusion
    # --------------------------------------------------------

    plot_confusion_matrix(
        confusion_matrix,
        labels
    )

    # --------------------------------------------------------
    # Résumé
    # --------------------------------------------------------

    create_summary(
        metrics,
        confusion_matrix
    )

    print()
    print("=" * 60)
    print("GÉNÉRATION TERMINÉE")
    print("=" * 60)

    print()

    print(
        "Les graphiques sont disponibles dans :"
    )

    print(
        PLOTS_DIR
    )

    print()


if __name__ == "__main__":
    main()

