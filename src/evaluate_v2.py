from pathlib import Path
import json
import pickle

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "v2"
    / "prepared"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "smart_contract_vulnerability_model_v2.keras"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "v2"
)

BATCH_SIZE = 32

EXPECTED_CLASSES = 2
EXPECTED_SEQUENCE_LENGTH = 512


# ============================================================
# CHARGEMENT
# ============================================================

def load_resources():

    print("=" * 70)
    print("EVALUATION COMPLETE DU MODELE V2")
    print("DETECTION DES VULNERABILITES")
    print("DES SMART CONTRACTS")
    print("=" * 70)

    required_files = [
        DATA_DIR / "X_test.npy",
        DATA_DIR / "y_test.npy",
        DATA_DIR / "labels.json",
        DATA_DIR / "vocabulary.pkl",
        MODEL_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Fichier introuvable : {path}"
            )

    X_test = np.load(
        DATA_DIR / "X_test.npy"
    )

    y_test = np.load(
        DATA_DIR / "y_test.npy"
    )

    with open(
        DATA_DIR / "labels.json",
        "r",
        encoding="utf-8"
    ) as file:
        labels_dict = json.load(file)

    with open(
        DATA_DIR / "vocabulary.pkl",
        "rb"
    ) as file:
        vocabulary = pickle.load(file)

    print()
    print("Chargement du modèle...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Modèle chargé.")

    print()
    print(
        f"X_test : {X_test.shape}"
    )

    print(
        f"y_test : {y_test.shape}"
    )

    print(
        f"Vocabulaire : {len(vocabulary):,}"
    )

    print(
        f"Labels : {labels_dict}"
    )

    return (
        X_test,
        y_test,
        labels_dict,
        vocabulary,
        model,
    )


# ============================================================
# VERIFICATIONS
# ============================================================

def validate_resources(
    X_test,
    y_test,
    labels_dict,
    vocabulary,
    model,
):

    print()
    print("=" * 70)
    print("VERIFICATIONS")
    print("=" * 70)

    if X_test.ndim != 2:
        raise ValueError(
            "X_test doit être une matrice 2D."
        )

    if X_test.shape[1] != EXPECTED_SEQUENCE_LENGTH:
        raise ValueError(
            f"Longueur attendue : "
            f"{EXPECTED_SEQUENCE_LENGTH}, "
            f"trouvée : {X_test.shape[1]}"
        )

    if len(labels_dict) != EXPECTED_CLASSES:
        raise ValueError(
            f"{EXPECTED_CLASSES} classes attendues, "
            f"{len(labels_dict)} trouvées."
        )

    unique_labels = set(
        int(x)
        for x in np.unique(y_test)
    )

    if unique_labels != {0, 1}:
        raise ValueError(
            f"Labels test invalides : "
            f"{unique_labels}"
        )

    max_token_id = int(
        X_test.max()
    )

    if max_token_id >= len(vocabulary):
        raise ValueError(
            "Un token ID dépasse la taille du vocabulaire."
        )

    output_shape = model.output_shape

    if output_shape[-1] != EXPECTED_CLASSES:
        raise ValueError(
            f"Sortie modèle incorrecte : "
            f"{output_shape}"
        )

    print(
        "Longueur des séquences : OK"
    )

    print(
        "Labels binaires : OK"
    )

    print(
        "Token IDs : OK"
    )

    print(
        "Nombre de classes du modèle : OK"
    )


# ============================================================
# EVALUATION KERAS
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
):

    print()
    print("=" * 70)
    print("EVALUATION KERAS")
    print("=" * 70)

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    print()
    print(
        f"Test Loss     : "
        f"{test_loss:.6f}"
    )

    print(
        f"Test Accuracy : "
        f"{test_accuracy:.6f}"
    )

    return (
        float(test_loss),
        float(test_accuracy),
    )


# ============================================================
# PREDICTIONS
# ============================================================

def generate_predictions(
    model,
    X_test,
):

    print()
    print("=" * 70)
    print("GENERATION DES PREDICTIONS")
    print("=" * 70)

    probabilities = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    confidences = np.max(
        probabilities,
        axis=1
    )

    print()
    print(
        f"Prédictions générées : "
        f"{len(predictions):,}"
    )

    return (
        probabilities,
        predictions,
        confidences,
    )


# ============================================================
# METRIQUES
# ============================================================

def calculate_metrics(
    y_test,
    predictions,
    test_loss,
    keras_accuracy,
):

    print()
    print("=" * 70)
    print("METRIQUES GLOBALES")
    print("=" * 70)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision_macro = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    recall_macro = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    f1_macro = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    precision_weighted = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall_weighted = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1_weighted = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    metrics = {
        "test_loss":
            float(test_loss),

        "keras_accuracy":
            float(keras_accuracy),

        "accuracy":
            float(accuracy),

        "precision_macro":
            float(precision_macro),

        "recall_macro":
            float(recall_macro),

        "f1_macro":
            float(f1_macro),

        "precision_weighted":
            float(precision_weighted),

        "recall_weighted":
            float(recall_weighted),

        "f1_weighted":
            float(f1_weighted),
    }

    print(
        f"Accuracy             : "
        f"{accuracy:.6f}"
    )

    print(
        f"Precision macro      : "
        f"{precision_macro:.6f}"
    )

    print(
        f"Recall macro         : "
        f"{recall_macro:.6f}"
    )

    print(
        f"F1 macro             : "
        f"{f1_macro:.6f}"
    )

    print(
        f"Precision weighted   : "
        f"{precision_weighted:.6f}"
    )

    print(
        f"Recall weighted      : "
        f"{recall_weighted:.6f}"
    )

    print(
        f"F1 weighted          : "
        f"{f1_weighted:.6f}"
    )

    return metrics


# ============================================================
# RAPPORT DE CLASSIFICATION
# ============================================================

def generate_classification_report(
    y_test,
    predictions,
    labels_dict,
):

    print()
    print("=" * 70)
    print("RAPPORT DE CLASSIFICATION")
    print("=" * 70)

    class_names = [
        labels_dict[
            str(index)
        ]
        for index in range(
            EXPECTED_CLASSES
        )
    ]

    report_text = classification_report(
        y_test,
        predictions,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    report_dict = classification_report(
        y_test,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    print()
    print(
        report_text
    )

    return (
        report_text,
        report_dict,
        class_names,
    )


# ============================================================
# MATRICE DE CONFUSION
# ============================================================

def generate_confusion_matrix(
    y_test,
    predictions,
    class_names,
):

    print()
    print("=" * 70)
    print("MATRICE DE CONFUSION")
    print("=" * 70)

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    print()
    print(
        matrix
    )

    if matrix.shape != (
        EXPECTED_CLASSES,
        EXPECTED_CLASSES
    ):
        raise ValueError(
            f"Matrice attendue 2x2, "
            f"trouvée {matrix.shape}"
        )

    tn = int(
        matrix[0][0]
    )

    fp = int(
        matrix[0][1]
    )

    fn = int(
        matrix[1][0]
    )

    tp = int(
        matrix[1][1]
    )

    print()
    print(
        f"True Negative  : {tn:,}"
    )

    print(
        f"False Positive : {fp:,}"
    )

    print(
        f"False Negative : {fn:,}"
    )

    print(
        f"True Positive  : {tp:,}"
    )

    matrix_data = {
        "labels":
            class_names,

        "matrix":
            matrix.tolist(),

        "true_negative":
            tn,

        "false_positive":
            fp,

        "false_negative":
            fn,

        "true_positive":
            tp,
    }

    return matrix_data


# ============================================================
# PREDICTIONS DETAILLEES
# ============================================================

def build_prediction_records(
    y_test,
    predictions,
    probabilities,
    confidences,
    labels_dict,
):

    records = []

    for index in range(
        len(y_test)
    ):

        true_label = int(
            y_test[index]
        )

        predicted_label = int(
            predictions[index]
        )

        record = {
            "index":
                int(index),

            "true_label":
                true_label,

            "true_class":
                labels_dict[
                    str(true_label)
                ],

            "predicted_label":
                predicted_label,

            "predicted_class":
                labels_dict[
                    str(predicted_label)
                ],

            "confidence":
                float(
                    confidences[index]
                ),

            "probability_non_vulnerable":
                float(
                    probabilities[
                        index
                    ][0]
                ),

            "probability_vulnerable":
                float(
                    probabilities[
                        index
                    ][1]
                ),

            "correct":
                bool(
                    true_label
                    == predicted_label
                ),
        }

        records.append(
            record
        )

    return records


# ============================================================
# AFFICHAGE DE QUELQUES PREDICTIONS
# ============================================================

def show_predictions(
    records,
):

    print()
    print("=" * 70)
    print("EXEMPLES DE PREDICTIONS")
    print("=" * 70)

    for record in records[:10]:

        print(
            f"#{record['index']:<4} "
            f"Réel="
            f"{record['true_class']:<16} "
            f"Prédit="
            f"{record['predicted_class']:<16} "
            f"Confiance="
            f"{record['confidence']:.4f}"
        )


# ============================================================
# SAUVEGARDE
# ============================================================

def save_results(
    metrics,
    report_text,
    report_dict,
    matrix_data,
    prediction_records,
):

    print()
    print("=" * 70)
    print("SAUVEGARDE DES RESULTATS V2")
    print("=" * 70)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    metrics_path = (
        RESULTS_DIR
        / "evaluation_metrics_v2.json"
    )

    report_json_path = (
        RESULTS_DIR
        / "classification_report_v2.json"
    )

    report_txt_path = (
        RESULTS_DIR
        / "classification_report_v2.txt"
    )

    confusion_path = (
        RESULTS_DIR
        / "confusion_matrix_v2.json"
    )

    predictions_path = (
        RESULTS_DIR
        / "predictions_v2.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4
        )

    with open(
        report_json_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            report_dict,
            file,
            indent=4
        )

    with open(
        report_txt_path,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(
            report_text
        )

    with open(
        confusion_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            matrix_data,
            file,
            indent=4
        )

    with open(
        predictions_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            prediction_records,
            file,
            indent=2
        )

    print(
        f"Métriques : {metrics_path}"
    )

    print(
        f"Rapport JSON : {report_json_path}"
    )

    print(
        f"Rapport texte : {report_txt_path}"
    )

    print(
        f"Matrice : {confusion_path}"
    )

    print(
        f"Prédictions : {predictions_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        f"TensorFlow : "
        f"{tf.__version__}"
    )

    (
        X_test,
        y_test,
        labels_dict,
        vocabulary,
        model,
    ) = load_resources()

    validate_resources(
        X_test,
        y_test,
        labels_dict,
        vocabulary,
        model,
    )

    (
        test_loss,
        keras_accuracy,
    ) = evaluate_model(
        model,
        X_test,
        y_test,
    )

    (
        probabilities,
        predictions,
        confidences,
    ) = generate_predictions(
        model,
        X_test,
    )

    metrics = calculate_metrics(
        y_test,
        predictions,
        test_loss,
        keras_accuracy,
    )

    (
        report_text,
        report_dict,
        class_names,
    ) = generate_classification_report(
        y_test,
        predictions,
        labels_dict,
    )

    matrix_data = (
        generate_confusion_matrix(
            y_test,
            predictions,
            class_names,
        )
    )

    prediction_records = (
        build_prediction_records(
            y_test,
            predictions,
            probabilities,
            confidences,
            labels_dict,
        )
    )

    show_predictions(
        prediction_records
    )

    save_results(
        metrics,
        report_text,
        report_dict,
        matrix_data,
        prediction_records,
    )

    print()
    print("=" * 70)
    print("EVALUATION V2 TERMINEE AVEC SUCCES")
    print("=" * 70)


if __name__ == "__main__":
    main()