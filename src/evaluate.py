from pathlib import Path
import json
import pickle

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "prepared"
)

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_PATH = (
    MODEL_DIR
    / "smart_contract_vulnerability_model.keras"
)

RANDOM_STATE = 42

MAX_SEQUENCE_LENGTH = 512
EXPECTED_NUM_CLASSES = 2
EXPECTED_VOCAB_SIZE = 5000


# ============================================================
# REPRODUCTIBILITE
# ============================================================

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# VERIFICATION DES FICHIERS
# ============================================================

def check_files():

    print("=" * 60)
    print("VERIFICATION DES FICHIERS")
    print("=" * 60)

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
                f"Fichier manquant : {path}"
            )

        print(f"[OK] {path}")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()


# ============================================================
# CHARGEMENT DU JEU DE TEST
# ============================================================

def load_test_data():

    print("=" * 60)
    print("CHARGEMENT DU JEU DE TEST")
    print("=" * 60)

    X_test = np.load(
        DATA_DIR / "X_test.npy"
    )

    y_test = np.load(
        DATA_DIR / "y_test.npy"
    )

    print(
        f"X_test : {X_test.shape}"
    )

    print(
        f"y_test : {y_test.shape}"
    )

    print(
        f"Nombre de contrats de test : "
        f"{len(y_test)}"
    )

    print()

    return X_test, y_test


# ============================================================
# CHARGEMENT DU VOCABULAIRE
# ============================================================

def load_vocabulary():

    print("=" * 60)
    print("CHARGEMENT DU VOCABULAIRE")
    print("=" * 60)

    path = DATA_DIR / "vocabulary.pkl"

    with open(
        path,
        "rb"
    ) as file:

        vocabulary = pickle.load(file)

    print(
        f"Taille du vocabulaire : "
        f"{len(vocabulary)}"
    )

    print()

    return vocabulary


# ============================================================
# CHARGEMENT DES CLASSES
# ============================================================

def load_labels():

    print("=" * 60)
    print("CHARGEMENT DES CLASSES")
    print("=" * 60)

    path = DATA_DIR / "labels.json"

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        labels = json.load(file)

    id_to_label = {
        int(class_id): label
        for class_id, label
        in labels["id_to_label"].items()
    }

    print()

    for class_id in sorted(id_to_label):

        print(
            f"{class_id} -> "
            f"{id_to_label[class_id]}"
        )

    print()

    return id_to_label


# ============================================================
# CHARGEMENT DU MODELE
# ============================================================

def load_model():

    print("=" * 60)
    print("CHARGEMENT DU MODELE")
    print("=" * 60)

    print(
        f"Modèle : {MODEL_PATH}"
    )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print()

    print(
        "Modèle chargé avec succès."
    )

    print()

    return model


# ============================================================
# VERIFICATION DES DONNEES
# ============================================================

def validate_data(
    X_test,
    y_test,
    vocabulary,
    id_to_label,
    model,
):

    print("=" * 60)
    print("VERIFICATION DES DONNEES")
    print("=" * 60)

    # --------------------------------------------------------
    # Dimensions de X_test
    # --------------------------------------------------------

    if X_test.ndim != 2:

        raise ValueError(
            "X_test doit être une matrice 2D."
        )

    if X_test.shape[1] != MAX_SEQUENCE_LENGTH:

        raise ValueError(
            f"X_test doit contenir "
            f"{MAX_SEQUENCE_LENGTH} tokens par contrat. "
            f"Dimension trouvée : {X_test.shape[1]}"
        )

    # --------------------------------------------------------
    # Correspondance X / y
    # --------------------------------------------------------

    if len(X_test) != len(y_test):

        raise ValueError(
            "X_test et y_test doivent "
            "avoir le même nombre d'éléments."
        )

    # --------------------------------------------------------
    # Nombre de classes
    # --------------------------------------------------------

    num_classes = len(id_to_label)

    if num_classes != EXPECTED_NUM_CLASSES:

        raise ValueError(
            f"{EXPECTED_NUM_CLASSES} classes attendues, "
            f"{num_classes} trouvées."
        )

    # --------------------------------------------------------
    # Identifiants des classes
    # --------------------------------------------------------

    expected_class_ids = {0, 1}

    actual_class_ids = set(
        id_to_label.keys()
    )

    if actual_class_ids != expected_class_ids:

        raise ValueError(
            "Les identifiants des classes doivent "
            "être exactement {0, 1}."
        )

    # --------------------------------------------------------
    # Valeurs des labels
    # --------------------------------------------------------

    unique_labels = np.unique(
        y_test
    )

    if not np.all(
        np.isin(
            unique_labels,
            [0, 1]
        )
    ):

        raise ValueError(
            "y_test contient des labels "
            "autres que 0 et 1."
        )

    if np.any(y_test < 0):

        raise ValueError(
            "y_test contient des labels négatifs."
        )

    if np.any(y_test >= num_classes):

        raise ValueError(
            "y_test contient un identifiant "
            "de classe invalide."
        )

    # --------------------------------------------------------
    # Vérification du vocabulaire
    # --------------------------------------------------------

    if len(vocabulary) != EXPECTED_VOCAB_SIZE:

        print(
            "AVERTISSEMENT : "
            f"vocabulaire attendu = "
            f"{EXPECTED_VOCAB_SIZE}, "
            f"vocabulaire trouvé = "
            f"{len(vocabulary)}"
        )

    max_token_id = int(
        np.max(X_test)
    )

    min_token_id = int(
        np.min(X_test)
    )

    if min_token_id < 0:

        raise ValueError(
            "X_test contient des identifiants "
            "de tokens négatifs."
        )

    if max_token_id >= len(vocabulary):

        raise ValueError(
            "Un identifiant de token "
            "dépasse la taille du vocabulaire."
        )

    # --------------------------------------------------------
    # Vérification du modèle
    # --------------------------------------------------------

    model_output_shape = model.output_shape

    model_num_classes = model_output_shape[-1]

    if model_num_classes != num_classes:

        raise ValueError(
            f"Incompatibilité entre le modèle "
            f"({model_num_classes} sorties) et les "
            f"classes ({num_classes})."
        )

    # --------------------------------------------------------
    # Résumé
    # --------------------------------------------------------

    print(
        "Vérification réussie."
    )

    print(
        f"Nombre de classes : "
        f"{num_classes}"
    )

    print(
        "0 = non_vulnerable"
    )

    print(
        "1 = vulnerable"
    )

    print(
        f"Longueur des séquences : "
        f"{X_test.shape[1]}"
    )

    print(
        f"Nombre d'échantillons : "
        f"{len(y_test)}"
    )

    print(
        f"Token ID minimum : "
        f"{min_token_id}"
    )

    print(
        f"Token ID maximum : "
        f"{max_token_id}"
    )

    print(
        f"Sorties du modèle : "
        f"{model_num_classes}"
    )

    print()


# ============================================================
# DISTRIBUTION DES CLASSES
# ============================================================

def show_class_distribution(
    y_test,
    id_to_label,
):

    print("=" * 60)
    print("DISTRIBUTION DES CLASSES - TEST")
    print("=" * 60)

    total = len(y_test)

    for class_id in sorted(id_to_label):

        count = int(
            np.sum(
                y_test == class_id
            )
        )

        percentage = (
            count / total * 100
        )

        print(
            f"Classe {class_id} "
            f"({id_to_label[class_id]}) : "
            f"{count} "
            f"({percentage:.2f}%)"
        )

    print()


# ============================================================
# EVALUATION GLOBALE
# ============================================================

def evaluate_basic_metrics(
    model,
    X_test,
    y_test,
):

    print("=" * 60)
    print("EVALUATION GLOBALE")
    print("=" * 60)

    results = model.evaluate(
        X_test,
        y_test,
        verbose=1,
    )

    test_loss = float(
        results[0]
    )

    test_accuracy = float(
        results[1]
    )

    print()

    print(
        f"Test Loss     : "
        f"{test_loss:.4f}"
    )

    print(
        f"Test Accuracy : "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Test Accuracy : "
        f"{test_accuracy * 100:.2f}%"
    )

    print()

    return (
        test_loss,
        test_accuracy,
    )


# ============================================================
# GENERATION DES PREDICTIONS
# ============================================================

def generate_predictions(
    model,
    X_test,
):

    print("=" * 60)
    print("GENERATION DES PREDICTIONS")
    print("=" * 60)

    probabilities = model.predict(
        X_test,
        verbose=1,
    )

    if probabilities.ndim != 2:

        raise ValueError(
            "Les probabilités générées "
            "doivent être une matrice 2D."
        )

    y_pred = np.argmax(
        probabilities,
        axis=1,
    )

    confidence = np.max(
        probabilities,
        axis=1,
    )

    print()

    print(
        f"Predictions générées : "
        f"{len(y_pred)}"
    )

    print()

    return (
        probabilities,
        y_pred,
        confidence,
    )


# ============================================================
# RAPPORT DE CLASSIFICATION
# ============================================================

def generate_classification_report(
    y_test,
    y_pred,
    id_to_label,
):

    print("=" * 60)
    print("RAPPORT DE CLASSIFICATION")
    print("=" * 60)

    class_ids = sorted(
        id_to_label.keys()
    )

    target_names = [
        id_to_label[class_id]
        for class_id in class_ids
    ]

    report = classification_report(
        y_test,
        y_pred,
        labels=class_ids,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )

    report_text = classification_report(
        y_test,
        y_pred,
        labels=class_ids,
        target_names=target_names,
        zero_division=0,
    )

    print()
    print(report_text)
    print()

    return (
        report,
        report_text,
    )


# ============================================================
# MATRICE DE CONFUSION
# ============================================================

def generate_confusion_matrix(
    y_test,
    y_pred,
    id_to_label,
):

    print("=" * 60)
    print("MATRICE DE CONFUSION")
    print("=" * 60)

    class_ids = sorted(
        id_to_label.keys()
    )

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=class_ids,
    )

    print()

    print(
        "Lignes = classes réelles"
    )

    print(
        "Colonnes = classes prédites"
    )

    print()

    print(
        "Classes :"
    )

    for class_id in class_ids:

        print(
            f"{class_id} -> "
            f"{id_to_label[class_id]}"
        )

    print()

    print(matrix)

    print()

    return matrix


# ============================================================
# METRIQUES DETAILLEES
# ============================================================

def calculate_global_metrics(
    y_test,
    y_pred,
):

    print("=" * 60)
    print("METRIQUES DETAILLEES")
    print("=" * 60)

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision_macro = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    recall_macro = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    f1_macro = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    precision_weighted = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    recall_weighted = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    f1_weighted = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print()

    print(
        f"Accuracy              : "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision macro       : "
        f"{precision_macro:.4f}"
    )

    print(
        f"Recall macro          : "
        f"{recall_macro:.4f}"
    )

    print(
        f"F1-score macro        : "
        f"{f1_macro:.4f}"
    )

    print()

    print(
        f"Precision weighted    : "
        f"{precision_weighted:.4f}"
    )

    print(
        f"Recall weighted       : "
        f"{recall_weighted:.4f}"
    )

    print(
        f"F1-score weighted     : "
        f"{f1_weighted:.4f}"
    )

    print()

    metrics = {
        "accuracy": float(accuracy),

        "precision_macro": float(
            precision_macro
        ),

        "recall_macro": float(
            recall_macro
        ),

        "f1_macro": float(
            f1_macro
        ),

        "precision_weighted": float(
            precision_weighted
        ),

        "recall_weighted": float(
            recall_weighted
        ),

        "f1_weighted": float(
            f1_weighted
        ),
    }

    return metrics


# ============================================================
# SAUVEGARDE DES RESULTATS
# ============================================================

def save_evaluation_results(
    test_loss,
    test_accuracy,
    metrics,
    report,
    confusion,
    id_to_label,
):

    print("=" * 60)
    print("SAUVEGARDE DES RESULTATS")
    print("=" * 60)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Métriques globales
    # --------------------------------------------------------

    evaluation = {

        "test_loss":
            float(test_loss),

        "test_accuracy":
            float(test_accuracy),

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

    metrics_path = (
        RESULTS_DIR
        / "evaluation_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            evaluation,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Rapport de classification
    # --------------------------------------------------------

    report_path = (
        RESULTS_DIR
        / "classification_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Matrice de confusion
    # --------------------------------------------------------

    confusion_data = {

        "labels": [
            id_to_label[class_id]
            for class_id
            in sorted(id_to_label)
        ],

        "matrix":
            confusion.tolist(),
    }

    confusion_path = (
        RESULTS_DIR
        / "confusion_matrix.json"
    )

    with open(
        confusion_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            confusion_data,
            file,
            indent=4,
        )

    print()

    print(
        f"Métriques sauvegardées : "
        f"{metrics_path}"
    )

    print(
        f"Rapport JSON sauvegardé : "
        f"{report_path}"
    )

    print(
        f"Matrice sauvegardée : "
        f"{confusion_path}"
    )

    print()


# ============================================================
# SAUVEGARDE DES PREDICTIONS
# ============================================================

def save_predictions(
    y_test,
    y_pred,
    probabilities,
    confidence,
    id_to_label,
):

    print("=" * 60)
    print("SAUVEGARDE DES PREDICTIONS")
    print("=" * 60)

    predictions = []

    for index in range(len(y_test)):

        real_id = int(
            y_test[index]
        )

        predicted_id = int(
            y_pred[index]
        )

        predictions.append(
            {
                "index": index,

                "real_class_id":
                    real_id,

                "real_label":
                    id_to_label[real_id],

                "predicted_class_id":
                    predicted_id,

                "predicted_label":
                    id_to_label[predicted_id],

                "confidence":
                    float(confidence[index]),

                "probability_non_vulnerable":
                    float(probabilities[index][0]),

                "probability_vulnerable":
                    float(probabilities[index][1]),

                "correct":
                    bool(
                        real_id == predicted_id
                    ),
            }
        )

    path = (
        RESULTS_DIR
        / "predictions.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            predictions,
            file,
            indent=4,
        )

    print()

    print(
        f"Predictions sauvegardées : "
        f"{path}"
    )

    print()


# ============================================================
# AFFICHAGE DES PREDICTIONS
# ============================================================

def show_predictions(
    y_test,
    y_pred,
    probabilities,
    confidence,
    id_to_label,
):

    print("=" * 60)
    print("EXEMPLES DE PREDICTIONS")
    print("=" * 60)

    number_to_show = min(
        10,
        len(y_test),
    )

    print()

    for index in range(number_to_show):

        real_id = int(
            y_test[index]
        )

        predicted_id = int(
            y_pred[index]
        )

        real_label = id_to_label[
            real_id
        ]

        predicted_label = id_to_label[
            predicted_id
        ]

        status = (
            "CORRECT"
            if real_id == predicted_id
            else "ERREUR"
        )

        print(
            f"[{status}] "
            f"Réel : "
            f"{real_label:<18} "
            f"| Prédit : "
            f"{predicted_label:<18} "
            f"| Confiance : "
            f"{confidence[index]:.2%}"
        )

    print()


# ============================================================
# RESUME FINAL
# ============================================================

def print_final_summary(
    test_accuracy,
    metrics,
):

    print("=" * 60)
    print("RESUME FINAL")
    print("=" * 60)

    print()

    print(
        f"Accuracy finale       : "
        f"{test_accuracy:.4f} "
        f"({test_accuracy * 100:.2f}%)"
    )

    print(
        f"Precision macro       : "
        f"{metrics['precision_macro']:.4f}"
    )

    print(
        f"Recall macro          : "
        f"{metrics['recall_macro']:.4f}"
    )

    print(
        f"F1-score macro        : "
        f"{metrics['f1_macro']:.4f}"
    )

    print(
        f"Precision weighted    : "
        f"{metrics['precision_weighted']:.4f}"
    )

    print(
        f"Recall weighted       : "
        f"{metrics['recall_weighted']:.4f}"
    )

    print(
        f"F1-score weighted     : "
        f"{metrics['f1_weighted']:.4f}"
    )

    print()

    print(
        "Résultats disponibles dans :"
    )

    print(
        RESULTS_DIR
    )

    print()


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("=" * 60)
    print("EVALUATION DU MODELE")
    print("DE DETECTION DES VULNERABILITES")
    print("DES SMART CONTRACTS")
    print("=" * 60)

    print()

    print(
        f"TensorFlow : "
        f"{tf.__version__}"
    )

    print()

    # --------------------------------------------------------
    # Vérification des fichiers
    # --------------------------------------------------------

    check_files()

    # --------------------------------------------------------
    # Chargement des données
    # --------------------------------------------------------

    X_test, y_test = (
        load_test_data()
    )

    # --------------------------------------------------------
    # Chargement du vocabulaire
    # --------------------------------------------------------

    vocabulary = load_vocabulary()

    # --------------------------------------------------------
    # Chargement des classes
    # --------------------------------------------------------

    id_to_label = load_labels()

    # --------------------------------------------------------
    # Chargement du modèle
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Vérification
    # --------------------------------------------------------

    validate_data(
        X_test,
        y_test,
        vocabulary,
        id_to_label,
        model,
    )

    # --------------------------------------------------------
    # Distribution des classes
    # --------------------------------------------------------

    show_class_distribution(
        y_test,
        id_to_label,
    )

    # --------------------------------------------------------
    # Evaluation globale
    # --------------------------------------------------------

    (
        test_loss,
        test_accuracy,
    ) = evaluate_basic_metrics(
        model,
        X_test,
        y_test,
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    (
        probabilities,
        y_pred,
        confidence,
    ) = generate_predictions(
        model,
        X_test,
    )

    # --------------------------------------------------------
    # Rapport de classification
    # --------------------------------------------------------

    (
        report,
        report_text,
    ) = generate_classification_report(
        y_test,
        y_pred,
        id_to_label,
    )

    # --------------------------------------------------------
    # Matrice de confusion
    # --------------------------------------------------------

    confusion = generate_confusion_matrix(
        y_test,
        y_pred,
        id_to_label,
    )

    # --------------------------------------------------------
    # Métriques détaillées
    # --------------------------------------------------------

    metrics = calculate_global_metrics(
        y_test,
        y_pred,
    )

    # --------------------------------------------------------
    # Sauvegarde des résultats
    # --------------------------------------------------------

    save_evaluation_results(
        test_loss,
        test_accuracy,
        metrics,
        report,
        confusion,
        id_to_label,
    )

    # --------------------------------------------------------
    # Sauvegarde du rapport texte
    # --------------------------------------------------------

    report_text_path = (
        RESULTS_DIR
        / "classification_report.txt"
    )

    with open(
        report_text_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            report_text
        )

    print(
        f"Rapport texte sauvegardé : "
        f"{report_text_path}"
    )

    print()

    # --------------------------------------------------------
    # Sauvegarde des prédictions
    # --------------------------------------------------------

    save_predictions(
        y_test,
        y_pred,
        probabilities,
        confidence,
        id_to_label,
    )

    # --------------------------------------------------------
    # Exemples de prédictions
    # --------------------------------------------------------

    show_predictions(
        y_test,
        y_pred,
        probabilities,
        confidence,
        id_to_label,
    )

    # --------------------------------------------------------
    # Résumé final
    # --------------------------------------------------------

    print_final_summary(
        test_accuracy,
        metrics,
    )

    print("=" * 60)
    print("EVALUATION TERMINEE")
    print("=" * 60)


# ============================================================
# POINT D'ENTREE
# ============================================================

if __name__ == "__main__":
    main()