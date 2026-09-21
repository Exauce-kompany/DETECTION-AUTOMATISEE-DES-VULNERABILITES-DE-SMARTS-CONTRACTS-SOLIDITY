from pathlib import Path
import json
import pickle

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight


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

RANDOM_STATE = 42

BATCH_SIZE = 32
EPOCHS = 30

MAX_SEQUENCE_LENGTH = 512

EMBEDDING_DIM = 64
LSTM_UNITS = 32


# Reproductibilité
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# VÉRIFICATION DES FICHIERS
# ============================================================

def check_data_files():

    required_files = [
        "X_train.npy",
        "y_train.npy",
        "X_val.npy",
        "y_val.npy",
        "X_test.npy",
        "y_test.npy",
        "vocabulary.pkl",
        "labels.json",
    ]

    for filename in required_files:

        path = DATA_DIR / filename

        if not path.exists():

            raise FileNotFoundError(
                f"Fichier manquant : {path}"
            )


# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

def load_data():

    print("=" * 60)
    print("CHARGEMENT DES DONNEES")
    print("=" * 60)

    check_data_files()

    X_train = np.load(
        DATA_DIR / "X_train.npy"
    )

    y_train = np.load(
        DATA_DIR / "y_train.npy"
    )

    X_val = np.load(
        DATA_DIR / "X_val.npy"
    )

    y_val = np.load(
        DATA_DIR / "y_val.npy"
    )

    X_test = np.load(
        DATA_DIR / "X_test.npy"
    )

    y_test = np.load(
        DATA_DIR / "y_test.npy"
    )

    print(
        f"X_train : {X_train.shape}"
    )

    print(
        f"y_train : {y_train.shape}"
    )

    print(
        f"X_val   : {X_val.shape}"
    )

    print(
        f"y_val   : {y_val.shape}"
    )

    print(
        f"X_test  : {X_test.shape}"
    )

    print(
        f"y_test  : {y_test.shape}"
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )


# ============================================================
# VOCABULAIRE
# ============================================================

def load_vocabulary():

    path = DATA_DIR / "vocabulary.pkl"

    with open(
        path,
        "rb"
    ) as file:

        vocabulary = pickle.load(file)

    print()
    print(
        f"Taille du vocabulaire : "
        f"{len(vocabulary)}"
    )

    return vocabulary


# ============================================================
# LABELS
# ============================================================

def load_labels():

    path = DATA_DIR / "labels.json"

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        labels = json.load(file)

    print()
    print("CLASSES")

    for class_id, label in labels[
        "id_to_label"
    ].items():

        print(
            f"  {class_id} -> {label}"
        )

    return labels


# ============================================================
# VÉRIFICATION DES DONNÉES
# ============================================================

def validate_data(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    vocabulary,
    labels,
):

    print()
    print("=" * 60)
    print("VERIFICATION DES DONNEES")
    print("=" * 60)

    datasets = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
    }

    # --------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------

    for name, X in datasets.items():

        if X.ndim != 2:

            raise ValueError(
                f"{name} doit etre une matrice 2D."
            )

        if X.shape[1] != MAX_SEQUENCE_LENGTH:

            raise ValueError(
                f"{name} doit avoir "
                f"{MAX_SEQUENCE_LENGTH} tokens."
            )

    # --------------------------------------------------------
    # Correspondance X / y
    # --------------------------------------------------------

    if len(X_train) != len(y_train):

        raise ValueError(
            "X_train et y_train sont incompatibles."
        )

    if len(X_val) != len(y_val):

        raise ValueError(
            "X_val et y_val sont incompatibles."
        )

    if len(X_test) != len(y_test):

        raise ValueError(
            "X_test et y_test sont incompatibles."
        )

    # --------------------------------------------------------
    # Nombre de classes
    # --------------------------------------------------------

    num_classes = len(
        labels["id_to_label"]
    )

    if num_classes != 2:

        raise ValueError(
            f"2 classes attendues, "
            f"{num_classes} trouvees."
        )

    # --------------------------------------------------------
    # Valeurs des labels
    # --------------------------------------------------------

    all_labels = np.concatenate(
        [
            y_train,
            y_val,
            y_test
        ]
    )

    unique_labels = np.unique(
        all_labels
    )

    if not np.array_equal(
        unique_labels,
        np.array([0, 1])
    ):

        raise ValueError(
            "Les labels doivent etre "
            "exactement [0, 1]."
        )

    # --------------------------------------------------------
    # Vérification du vocabulaire
    # --------------------------------------------------------

    max_token_id = int(
        max(
            X_train.max(),
            X_val.max(),
            X_test.max(),
        )
    )

    if max_token_id >= len(vocabulary):

        raise ValueError(
            "Un identifiant de token "
            "depasse la taille du vocabulaire."
        )

    # --------------------------------------------------------
    # Résumé
    # --------------------------------------------------------

    print(
        "Verification reussie."
    )

    print(
        f"Classes : {num_classes}"
    )

    print(
        "0 = non_vulnerable"
    )

    print(
        "1 = vulnerable"
    )

    print(
        f"Longueur des sequences : "
        f"{MAX_SEQUENCE_LENGTH}"
    )

    print(
        f"Token ID maximum : "
        f"{max_token_id}"
    )


# ============================================================
# POIDS DES CLASSES
# ============================================================

def compute_class_weights(y_train):

    print()
    print("=" * 60)
    print("POIDS DES CLASSES")
    print("=" * 60)

    classes = np.unique(
        y_train
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    class_weights = {
        int(class_id): float(weight)
        for class_id, weight in zip(
            classes,
            weights
        )
    }

    for class_id in classes:

        count = int(
            np.sum(
                y_train == class_id
            )
        )

        print(
            f"Classe {class_id} : "
            f"{count} exemples -> "
            f"poids "
            f"{class_weights[int(class_id)]:.4f}"
        )

    return class_weights


# ============================================================
# CONSTRUCTION DU MODÈLE
# ============================================================

def build_model(
    vocab_size,
    num_classes
):

    print()
    print("=" * 60)
    print("CONSTRUCTION DU MODELE")
    print("=" * 60)

    model = tf.keras.Sequential(
        [

            tf.keras.layers.Input(
                shape=(
                    MAX_SEQUENCE_LENGTH,
                ),
                dtype=tf.int32,
            ),

            tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=EMBEDDING_DIM,
                mask_zero=True,
            ),

            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    LSTM_UNITS,
                    return_sequences=True,
                )
            ),

            tf.keras.layers.GlobalAveragePooling1D(),

            tf.keras.layers.Dropout(
                0.40
            ),

            tf.keras.layers.Dense(
                32,
                activation="relu",
            ),

            tf.keras.layers.Dropout(
                0.30
            ),

            tf.keras.layers.Dense(
                num_classes,
                activation="softmax",
            ),
        ]
    )

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss=(
            "sparse_categorical_crossentropy"
        ),

        metrics=[
            "accuracy"
        ],
    )

    model.summary()

    return model


# ============================================================
# ENTRAÎNEMENT
# ============================================================

def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    class_weights,
):

    print()
    print("=" * 60)
    print("ENTRAINEMENT")
    print("=" * 60)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        MODEL_DIR
        / "smart_contract_vulnerability_model.keras"
    )

    callbacks = [

        tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

            patience=6,

            restore_best_weights=True,

            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.5,

            patience=3,

            min_lr=1e-6,

            verbose=1,
        ),

        tf.keras.callbacks.ModelCheckpoint(

            filepath=str(
                model_path
            ),

            monitor="val_loss",

            save_best_only=True,

            verbose=1,
        ),
    ]

    history = model.fit(

        X_train,

        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        class_weight=class_weights,

        callbacks=callbacks,

        verbose=1,
    )

    return history


# ============================================================
# SAUVEGARDE HISTORIQUE
# ============================================================

def save_history(history):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    path = (
        RESULTS_DIR
        / "training_history.json"
    )

    data = {

        key: [
            float(value)
            for value in values
        ]

        for key, values
        in history.history.items()
    }

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )

    print()
    print(
        f"Historique sauvegarde : "
        f"{path}"
    )


# ============================================================
# ÉVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    print()
    print("=" * 60)
    print("EVALUATION SUR LE JEU DE TEST")
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

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    path = (
        RESULTS_DIR
        / "test_metrics.json"
    )

    metrics = {

        "test_loss":
            test_loss,

        "test_accuracy":
            test_accuracy,
    }

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print()
    print(
        f"Metriques sauvegardees : "
        f"{path}"
    )


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("=" * 60)
    print("DETECTION DES VULNERABILITES")
    print("DES SMART CONTRACTS")
    print("=" * 60)

    print()
    print(
        f"TensorFlow : "
        f"{tf.__version__}"
    )

    # --------------------------------------------------------
    # Chargement
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = load_data()

    vocabulary = load_vocabulary()

    labels = load_labels()

    # --------------------------------------------------------
    # Vérification
    # --------------------------------------------------------

    validate_data(

        X_train,
        y_train,

        X_val,
        y_val,

        X_test,
        y_test,

        vocabulary,
        labels,
    )

    # --------------------------------------------------------
    # Paramètres
    # --------------------------------------------------------

    num_classes = len(
        labels["id_to_label"]
    )

    vocab_size = len(
        vocabulary
    )

    # --------------------------------------------------------
    # Poids
    # --------------------------------------------------------

    class_weights = (
        compute_class_weights(
            y_train
        )
    )

    # --------------------------------------------------------
    # Modèle
    # --------------------------------------------------------

    model = build_model(
        vocab_size,
        num_classes,
    )

    # --------------------------------------------------------
    # Entraînement
    # --------------------------------------------------------

    history = train_model(

        model,

        X_train,
        y_train,

        X_val,
        y_val,

        class_weights,
    )

    # --------------------------------------------------------
    # Historique
    # --------------------------------------------------------

    save_history(
        history
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    evaluate_model(
        model,
        X_test,
        y_test,
    )

    print()
    print("=" * 60)
    print("ENTRAINEMENT TERMINE")
    print("=" * 60)


if __name__ == "__main__":
    main()