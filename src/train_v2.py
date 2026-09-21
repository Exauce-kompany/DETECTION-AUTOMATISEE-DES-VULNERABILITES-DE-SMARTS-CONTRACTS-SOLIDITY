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
    / "v2"
    / "prepared"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "v2"
)

MODEL_PATH = (
    MODEL_DIR
    / "smart_contract_vulnerability_model_v2.keras"
)

RANDOM_STATE = 42

BATCH_SIZE = 32
EPOCHS = 30

MAX_SEQUENCE_LENGTH = 512
EMBEDDING_DIM = 64
LSTM_UNITS = 32


# ============================================================
# REPRODUCTIBILITE
# ============================================================

np.random.seed(
    RANDOM_STATE
)

tf.random.set_seed(
    RANDOM_STATE
)


# ============================================================
# CHARGEMENT DES DONNEES
# ============================================================

def load_data():

    print("=" * 70)
    print("ENTRAINEMENT DU MODELE V2")
    print("DETECTION DES VULNERABILITES")
    print("DES SMART CONTRACTS")
    print("=" * 70)

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

        path = (
            DATA_DIR
            / filename
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Fichier introuvable : {path}"
            )

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

    with open(
        DATA_DIR / "vocabulary.pkl",
        "rb"
    ) as file:

        vocabulary = pickle.load(
            file
        )

    with open(
        DATA_DIR / "labels.json",
        "r",
        encoding="utf-8"
    ) as file:

        labels = json.load(
            file
        )

    print()
    print("Dataset chargé :")

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

    print(
        f"Vocabulaire : "
        f"{len(vocabulary):,}"
    )

    print(
        f"Labels : {labels}"
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        vocabulary,
        labels,
    )


# ============================================================
# VERIFICATIONS
# ============================================================

def validate_data(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    vocabulary,
):

    print()
    print("=" * 70)
    print("VERIFICATION DES DONNEES")
    print("=" * 70)

    if X_train.shape[1] != MAX_SEQUENCE_LENGTH:

        raise ValueError(
            "Longueur de séquence incorrecte."
        )

    if X_val.shape[1] != MAX_SEQUENCE_LENGTH:

        raise ValueError(
            "Longueur validation incorrecte."
        )

    if X_test.shape[1] != MAX_SEQUENCE_LENGTH:

        raise ValueError(
            "Longueur test incorrecte."
        )

    train_labels = set(
        np.unique(y_train)
    )

    val_labels = set(
        np.unique(y_val)
    )

    test_labels = set(
        np.unique(y_test)
    )

    expected = {
        0,
        1
    }

    if train_labels != expected:
        raise ValueError(
            f"Labels train invalides : "
            f"{train_labels}"
        )

    if val_labels != expected:
        raise ValueError(
            f"Labels validation invalides : "
            f"{val_labels}"
        )

    if test_labels != expected:
        raise ValueError(
            f"Labels test invalides : "
            f"{test_labels}"
        )

    max_token = max(
        int(X_train.max()),
        int(X_val.max()),
        int(X_test.max()),
    )

    if max_token >= len(
        vocabulary
    ):

        raise ValueError(
            "Token ID hors vocabulaire."
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


# ============================================================
# DISTRIBUTION
# ============================================================

def print_distribution(
    y,
    name
):

    values, counts = np.unique(
        y,
        return_counts=True
    )

    print()
    print(
        f"Distribution {name} :"
    )

    for value, count in zip(
        values,
        counts
    ):

        percentage = (
            count
            / len(y)
            * 100
        )

        class_name = (
            "non_vulnerable"
            if int(value) == 0
            else "vulnerable"
        )

        print(
            f"  {int(value)} "
            f"{class_name:<16} "
            f"{int(count):,} "
            f"({percentage:.2f} %)"
        )


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(
    y_train
):

    classes = np.unique(
        y_train
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    class_weights = {
        int(class_id):
            float(weight)
        for class_id, weight
        in zip(
            classes,
            weights
        )
    }

    print()
    print("=" * 70)
    print("POIDS DES CLASSES")
    print("=" * 70)

    for class_id in sorted(
        class_weights
    ):

        count = int(
            (
                y_train
                == class_id
            ).sum()
        )

        print(
            f"Classe {class_id} : "
            f"{count:,} échantillons "
            f"-> poids "
            f"{class_weights[class_id]:.4f}"
        )

    return class_weights


# ============================================================
# MODELE
# ============================================================

def build_model(
    vocabulary_size
):

    print()
    print("=" * 70)
    print("CONSTRUCTION DU MODELE V2")
    print("=" * 70)

    inputs = tf.keras.Input(
        shape=(
            MAX_SEQUENCE_LENGTH,
        ),
        dtype=tf.int32,
        name="tokens",
    )

    x = tf.keras.layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=EMBEDDING_DIM,
        mask_zero=True,
        name="embedding",
    )(
        inputs
    )

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            LSTM_UNITS,
            return_sequences=True,
        ),
        name="bidirectional_lstm",
    )(
        x
    )

    x = tf.keras.layers.GlobalAveragePooling1D(
        name="global_average_pooling"
    )(
        x
    )

    x = tf.keras.layers.Dropout(
        0.40,
        name="dropout_1"
    )(
        x
    )

    x = tf.keras.layers.Dense(
        32,
        activation="relu",
        name="dense_1",
    )(
        x
    )

    x = tf.keras.layers.Dropout(
        0.30,
        name="dropout_2"
    )(
        x
    )

    outputs = tf.keras.layers.Dense(
        2,
        activation="softmax",
        name="classification",
    )(
        x
    )

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="smart_contract_vulnerability_v2",
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001
    )

    model.compile(
        optimizer=optimizer,
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
# CALLBACKS
# ============================================================

def create_callbacks():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    early_stopping = (
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        )
    )

    reduce_lr = (
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        )
    )

    checkpoint = (
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        )
    )

    return [
        early_stopping,
        reduce_lr,
        checkpoint,
    ]


# ============================================================
# SAUVEGARDE HISTORIQUE
# ============================================================

def save_history(
    history
):

    history_dict = {}

    for key, values in (
        history.history.items()
    ):

        history_dict[key] = [
            float(value)
            for value in values
        ]

    output_path = (
        RESULTS_DIR
        / "training_history_v2.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history_dict,
            file,
            indent=4
        )

    print()
    print(
        f"Historique sauvegardé : "
        f"{output_path}"
    )


# ============================================================
# TEST FINAL
# ============================================================

def evaluate_test(
    model,
    X_test,
    y_test,
):

    print()
    print("=" * 70)
    print("EVALUATION RAPIDE SUR TEST")
    print("=" * 70)

    test_loss, test_accuracy = (
        model.evaluate(
            X_test,
            y_test,
            batch_size=BATCH_SIZE,
            verbose=1,
        )
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

    metrics = {
        "test_loss":
            float(test_loss),

        "test_accuracy":
            float(test_accuracy),
    }

    output_path = (
        RESULTS_DIR
        / "test_metrics_v2.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print(
        f"Métriques sauvegardées : "
        f"{output_path}"
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
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        vocabulary,
        labels,
    ) = load_data()

    validate_data(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        vocabulary,
    )

    print_distribution(
        y_train,
        "TRAIN"
    )

    print_distribution(
        y_val,
        "VALIDATION"
    )

    print_distribution(
        y_test,
        "TEST"
    )

    class_weights = (
        calculate_class_weights(
            y_train
        )
    )

    model = build_model(
        len(vocabulary)
    )

    callbacks = (
        create_callbacks()
    )

    print()
    print("=" * 70)
    print("DEBUT DE L'ENTRAINEMENT V2")
    print("=" * 70)

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
        shuffle=True,
        verbose=1,
    )

    save_history(
        history
    )

    print()
    print("=" * 70)
    print("MEILLEUR MODELE")
    print("=" * 70)

    print(
        MODEL_PATH
    )

    evaluate_test(
        model,
        X_test,
        y_test,
    )

    print()
    print("=" * 70)
    print("ENTRAINEMENT V2 TERMINE")
    print("=" * 70)

    print()
    print(
        "Le modèle V1 n'a pas été modifié."
    )

    print(
        "Modèle V2 :"
    )

    print(
        MODEL_PATH
    )


if __name__ == "__main__":
    main()