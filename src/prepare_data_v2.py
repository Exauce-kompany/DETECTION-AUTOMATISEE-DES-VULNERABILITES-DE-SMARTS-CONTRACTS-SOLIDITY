from pathlib import Path
from collections import Counter
import json
import pickle

import numpy as np

try:
    from .solidity_tokenizer import tokenize_solidity
except ImportError:
    from solidity_tokenizer import tokenize_solidity


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "v2"
    / "raw"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "v2"
    / "prepared"
)

MAX_SEQUENCE_LENGTH = 512
MAX_VOCAB_SIZE = 5000

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

PAD_ID = 0
UNK_ID = 1


# ============================================================
# CHARGEMENT
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


def load_dataset():

    print("=" * 70)
    print("PREPARATION DU DATASET V2")
    print("DETECTION BINAIRE DES VULNERABILITES")
    print("=" * 70)

    train = load_json(
        INPUT_DIR / "train.json"
    )

    val = load_json(
        INPUT_DIR / "val.json"
    )

    test = load_json(
        INPUT_DIR / "test.json"
    )

    print()
    print("Splits chargés :")

    print(
        f"Train      : {len(train):,}"
    )

    print(
        f"Validation : {len(val):,}"
    )

    print(
        f"Test       : {len(test):,}"
    )

    return train, val, test


# ============================================================
# CODE SOURCE
# ============================================================

def sample_to_code(sample):

    context = sample.get(
        "context",
        []
    )

    if isinstance(context, list):

        return "\n".join(
            str(line)
            for line in context
        )

    if isinstance(context, str):
        return context

    return ""


# ============================================================
# TOKENISATION
# ============================================================

def tokenize_samples(
    samples,
    split_name
):

    print()
    print(
        f"Tokenisation : {split_name}"
    )

    tokenized = []
    labels = []

    empty_samples = 0

    total = len(samples)

    for index, sample in enumerate(
        samples,
        start=1
    ):

        code = sample_to_code(
            sample
        )

        tokens = tokenize_solidity(
            code
        )

        if not tokens:
            empty_samples += 1

        tokenized.append(
            tokens
        )

        label = int(
            sample[
                "has_vulnerability"
            ]
        )

        if label not in {
            0,
            1
        }:

            raise ValueError(
                f"Label invalide : {label}"
            )

        labels.append(
            label
        )

        if (
            index % 2000 == 0
            or index == total
        ):

            print(
                f"  {index:,}/{total:,}"
            )

    print(
        f"Échantillons sans tokens : "
        f"{empty_samples:,}"
    )

    return (
        tokenized,
        np.asarray(
            labels,
            dtype=np.int32
        ),
    )


# ============================================================
# VOCABULAIRE
# ============================================================

def build_vocabulary(
    tokenized_train
):

    print()
    print("=" * 70)
    print("CONSTRUCTION DU VOCABULAIRE V2")
    print("=" * 70)

    counter = Counter()

    for tokens in tokenized_train:
        counter.update(tokens)

    vocabulary = {
        PAD_TOKEN: PAD_ID,
        UNK_TOKEN: UNK_ID,
    }

    # 5000 inclut PAD et UNK.
    available_slots = (
        MAX_VOCAB_SIZE
        - len(vocabulary)
    )

    most_common = counter.most_common(
        available_slots
    )

    for token, _ in most_common:

        if token not in vocabulary:

            vocabulary[token] = (
                len(vocabulary)
            )

    print(
        f"Tokens distincts observés : "
        f"{len(counter):,}"
    )

    print(
        f"Taille du vocabulaire conservé : "
        f"{len(vocabulary):,}"
    )

    print(
        f"{PAD_TOKEN} = {PAD_ID}"
    )

    print(
        f"{UNK_TOKEN} = {UNK_ID}"
    )

    return vocabulary


# ============================================================
# ENCODAGE
# ============================================================

def encode_tokens(
    tokenized_samples,
    vocabulary,
    split_name
):

    print()
    print(
        f"Encodage : {split_name}"
    )

    total = len(
        tokenized_samples
    )

    X = np.zeros(
        (
            total,
            MAX_SEQUENCE_LENGTH
        ),
        dtype=np.int32
    )

    unknown_count = 0
    token_count = 0
    truncated_samples = 0

    for index, tokens in enumerate(
        tokenized_samples
    ):

        if len(tokens) > MAX_SEQUENCE_LENGTH:
            truncated_samples += 1

        truncated = tokens[
            :MAX_SEQUENCE_LENGTH
        ]

        ids = []

        for token in truncated:

            token_id = vocabulary.get(
                token,
                UNK_ID
            )

            if token_id == UNK_ID:
                unknown_count += 1

            token_count += 1

            ids.append(
                token_id
            )

        if ids:

            X[
                index,
                :len(ids)
            ] = ids

    unknown_rate = (
        unknown_count / token_count
        if token_count
        else 0.0
    )

    print(
        f"Shape : {X.shape}"
    )

    print(
        f"Séquences tronquées : "
        f"{truncated_samples:,}"
    )

    print(
        f"Tokens inconnus : "
        f"{unknown_count:,}"
    )

    print(
        f"Taux <UNK> : "
        f"{unknown_rate:.4%}"
    )

    return X


# ============================================================
# DISTRIBUTION DES LABELS
# ============================================================

def print_distribution(
    y,
    split_name
):

    values, counts = np.unique(
        y,
        return_counts=True
    )

    print()
    print(
        f"Distribution {split_name} :"
    )

    for label, count in zip(
        values,
        counts
    ):

        class_name = (
            "non_vulnerable"
            if int(label) == 0
            else "vulnerable"
        )

        percentage = (
            count
            / len(y)
            * 100
        )

        print(
            f"  {int(label)} "
            f"{class_name:<16} : "
            f"{int(count):,} "
            f"({percentage:.2f} %)"
        )


# ============================================================
# SAUVEGARDE
# ============================================================

def save_prepared_data(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    vocabulary
):

    print()
    print("=" * 70)
    print("SAUVEGARDE")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        OUTPUT_DIR / "X_train.npy",
        X_train
    )

    np.save(
        OUTPUT_DIR / "y_train.npy",
        y_train
    )

    np.save(
        OUTPUT_DIR / "X_val.npy",
        X_val
    )

    np.save(
        OUTPUT_DIR / "y_val.npy",
        y_val
    )

    np.save(
        OUTPUT_DIR / "X_test.npy",
        X_test
    )

    np.save(
        OUTPUT_DIR / "y_test.npy",
        y_test
    )

    with open(
        OUTPUT_DIR / "vocabulary.pkl",
        "wb"
    ) as file:

        pickle.dump(
            vocabulary,
            file
        )

    labels = {
        "0": "non_vulnerable",
        "1": "vulnerable",
    }

    with open(
        OUTPUT_DIR / "labels.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            labels,
            file,
            indent=4,
            ensure_ascii=False
        )

    metadata = {
        "dataset_version":
            "v2_cgt",

        "max_sequence_length":
            MAX_SEQUENCE_LENGTH,

        "max_vocab_size":
            MAX_VOCAB_SIZE,

        "vocabulary_size":
            len(vocabulary),

        "train_samples":
            int(len(y_train)),

        "validation_samples":
            int(len(y_val)),

        "test_samples":
            int(len(y_test)),

        "train_non_vulnerable":
            int(
                (y_train == 0).sum()
            ),

        "train_vulnerable":
            int(
                (y_train == 1).sum()
            ),

        "validation_non_vulnerable":
            int(
                (y_val == 0).sum()
            ),

        "validation_vulnerable":
            int(
                (y_val == 1).sum()
            ),

        "test_non_vulnerable":
            int(
                (y_test == 0).sum()
            ),

        "test_vulnerable":
            int(
                (y_test == 1).sum()
            ),
    }

    with open(
        OUTPUT_DIR
        / "preparation_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print(
        f"Dataset préparé dans :"
    )

    print(
        OUTPUT_DIR
    )


# ============================================================
# VERIFICATIONS FINALES
# ============================================================

def final_checks(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    vocabulary
):

    print()
    print("=" * 70)
    print("VERIFICATIONS FINALES")
    print("=" * 70)

    if not vocabulary or len(vocabulary) > MAX_VOCAB_SIZE:
        raise ValueError("Taille du vocabulaire invalide.")
    if set(vocabulary.values()) != set(range(len(vocabulary))):
        raise ValueError("Les IDs du vocabulaire doivent être contigus.")
    for name, X, y in (("train", X_train, y_train), ("validation", X_val, y_val), ("test", X_test, y_test)):
        if X.ndim != 2 or X.shape[1] != MAX_SEQUENCE_LENGTH or y.ndim != 1 or len(X) != len(y) or not len(y):
            raise ValueError(f"Dimensions invalides : {name}.")
        if not np.issubdtype(X.dtype, np.integer) or not np.issubdtype(y.dtype, np.integer):
            raise ValueError(f"Types non entiers : {name}.")
        if set(np.unique(y)) != {0, 1}:
            raise ValueError(f"Les deux classes binaires sont requises : {name}.")
        if X.min() < 0 or X.max() >= len(vocabulary):
            raise ValueError(f"Token ID hors vocabulaire : {name}.")
    print("Dimensions dynamiques, labels et IDs : OK")


# ============================================================
# MAIN
# ============================================================

def main():

    train, val, test = (
        load_dataset()
    )

    tokenized_train, y_train = (
        tokenize_samples(
            train,
            "TRAIN"
        )
    )

    tokenized_val, y_val = (
        tokenize_samples(
            val,
            "VALIDATION"
        )
    )

    tokenized_test, y_test = (
        tokenize_samples(
            test,
            "TEST"
        )
    )

    vocabulary = build_vocabulary(
        tokenized_train
    )

    print()
    print("=" * 70)
    print("ENCODAGE DES SEQUENCES")
    print("=" * 70)

    X_train = encode_tokens(
        tokenized_train,
        vocabulary,
        "TRAIN"
    )

    X_val = encode_tokens(
        tokenized_val,
        vocabulary,
        "VALIDATION"
    )

    X_test = encode_tokens(
        tokenized_test,
        vocabulary,
        "TEST"
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

    final_checks(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        vocabulary
    )

    save_prepared_data(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        vocabulary
    )

    print()
    print("=" * 70)
    print("PREPARATION V2 TERMINEE AVEC SUCCES")
    print("=" * 70)

    print()
    print(
        "Le dataset V1 n'a pas été modifié."
    )

    print(
        "Aucun entraînement n'a encore été lancé."
    )


if __name__ == "__main__":
    main()
