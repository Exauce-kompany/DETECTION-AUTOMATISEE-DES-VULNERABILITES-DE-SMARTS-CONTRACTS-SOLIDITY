from pathlib import Path
import json
import pickle

import numpy as np

from solidity_tokenizer import tokenize_solidity


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Nouveau dataset
DATASET_ROOT = (
    PROJECT_ROOT.parent
    / "smart-contract-vuln-dataset"
    / "data"
    / "processed"
    / "balanced_stage1_resplit_721"
    / "has_vul_721_stratified_v1"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "prepared"
)

RANDOM_STATE = 42

MAX_SEQUENCE_LENGTH = 512
MAX_VOCAB_SIZE = 5000

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


# ============================================================
# CHARGEMENT DES FICHIERS JSON
# ============================================================

def load_json_file(filename):

    path = DATASET_ROOT / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )

    print(f"Chargement : {path}")

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            f"{filename} doit contenir une liste JSON."
        )

    print(
        f"  {len(data)} échantillons chargés."
    )

    return data


def load_dataset():

    print()
    print("=" * 60)
    print("CHARGEMENT DU NOUVEAU DATASET")
    print("=" * 60)

    train_data = load_json_file("train.json")
    val_data = load_json_file("val.json")
    test_data = load_json_file("test.json")

    return train_data, val_data, test_data


# ============================================================
# EXTRACTION DU CODE SOLIDITY
# ============================================================

def extract_code(sample):

    context = sample.get("context", [])

    if isinstance(context, list):

        return "\n".join(
            str(line)
            for line in context
        )

    if isinstance(context, str):

        return context

    return ""


# ============================================================
# VALIDATION DES LABELS
# ============================================================

def validate_labels(data, name):

    labels = set()

    for sample in data:

        if "has_vulnerability" not in sample:
            raise ValueError(
                f"Le champ 'has_vulnerability' "
                f"est absent dans {name}."
            )

        label = sample["has_vulnerability"]

        if label not in [0, 1]:
            raise ValueError(
                f"Label invalide dans {name}: {label}"
            )

        labels.add(label)

    print(
        f"{name} - labels présents : "
        f"{sorted(labels)}"
    )


# ============================================================
# TOKENISATION
# ============================================================

def tokenize_dataset(data, name):

    print()
    print(
        f"Tokenisation : {name}"
    )

    tokenized = []

    for index, sample in enumerate(data):

        code = extract_code(sample)

        tokens = tokenize_solidity(code)

        tokenized.append(
            {
                "tokens": tokens,
                "label": int(
                    sample["has_vulnerability"]
                ),
                "sample_id": sample.get(
                    "sample_id",
                    f"{name}_{index}"
                )
            }
        )

        if (index + 1) % 500 == 0:

            print(
                f"  {index + 1}/"
                f"{len(data)} échantillons traités"
            )

    print(
        f"  {len(data)} échantillons traités."
    )

    return tokenized


# ============================================================
# CONSTRUCTION DU VOCABULAIRE
# IMPORTANT :
# LE VOCABULAIRE EST CONSTRUIT UNIQUEMENT SUR TRAIN
# ============================================================

def build_vocabulary(train_data):

    print()
    print("=" * 60)
    print("CONSTRUCTION DU VOCABULAIRE")
    print("=" * 60)

    frequency = {}

    for sample in train_data:

        for token in sample["tokens"]:

            frequency[token] = (
                frequency.get(token, 0) + 1
            )

    sorted_tokens = sorted(
        frequency.items(),
        key=lambda item: item[1],
        reverse=True
    )

    vocabulary = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1
    }

    for token, _ in sorted_tokens:

        if len(vocabulary) >= MAX_VOCAB_SIZE:
            break

        if token not in vocabulary:

            vocabulary[token] = len(
                vocabulary
            )

    print(
        f"Taille du vocabulaire : "
        f"{len(vocabulary)}"
    )

    return vocabulary


# ============================================================
# CONVERSION TOKENS -> IDS
# ============================================================

def tokens_to_sequence(
    tokens,
    vocabulary
):

    sequence = []

    for token in tokens:

        token_id = vocabulary.get(
            token,
            vocabulary[UNK_TOKEN]
        )

        sequence.append(token_id)

    return sequence


# ============================================================
# PADDING / TRONCATURE
# ============================================================

def pad_sequence(sequence):

    sequence = sequence[
        :MAX_SEQUENCE_LENGTH
    ]

    if len(sequence) < MAX_SEQUENCE_LENGTH:

        sequence += [
            0
        ] * (
            MAX_SEQUENCE_LENGTH
            - len(sequence)
        )

    return sequence


# ============================================================
# DATASET -> MATRICES NUMPY
# ============================================================

def convert_to_arrays(
    data,
    vocabulary,
    name
):

    print()
    print(
        f"Conversion en matrices : {name}"
    )

    sequences = []
    labels = []

    for sample in data:

        sequence = tokens_to_sequence(
            sample["tokens"],
            vocabulary
        )

        sequence = pad_sequence(
            sequence
        )

        sequences.append(sequence)

        labels.append(
            sample["label"]
        )

    X = np.array(
        sequences,
        dtype=np.int32
    )

    y = np.array(
        labels,
        dtype=np.int32
    )

    print(
        f"X_{name} : {X.shape}"
    )

    print(
        f"y_{name} : {y.shape}"
    )

    return X, y


# ============================================================
# STATISTIQUES DES LABELS
# ============================================================

def print_label_statistics(
    y,
    name
):

    print()
    print(
        f"Distribution des classes - {name}"
    )

    unique, counts = np.unique(
        y,
        return_counts=True
    )

    for label, count in zip(
        unique,
        counts
    ):

        if label == 0:
            label_name = "Non vulnérable"
        else:
            label_name = "Vulnérable"

        print(
            f"  Classe {label} "
            f"({label_name}) : "
            f"{count}"
        )


# ============================================================
# SAUVEGARDE
# ============================================================

def save_data(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    vocabulary
):

    print()
    print("=" * 60)
    print("SAUVEGARDE DES DONNÉES")
    print("=" * 60)

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

    # Mapping binaire explicite
    labels = {
        "label_to_id": {
            "non_vulnerable": 0,
            "vulnerable": 1
        },
        "id_to_label": {
            "0": "non_vulnerable",
            "1": "vulnerable"
        }
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

    print()
    print(
        f"Données sauvegardées dans : "
        f"{OUTPUT_DIR}"
    )


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("=" * 60)
    print("PRÉPARATION DES DONNÉES")
    print("SMART CONTRACT VULNERABILITY DATASET")
    print("=" * 60)

    print()
    print(
        f"Dataset source :\n"
        f"{DATASET_ROOT}"
    )

    if not DATASET_ROOT.exists():

        raise FileNotFoundError(
            "\nLe dossier du nouveau dataset "
            "est introuvable :\n"
            f"{DATASET_ROOT}\n\n"
            "Vérifie que le dépôt "
            "smart-contract-vuln-dataset "
            "se trouve bien dans le dossier Documents."
        )

    # --------------------------------------------------------
    # Chargement
    # --------------------------------------------------------

    (
        train_raw,
        val_raw,
        test_raw
    ) = load_dataset()

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("VÉRIFICATION DES LABELS")
    print("=" * 60)

    validate_labels(
        train_raw,
        "TRAIN"
    )

    validate_labels(
        val_raw,
        "VALIDATION"
    )

    validate_labels(
        test_raw,
        "TEST"
    )

    # --------------------------------------------------------
    # Tokenisation
    # --------------------------------------------------------

    train_data = tokenize_dataset(
        train_raw,
        "TRAIN"
    )

    val_data = tokenize_dataset(
        val_raw,
        "VALIDATION"
    )

    test_data = tokenize_dataset(
        test_raw,
        "TEST"
    )

    # --------------------------------------------------------
    # Vocabulaire
    # --------------------------------------------------------

    vocabulary = build_vocabulary(
        train_data
    )

    # --------------------------------------------------------
    # Conversion
    # --------------------------------------------------------

    X_train, y_train = convert_to_arrays(
        train_data,
        vocabulary,
        "train"
    )

    X_val, y_val = convert_to_arrays(
        val_data,
        vocabulary,
        "val"
    )

    X_test, y_test = convert_to_arrays(
        test_data,
        vocabulary,
        "test"
    )

    # --------------------------------------------------------
    # Statistiques
    # --------------------------------------------------------

    print_label_statistics(
        y_train,
        "TRAIN"
    )

    print_label_statistics(
        y_val,
        "VALIDATION"
    )

    print_label_statistics(
        y_test,
        "TEST"
    )

    # --------------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------------

    save_data(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        vocabulary
    )

    # --------------------------------------------------------
    # Résumé
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PRÉPARATION TERMINÉE")
    print("=" * 60)

    print(
        f"\nX_train : {X_train.shape}"
    )

    print(
        f"X_val   : {X_val.shape}"
    )

    print(
        f"X_test  : {X_test.shape}"
    )

    print(
        f"\nVocabulaire : "
        f"{len(vocabulary)} tokens"
    )

    print(
        "\nClasses : 2"
    )

    print(
        "\n0 = non vulnérable"
    )

    print(
        "1 = vulnérable"
    )


if __name__ == "__main__":
    main()