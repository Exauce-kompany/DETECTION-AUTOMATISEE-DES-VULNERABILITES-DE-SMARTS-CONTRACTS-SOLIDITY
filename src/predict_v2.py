from pathlib import Path
import argparse
import json
import pickle
import sys

import numpy as np
import tensorflow as tf

from solidity_tokenizer import tokenize_solidity


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "smart_contract_vulnerability_model_v2.keras"
)

DATA_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "v2"
    / "prepared"
)

VOCABULARY_PATH = (
    DATA_DIR
    / "vocabulary.pkl"
)

LABELS_PATH = (
    DATA_DIR
    / "labels.json"
)

MAX_SEQUENCE_LENGTH = 512

PAD_ID = 0
UNK_ID = 1


# ============================================================
# CHARGEMENT DES RESSOURCES
# ============================================================

def load_resources():

    print("=" * 70)
    print("PREDICTION DE VULNERABILITE")
    print("SMART CONTRACT SOLIDITY - MODELE V2")
    print("=" * 70)

    required_files = [
        MODEL_PATH,
        VOCABULARY_PATH,
        LABELS_PATH,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Fichier introuvable : {path}"
            )

    print()
    print("Chargement du modèle V2...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Modèle chargé.")

    with open(
        VOCABULARY_PATH,
        "rb"
    ) as file:

        vocabulary = pickle.load(
            file
        )

    with open(
        LABELS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        labels = json.load(
            file
        )

    print(
        f"Vocabulaire : {len(vocabulary):,}"
    )

    print(
        f"Labels : {labels}"
    )

    return (
        model,
        vocabulary,
        labels,
    )


# ============================================================
# CHARGEMENT DU CONTRAT
# ============================================================

def load_contract(
    contract_path
):

    path = Path(
        contract_path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Contrat introuvable : {path}"
        )

    if not path.is_file():

        raise ValueError(
            f"Le chemin n'est pas un fichier : {path}"
        )

    if path.suffix.lower() != ".sol":

        print()
        print(
            "ATTENTION : le fichier ne possède "
            "pas l'extension .sol."
        )

    code = path.read_text(
        encoding="utf-8",
        errors="replace"
    )

    if not code.strip():

        raise ValueError(
            "Le fichier Solidity est vide."
        )

    return (
        path,
        code
    )


# ============================================================
# PREPARATION DU CODE
# ============================================================

def prepare_contract(
    code,
    vocabulary
):

    tokens = tokenize_solidity(
        code
    )

    if not tokens:

        raise ValueError(
            "Aucun token Solidity n'a été trouvé."
        )

    original_token_count = len(
        tokens
    )

    truncated = (
        original_token_count
        > MAX_SEQUENCE_LENGTH
    )

    tokens = tokens[
        :MAX_SEQUENCE_LENGTH
    ]

    token_ids = []

    unknown_count = 0

    for token in tokens:

        token_id = vocabulary.get(
            token,
            UNK_ID
        )

        if token_id == UNK_ID:
            unknown_count += 1

        token_ids.append(
            token_id
        )

    sequence = np.full(
        (
            1,
            MAX_SEQUENCE_LENGTH
        ),
        PAD_ID,
        dtype=np.int32
    )

    sequence[
        0,
        :len(token_ids)
    ] = token_ids

    unknown_rate = (
        unknown_count
        / len(tokens)
        if tokens
        else 0.0
    )

    preprocessing_info = {

        "original_tokens":
            original_token_count,

        "used_tokens":
            len(tokens),

        "truncated":
            truncated,

        "unknown_tokens":
            unknown_count,

        "unknown_rate":
            unknown_rate,
    }

    return (
        sequence,
        preprocessing_info,
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_contract(
    model,
    sequence,
    labels
):

    probabilities = model.predict(
        sequence,
        verbose=0
    )[0]

    predicted_label = int(
        np.argmax(
            probabilities
        )
    )

    confidence = float(
        probabilities[
            predicted_label
        ]
    )

    probability_non_vulnerable = float(
        probabilities[0]
    )

    probability_vulnerable = float(
        probabilities[1]
    )

    predicted_class = labels[
        str(predicted_label)
    ]

    return {

        "predicted_label":
            predicted_label,

        "predicted_class":
            predicted_class,

        "confidence":
            confidence,

        "probability_non_vulnerable":
            probability_non_vulnerable,

        "probability_vulnerable":
            probability_vulnerable,
    }


# ============================================================
# NIVEAU DE CONFIANCE
# ============================================================

def confidence_level(
    confidence
):

    if confidence >= 0.90:
        return "très élevée"

    if confidence >= 0.75:
        return "élevée"

    if confidence >= 0.60:
        return "modérée"

    return "faible"


# ============================================================
# AFFICHAGE
# ============================================================

def display_result(
    path,
    preprocessing,
    prediction
):

    predicted_class = (
        prediction[
            "predicted_class"
        ]
    )

    confidence = (
        prediction[
            "confidence"
        ]
    )

    probability_safe = (
        prediction[
            "probability_non_vulnerable"
        ]
    )

    probability_vulnerable = (
        prediction[
            "probability_vulnerable"
        ]
    )

    print()
    print("=" * 70)
    print("RESULTAT DE L'ANALYSE")
    print("=" * 70)

    print()
    print(
        f"Fichier : {path}"
    )

    print()
    print("Prétraitement :")

    print(
        f"  Tokens détectés : "
        f"{preprocessing['original_tokens']:,}"
    )

    print(
        f"  Tokens utilisés : "
        f"{preprocessing['used_tokens']:,}"
    )

    print(
        f"  Tronqué : "
        f"{'oui' if preprocessing['truncated'] else 'non'}"
    )

    print(
        f"  Tokens inconnus : "
        f"{preprocessing['unknown_tokens']:,}"
    )

    print(
        f"  Taux <UNK> : "
        f"{preprocessing['unknown_rate']:.2%}"
    )

    print()
    print("-" * 70)

    if predicted_class == "vulnerable":

        print(
            "DECISION : CONTRAT POTENTIELLEMENT VULNERABLE"
        )

    else:

        print(
            "DECISION : AUCUNE VULNERABILITE "
            "DETECTEE PAR LE MODELE"
        )

    print("-" * 70)

    print()
    print(
        f"Classe prédite : "
        f"{predicted_class}"
    )

    print(
        f"Confiance : "
        f"{confidence:.2%}"
    )

    print(
        f"Niveau de confiance : "
        f"{confidence_level(confidence)}"
    )

    print()
    print("Probabilités :")

    print(
        f"  non_vulnerable : "
        f"{probability_safe:.2%}"
    )

    print(
        f"  vulnerable     : "
        f"{probability_vulnerable:.2%}"
    )

    print()
    print(
        "IMPORTANT : cette prédiction est une "
        "aide à l'audit et ne remplace pas une "
        "analyse de sécurité complète."
    )


# ============================================================
# SAUVEGARDE OPTIONNELLE
# ============================================================

def save_prediction(
    path,
    preprocessing,
    prediction,
):

    output_dir = (
        PROJECT_ROOT
        / "results"
        / "v2"
        / "individual_predictions"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir
        / f"{path.stem}_prediction.json"
    )

    data = {

        "contract_file":
            str(path),

        "model":
            str(MODEL_PATH),

        "preprocessing":
            preprocessing,

        "prediction":
            prediction,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )

    print()
    print(
        f"Résultat sauvegardé : "
        f"{output_path}"
    )


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Analyse un smart contract Solidity "
            "avec le modèle BiLSTM V2."
        )
    )

    parser.add_argument(
        "contract",
        nargs="?",
        help=(
            "Chemin vers le fichier Solidity .sol"
        ),
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help=(
            "Sauvegarde la prédiction en JSON."
        ),
    )

    return parser.parse_args()


# ============================================================
# MODE INTERACTIF
# ============================================================

def ask_contract_path():

    print()
    print(
        "Entre le chemin du fichier Solidity "
        "à analyser :"
    )

    value = input(
        "> "
    ).strip()

    # Permettre à l'utilisateur de coller
    # un chemin entouré de guillemets.
    value = value.strip(
        '"'
    ).strip(
        "'"
    )

    return value


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()

    try:

        (
            model,
            vocabulary,
            labels,
        ) = load_resources()

        contract_path = (
            args.contract
            if args.contract
            else ask_contract_path()
        )

        if not contract_path:

            raise ValueError(
                "Aucun fichier spécifié."
            )

        path, code = load_contract(
            contract_path
        )

        (
            sequence,
            preprocessing,
        ) = prepare_contract(
            code,
            vocabulary
        )

        prediction = predict_contract(
            model,
            sequence,
            labels
        )

        display_result(
            path,
            preprocessing,
            prediction
        )

        if args.save:

            save_prediction(
                path,
                preprocessing,
                prediction
            )

        print()
        print("=" * 70)
        print("ANALYSE TERMINEE")
        print("=" * 70)

    except Exception as error:

        print()
        print("=" * 70)
        print("ERREUR")
        print("=" * 70)

        print(
            str(error)
        )

        sys.exit(1)


if __name__ == "__main__":
    main()