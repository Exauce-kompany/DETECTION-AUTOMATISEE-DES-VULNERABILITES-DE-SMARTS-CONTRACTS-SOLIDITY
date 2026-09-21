from pathlib import Path
import json
import pickle

import numpy as np
import tensorflow as tf

import sys


# ============================================================
# CHEMINS DU PROJET
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR)
    )

from solidity_tokenizer import tokenize_solidity


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
# CLASSE PRINCIPALE
# ============================================================

class SmartContractPredictor:

    def __init__(self):

        self.model = None
        self.vocabulary = None
        self.labels = None

        self.load_resources()


    # ========================================================
    # CHARGEMENT DU MODELE
    # ========================================================

    def load_resources(self):

        print("=" * 60)
        print("CHARGEMENT DU MOTEUR DE PREDICTION V2")
        print("=" * 60)

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

        print(
            f"Modèle : {MODEL_PATH}"
        )

        self.model = (
            tf.keras.models.load_model(
                MODEL_PATH
            )
        )

        print(
            "Modèle V2 chargé."
        )

        with open(
            VOCABULARY_PATH,
            "rb"
        ) as file:

            self.vocabulary = (
                pickle.load(file)
            )

        print(
            f"Vocabulaire : "
            f"{len(self.vocabulary):,}"
        )

        with open(
            LABELS_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            self.labels = json.load(
                file
            )

        print(
            f"Labels : {self.labels}"
        )

        print(
            "Moteur prêt."
        )


    # ========================================================
    # PREPARATION DU CODE
    # ========================================================

    def prepare_code(
        self,
        code
    ):

        if not isinstance(
            code,
            str
        ):

            raise ValueError(
                "Le code Solidity doit être "
                "une chaîne de caractères."
            )

        if not code.strip():

            raise ValueError(
                "Le code Solidity est vide."
            )

        tokens = tokenize_solidity(
            code
        )

        if not tokens:

            raise ValueError(
                "Aucun token Solidity détecté."
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

        sequence = np.full(
            (
                1,
                MAX_SEQUENCE_LENGTH
            ),
            PAD_ID,
            dtype=np.int32
        )

        unknown_count = 0

        for index, token in enumerate(
            tokens
        ):

            token_id = (
                self.vocabulary.get(
                    token,
                    UNK_ID
                )
            )

            if token_id == UNK_ID:
                unknown_count += 1

            sequence[
                0,
                index
            ] = token_id

        unknown_rate = (
            unknown_count
            / len(tokens)
            if tokens
            else 0.0
        )

        preprocessing = {

            "tokens_detected":
                original_token_count,

            "tokens_used":
                len(tokens),

            "truncated":
                truncated,

            "unknown_tokens":
                unknown_count,

            "unknown_rate":
                float(
                    unknown_rate
                ),
        }

        return (
            sequence,
            preprocessing
        )


    # ========================================================
    # NIVEAU DE CONFIANCE
    # ========================================================

    @staticmethod
    def confidence_level(
        confidence
    ):

        if confidence >= 0.90:
            return "Très élevée"

        if confidence >= 0.75:
            return "Élevée"

        if confidence >= 0.60:
            return "Modérée"

        return "Faible"


    # ========================================================
    # PREDICTION
    # ========================================================

    def predict(
        self,
        code
    ):

        (
            sequence,
            preprocessing
        ) = self.prepare_code(
            code
        )

        probabilities = (
            self.model.predict(
                sequence,
                verbose=0
            )[0]
        )

        predicted_label = int(
            np.argmax(
                probabilities
            )
        )

        predicted_class = (
            self.labels[
                str(
                    predicted_label
                )
            ]
        )

        confidence = float(
            probabilities[
                predicted_label
            ]
        )

        probability_non_vulnerable = (
            float(
                probabilities[0]
            )
        )

        probability_vulnerable = (
            float(
                probabilities[1]
            )
        )

        # ----------------------------------------------------
        # Texte destiné à l'interface
        # ----------------------------------------------------

        if predicted_label == 1:

            verdict = (
                "Contrat potentiellement vulnérable"
            )

            message = (
                "Le modèle a détecté des "
                "caractéristiques associées à des "
                "smart contracts vulnérables."
            )

        else:

            verdict = (
                "Aucune vulnérabilité détectée"
            )

            message = (
                "Le modèle n'a pas détecté de "
                "caractéristiques suffisantes pour "
                "classer ce contrat comme vulnérable."
            )

        result = {

            "success":
                True,

            "predicted_label":
                predicted_label,

            "predicted_class":
                predicted_class,

            "verdict":
                verdict,

            "message":
                message,

            "confidence":
                confidence,

            "confidence_percent":
                round(
                    confidence * 100,
                    2
                ),

            "confidence_level":
                self.confidence_level(
                    confidence
                ),

            "probability_non_vulnerable":
                probability_non_vulnerable,

            "probability_non_vulnerable_percent":
                round(
                    probability_non_vulnerable
                    * 100,
                    2
                ),

            "probability_vulnerable":
                probability_vulnerable,

            "probability_vulnerable_percent":
                round(
                    probability_vulnerable
                    * 100,
                    2
                ),

            "preprocessing":
                preprocessing,

            "model_info": {

                "version":
                    "V2",

                "architecture":
                    "BiLSTM",

                "sequence_length":
                    MAX_SEQUENCE_LENGTH,

                "vocabulary_size":
                    len(
                        self.vocabulary
                    ),
            },

            "warning":
                (
                    "Cette prédiction constitue une "
                    "aide à l'audit et ne remplace "
                    "pas une analyse de sécurité "
                    "complète du smart contract."
                ),
        }

        return result


# ============================================================
# INSTANCE UNIQUE DU MODELE
# ============================================================

predictor = SmartContractPredictor()