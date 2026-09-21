from pathlib import Path

import re

import pandas as pd


# ============================================================
# Configuration des chemins
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "smartbugs-curated"
    / "dataset"
)

PROCESSED_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "contracts.csv"
)

# ============================================================
# Contrats multi-vulnérabilités exclus
# ============================================================

# Ce contrat apparaît dans deux catégories différentes
# du dataset SmartBugs Curated :
#
#   - reentrancy
#   - unchecked_low_level_calls
#
# Comme notre modèle actuel effectue une classification
# mono-classe, le même contrat ne doit pas apparaître
# avec deux labels différents.
#
# Le fichier original reste conservé dans le dataset brut.
# Il est simplement exclu du dataset utilisé pour
# l'apprentissage.

EXCLUDED_FILES = {
    "0x627fa62ccbb1c1b04ffaecd72a53e37fc0e17839.sol"
}


# ============================================================
# Nettoyage du code Solidity
# ============================================================

def clean_solidity_code(code):

    """
    Supprime les commentaires Solidity afin d'éviter
    que les annotations de vulnérabilité contaminent
    les données utilisées par le modèle.
    """

    # Suppression des commentaires multilignes /* ... */
    code = re.sub(
        r"/\*.*?\*/",
        " ",
        code,
        flags=re.DOTALL
    )

    # Suppression des commentaires // ...
    code = re.sub(
        r"//.*",
        " ",
        code
    )

    # Normalisation des espaces
    code = re.sub(
        r"\s+",
        " ",
        code
    )

    return code.strip()


# ============================================================
# Lecture des contrats
# ============================================================

def load_contracts():

    records = []

    if not RAW_DATASET.exists():

        raise FileNotFoundError(
            f"Dataset introuvable : {RAW_DATASET}"
        )

    for vulnerability_dir in sorted(
        RAW_DATASET.iterdir()
    ):

        if not vulnerability_dir.is_dir():
            continue

        vulnerability = vulnerability_dir.name

        for contract_file in sorted(
            vulnerability_dir.glob("*.sol")
        ):

            # ------------------------------------------------
            # Exclusion des contrats multi-vulnérabilités
            # ------------------------------------------------

            if contract_file.name in EXCLUDED_FILES:

                print(
                    f"Contrat multi-vulnérabilités ignoré : "
                    f"{contract_file.name}"
                )

                continue

            try:

                code = contract_file.read_text(
                    encoding="utf-8",
                    errors="ignore"
                )

                clean_code = clean_solidity_code(
                    code
                )

                records.append({

                    "filename":
                        contract_file.name,

                    "vulnerability":
                        vulnerability,

                    "filepath":
                        str(
                            contract_file.relative_to(
                                PROJECT_ROOT
                            )
                        ),

                    "code":
                        code,

                    "code_clean":
                        clean_code,

                    "lines":
                        len(
                            code.splitlines()
                        ),

                    "characters":
                        len(code),

                    "clean_characters":
                        len(clean_code)

                })

            except Exception as error:

                print(
                    f"Erreur lors de la lecture de "
                    f"{contract_file}: {error}"
                )

    return records


# ============================================================
# Création du dataset
# ============================================================

def create_dataset():

    print(
        "Lecture des contrats Solidity..."
    )

    records = load_contracts()

    if not records:

        raise ValueError(
            "Aucun contrat Solidity n'a été trouvé."
        )

    dataframe = pd.DataFrame(
        records
    )

    PROCESSED_DATASET.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dataframe.to_csv(
        PROCESSED_DATASET,
        index=False,
        encoding="utf-8"
    )

    print()

    print(
        "Dataset créé avec succès."
    )

    print(
        f"Nombre total de contrats : "
        f"{len(dataframe)}"
    )

    print(
        f"Fichier : "
        f"{PROCESSED_DATASET}"
    )

    print()

    print(
        "Répartition des vulnérabilités:"
    )

    print(
        dataframe["vulnerability"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Vérification des doublons de fichiers
    # --------------------------------------------------------

    duplicate_filenames = (
        dataframe["filename"]
        .duplicated()
        .sum()
    )

    print()

    print(
        f"Filenames dupliqués : "
        f"{duplicate_filenames}"
    )

    if duplicate_filenames == 0:

        print(
            "Aucun fichier n'apparaît dans "
            "plusieurs classes."
        )

    else:

        print(
            "ATTENTION : certains fichiers "
            "apparaissent encore dans plusieurs classes."
        )


# ============================================================
# Programme principal
# ============================================================

if __name__ == "__main__":

    create_dataset()