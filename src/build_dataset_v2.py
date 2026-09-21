from pathlib import Path
import json

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset V1
V1_DIR = (
    PROJECT_ROOT.parent
    / "smart-contract-vuln-dataset"
    / "data"
    / "processed"
    / "balanced_stage1_resplit_721"
    / "has_vul_721_stratified_v1"
)

# Résultat de notre audit CGT
CGT_CANDIDATES_FILE = (
    PROJECT_ROOT
    / "results"
    / "cgt_overlap_audit"
    / "cgt_unique_final.csv"
)

# Nouveau dataset V2
V2_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "v2"
    / "raw"
)

# Minimum de types évalués pour accepter un négatif CGT
MIN_NEGATIVE_COVERAGE = 4


# ============================================================
# CHARGEMENT JSON
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


# ============================================================
# SAUVEGARDE JSON
# ============================================================

def save_json(data, path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# CHARGEMENT V1
# ============================================================

def load_v1():

    print()
    print("=" * 70)
    print("CHARGEMENT DU DATASET V1")
    print("=" * 70)

    train = load_json(
        V1_DIR / "train.json"
    )

    val = load_json(
        V1_DIR / "val.json"
    )

    test = load_json(
        V1_DIR / "test.json"
    )

    print(
        f"Train V1 : {len(train):,}"
    )

    print(
        f"Validation V1 : {len(val):,}"
    )

    print(
        f"Test V1 : {len(test):,}"
    )

    return train, val, test


# ============================================================
# CHARGEMENT CGT
# ============================================================

def load_cgt_candidates():

    print()
    print("=" * 70)
    print("CHARGEMENT DES CANDIDATS CGT")
    print("=" * 70)

    if not CGT_CANDIDATES_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : "
            f"{CGT_CANDIDATES_FILE}"
        )

    df = pd.read_csv(
        CGT_CANDIDATES_FILE,
        dtype=str
    )

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    df["n_assessed_types"] = pd.to_numeric(
        df["n_assessed_types"],
        errors="coerce"
    )

    df["n_assessments"] = pd.to_numeric(
        df["n_assessments"],
        errors="coerce"
    )

    print(
        f"Candidats CGT disponibles : "
        f"{len(df):,}"
    )

    return df


# ============================================================
# SELECTION CGT
# ============================================================

def select_cgt_candidates(df):

    print()
    print("=" * 70)
    print("SELECTION DES CONTRATS CGT")
    print("=" * 70)

    # --------------------------------------------------------
    # Positifs
    # --------------------------------------------------------

    positives = df[
        df["label"] == 1
    ].copy()

    # --------------------------------------------------------
    # Négatifs
    # --------------------------------------------------------

    negatives = df[
        (
            df["label"] == 0
        )
        &
        (
            df["n_assessed_types"]
            >= MIN_NEGATIVE_COVERAGE
        )
    ].copy()

    selected = pd.concat(
        [
            positives,
            negatives
        ],
        ignore_index=True
    )

    print(
        f"Positifs retenus : "
        f"{len(positives):,}"
    )

    print(
        f"Négatifs retenus : "
        f"{len(negatives):,}"
    )

    print(
        f"Total CGT retenu : "
        f"{len(selected):,}"
    )

    print()

    print(
        "Distribution :"
    )

    print(
        selected[
            "label"
        ].value_counts().sort_index()
    )

    return selected


# ============================================================
# CONVERSION CGT AU FORMAT V1
# ============================================================

def convert_cgt_to_v1_format(selected):

    print()
    print("=" * 70)
    print("CONVERSION DES CONTRATS CGT")
    print("=" * 70)

    samples = []

    total = len(selected)

    for index, row in selected.iterrows():

        source_path = Path(
            row["source_path"]
        )

        if not source_path.exists():

            print(
                f"ATTENTION : fichier absent : "
                f"{source_path}"
            )

            continue

        code = source_path.read_text(
            encoding="utf-8",
            errors="replace"
        )

        if not code.strip():
            continue

        # Le dataset V1 stocke context comme une liste.
        context = code.splitlines()

        label = int(
            row["label"]
        )

        positive_types = []

        value = row.get(
            "positive_types",
            ""
        )

        if pd.notna(value):

            positive_types = [
                item
                for item in str(value).split("|")
                if item
            ]

        sample = {

            "sample_id":
                f"cgt_{row['fp_sol']}",

            "source_dataset":
                "CGT",

            "source_split":
                "train",

            "source_path":
                str(source_path),

            "granularity":
                "contract",

            "contract_name":
                "",

            "function_name":
                "",

            "context":
                context,

            "has_vulnerability":
                label,

            "vulnerabilities":
                positive_types,

            "label_confidence":
                "verified_cgt",

            "label_origin":
                "manual_consolidated_ground_truth",

            "dedup_hash_raw":
                str(
                    row.get(
                        "fp_sol",
                        ""
                    )
                ),

            "dedup_hash_normalized":
                str(
                    row.get(
                        "normalized_hash",
                        ""
                    )
                ),

            "metadata": {

                "source":
                    "CGT",

                "fp_sol":
                    str(
                        row.get(
                            "fp_sol",
                            ""
                        )
                    ),

                "n_assessments":
                    int(
                        row[
                            "n_assessments"
                        ]
                    ),

                "n_assessed_types":
                    int(
                        row[
                            "n_assessed_types"
                        ]
                    ),

                "assessed_types":
                    str(
                        row.get(
                            "assessed_types",
                            ""
                        )
                    ),

                "positive_types":
                    str(
                        row.get(
                            "positive_types",
                            ""
                        )
                    ),

                "assessed_swcs":
                    str(
                        row.get(
                            "assessed_swcs",
                            ""
                        )
                    ),

                "source_datasets":
                    str(
                        row.get(
                            "source_datasets",
                            ""
                        )
                    ),
            },
        }

        samples.append(
            sample
        )

        if (
            len(samples) % 250 == 0
            or len(samples) == total
        ):

            print(
                f"  {len(samples):,}/"
                f"{total:,} contrats convertis"
            )

    print()

    print(
        f"Contrats CGT convertis : "
        f"{len(samples):,}"
    )

    return samples


# ============================================================
# DISTRIBUTION
# ============================================================

def count_labels(samples):

    labels = {
        0: 0,
        1: 0,
    }

    for sample in samples:

        label = int(
            sample[
                "has_vulnerability"
            ]
        )

        labels[label] = (
            labels.get(
                label,
                0
            )
            + 1
        )

    return labels


# ============================================================
# CREATION V2
# ============================================================

def build_v2(
    train_v1,
    val_v1,
    test_v1,
    cgt_samples
):

    print()
    print("=" * 70)
    print("CONSTRUCTION DU DATASET V2")
    print("=" * 70)

    # CGT est ajouté UNIQUEMENT au train.
    train_v2 = (
        train_v1
        + cgt_samples
    )

    # Validation et test restent EXACTEMENT identiques.
    val_v2 = val_v1

    test_v2 = test_v1

    print(
        f"Train V2 : "
        f"{len(train_v2):,}"
    )

    print(
        f"Validation V2 : "
        f"{len(val_v2):,}"
    )

    print(
        f"Test V2 : "
        f"{len(test_v2):,}"
    )

    print()

    print(
        "Distribution Train V2 :"
    )

    train_labels = count_labels(
        train_v2
    )

    print(
        f"0 non_vulnerable : "
        f"{train_labels[0]:,}"
    )

    print(
        f"1 vulnerable     : "
        f"{train_labels[1]:,}"
    )

    return (
        train_v2,
        val_v2,
        test_v2,
    )


# ============================================================
# VERIFICATION VAL / TEST
# ============================================================

def verify_unchanged_split(
    original,
    new,
    split_name
):

    if len(original) != len(new):

        raise ValueError(
            f"{split_name} a changé de taille."
        )

    original_ids = [
        sample.get(
            "sample_id"
        )
        for sample in original
    ]

    new_ids = [
        sample.get(
            "sample_id"
        )
        for sample in new
    ]

    if original_ids != new_ids:

        raise ValueError(
            f"{split_name} n'est plus "
            f"identique à V1."
        )

    print(
        f"{split_name} identique à V1 : OK"
    )


# ============================================================
# SAUVEGARDE
# ============================================================

def save_v2(
    train_v2,
    val_v2,
    test_v2,
    cgt_samples
):

    print()
    print("=" * 70)
    print("SAUVEGARDE DU DATASET V2")
    print("=" * 70)

    V2_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    save_json(
        train_v2,
        V2_DIR / "train.json"
    )

    save_json(
        val_v2,
        V2_DIR / "val.json"
    )

    save_json(
        test_v2,
        V2_DIR / "test.json"
    )

    summary = {

        "version":
            "v2_cgt",

        "negative_min_assessed_types":
            MIN_NEGATIVE_COVERAGE,

        "train_samples":
            len(train_v2),

        "validation_samples":
            len(val_v2),

        "test_samples":
            len(test_v2),

        "cgt_added":
            len(cgt_samples),

        "cgt_vulnerable_added":
            sum(
                1
                for sample in cgt_samples
                if sample[
                    "has_vulnerability"
                ] == 1
            ),

        "cgt_non_vulnerable_added":
            sum(
                1
                for sample in cgt_samples
                if sample[
                    "has_vulnerability"
                ] == 0
            ),

        "train_distribution":
            count_labels(
                train_v2
            ),

        "validation_distribution":
            count_labels(
                val_v2
            ),

        "test_distribution":
            count_labels(
                test_v2
            ),
    }

    save_json(
        summary,
        V2_DIR
        / "dataset_v2_summary.json"
    )

    print(
        f"Dataset V2 sauvegardé dans :"
    )

    print(
        V2_DIR
    )


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("=" * 70)
    print("CONSTRUCTION DATASET V2")
    print("V1 + CONSOLIDATED GROUND TRUTH")
    print("=" * 70)

    train_v1, val_v1, test_v1 = (
        load_v1()
    )

    cgt_df = (
        load_cgt_candidates()
    )

    selected = (
        select_cgt_candidates(
            cgt_df
        )
    )

    cgt_samples = (
        convert_cgt_to_v1_format(
            selected
        )
    )

    (
        train_v2,
        val_v2,
        test_v2,
    ) = build_v2(
        train_v1,
        val_v1,
        test_v1,
        cgt_samples,
    )

    print()
    print("=" * 70)
    print("VERIFICATION DES SPLITS")
    print("=" * 70)

    verify_unchanged_split(
        val_v1,
        val_v2,
        "Validation"
    )

    verify_unchanged_split(
        test_v1,
        test_v2,
        "Test"
    )

    save_v2(
        train_v2,
        val_v2,
        test_v2,
        cgt_samples,
    )

    print()
    print("=" * 70)
    print("DATASET V2 CONSTRUIT AVEC SUCCES")
    print("=" * 70)

    print()
    print(
        "IMPORTANT : aucun modèle "
        "n'a encore été réentraîné."
    )


if __name__ == "__main__":
    main()