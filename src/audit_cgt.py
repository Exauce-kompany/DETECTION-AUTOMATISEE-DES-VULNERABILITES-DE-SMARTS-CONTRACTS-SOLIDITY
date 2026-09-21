from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CGT_ROOT = (
    PROJECT_ROOT.parent
    / "smart-contract-cgt"
)

CSV_PATH = (
    CGT_ROOT
    / "consolidated.csv"
)

SOURCE_DIR = (
    CGT_ROOT
    / "source"
)


# ============================================================
# CHARGEMENT
# ============================================================

def load_cgt():

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {CSV_PATH}"
        )

    print("=" * 70)
    print("AUDIT DU DATASET CGT")
    print("=" * 70)

    print()
    print(f"CSV : {CSV_PATH}")
    print(f"Sources : {SOURCE_DIR}")

    df = pd.read_csv(
        CSV_PATH,
        sep=";",
        dtype=str,
        low_memory=False
    )

    return df


# ============================================================
# STATISTIQUES GENERALES
# ============================================================

def general_statistics(df):

    print()
    print("=" * 70)
    print("STATISTIQUES GENERALES")
    print("=" * 70)

    print(
        f"Nombre total d'évaluations : "
        f"{len(df):,}"
    )

    print(
        f"Nombre de datasets sources : "
        f"{df['dataset'].nunique():,}"
    )

    print(
        f"Nombre de propriétés : "
        f"{df['property'].nunique():,}"
    )

    fp_sol = (
        df["fp_sol"]
        .dropna()
        .astype(str)
    )

    fp_sol = fp_sol[
        fp_sol.str.strip() != ""
    ]

    print(
        f"fp_sol renseignés : "
        f"{len(fp_sol):,}"
    )

    print(
        f"Contrats Solidity uniques (fp_sol) : "
        f"{fp_sol.nunique():,}"
    )

    fp_sol2 = (
        df["fp_sol2"]
        .dropna()
        .astype(str)
    )

    fp_sol2 = fp_sol2[
        fp_sol2.str.strip() != ""
    ]

    print(
        f"Contrats uniques fp_sol2 : "
        f"{fp_sol2.nunique():,}"
    )


# ============================================================
# PROPERTY_HOLDS
# ============================================================

def property_holds_statistics(df):

    print()
    print("=" * 70)
    print("DISTRIBUTION PROPERTY_HOLDS")
    print("=" * 70)

    print(
        df["property_holds"]
        .value_counts(
            dropna=False
        )
    )


# ============================================================
# SOURCES
# ============================================================

def dataset_statistics(df):

    print()
    print("=" * 70)
    print("CONTRIBUTION PAR DATASET")
    print("=" * 70)

    stats = (
        df.groupby("dataset")
        .agg(
            evaluations=(
                "property_holds",
                "size"
            ),
            contracts=(
                "fp_sol",
                "nunique"
            )
        )
        .sort_values(
            "contracts",
            ascending=False
        )
    )

    print(stats.to_string())


# ============================================================
# SWC
# ============================================================

def swc_statistics(df):

    print()
    print("=" * 70)
    print("DISTRIBUTION DES SWC")
    print("=" * 70)

    swc_df = df[
        df["swc"].notna()
    ].copy()

    swc_df["swc"] = (
        swc_df["swc"]
        .astype(str)
        .str.strip()
    )

    swc_df = swc_df[
        swc_df["swc"] != ""
    ]

    stats = (
        swc_df.groupby(
            [
                "swc",
                "property_holds"
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    if "t" not in stats.columns:
        stats["t"] = 0

    if "f" not in stats.columns:
        stats["f"] = 0

    stats["total"] = (
        stats["t"]
        + stats["f"]
    )

    stats = stats.sort_values(
        "total",
        ascending=False
    )

    print(stats.to_string())


# ============================================================
# CATEGORIES QUI NOUS INTERESSENT
# ============================================================

def target_swc_statistics(df):

    print()
    print("=" * 70)
    print("SWC PERTINENTS POUR NOTRE DATASET")
    print("=" * 70)

    # Mappings principaux utilisés par notre taxonomy actuelle.
    target_mapping = {
        "101": "arithmetic",
        "104": "unchecked_low_calls",
        "107": "reentrancy",
        "113": "denial_service",
        "128": "denial_service",
        "114": "front_running",
        "116": "time_manipulation",
        "120": "bad_randomness",
    }

    work = df.copy()

    work["swc"] = (
        work["swc"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    target = work[
        work["swc"].isin(
            target_mapping.keys()
        )
    ].copy()

    target["normalized_type"] = (
        target["swc"]
        .map(target_mapping)
    )

    print(
        f"Evaluations pertinentes : "
        f"{len(target):,}"
    )

    print(
        f"Contrats concernés : "
        f"{target['fp_sol'].nunique():,}"
    )

    print()

    stats = (
        target.groupby(
            [
                "normalized_type",
                "property_holds"
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    if "t" not in stats.columns:
        stats["t"] = 0

    if "f" not in stats.columns:
        stats["f"] = 0

    stats["total"] = (
        stats["t"]
        + stats["f"]
    )

    print(stats.to_string())

    print()
    print(
        "REMARQUE : access_control sera traité "
        "séparément car cette catégorie couvre "
        "plusieurs propriétés/SWC."
    )


# ============================================================
# CONTRATS VULNERABLES / POTENTIELLEMENT NEGATIFS
# ============================================================

def contract_level_statistics(df):

    print()
    print("=" * 70)
    print("STATISTIQUES AU NIVEAU CONTRAT")
    print("=" * 70)

    work = df[
        df["fp_sol"].notna()
    ].copy()

    work["fp_sol"] = (
        work["fp_sol"]
        .astype(str)
        .str.strip()
    )

    work = work[
        work["fp_sol"] != ""
    ]

    grouped = (
        work.groupby("fp_sol")[
            "property_holds"
        ]
        .agg(list)
    )

    any_true = 0
    only_false = 0
    mixed = 0

    for values in grouped:

        values = set(values)

        has_true = "t" in values
        has_false = "f" in values

        if has_true and has_false:
            mixed += 1

        elif has_true:
            any_true += 1

        elif has_false:
            only_false += 1

    # mixed contient aussi des contrats ayant
    # au moins une propriété vraie.
    vulnerable_candidate = (
        any_true
        + mixed
    )

    print(
        f"Contrats avec au moins un 't' : "
        f"{vulnerable_candidate:,}"
    )

    print(
        f"Contrats avec uniquement des 'f' : "
        f"{only_false:,}"
    )

    print(
        f"Contrats ayant à la fois t et f : "
        f"{mixed:,}"
    )


# ============================================================
# VERIFICATION DES FICHIERS SOURCE
# ============================================================

def verify_source_files(df):

    print()
    print("=" * 70)
    print("VERIFICATION DES FICHIERS SOLIDITY")
    print("=" * 70)

    fingerprints = (
        df["fp_sol"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    fingerprints = [
        fp
        for fp in fingerprints
        if fp
    ]

    existing = 0
    missing = []

    for fp in fingerprints:

        path = (
            SOURCE_DIR
            / f"{fp}.sol"
        )

        if path.exists():
            existing += 1
        else:
            missing.append(fp)

    print(
        f"fp_sol uniques : "
        f"{len(fingerprints):,}"
    )

    print(
        f"Sources trouvées : "
        f"{existing:,}"
    )

    print(
        f"Sources manquantes : "
        f"{len(missing):,}"
    )

    if missing:

        print()
        print(
            "Exemples de fichiers manquants :"
        )

        for fp in missing[:10]:

            print(
                f"  {fp}.sol"
            )


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    df = load_cgt()

    general_statistics(df)

    property_holds_statistics(df)

    dataset_statistics(df)

    swc_statistics(df)

    target_swc_statistics(df)

    contract_level_statistics(df)

    verify_source_files(df)

    print()
    print("=" * 70)
    print("AUDIT CGT TERMINE")
    print("=" * 70)


if __name__ == "__main__":
    main()