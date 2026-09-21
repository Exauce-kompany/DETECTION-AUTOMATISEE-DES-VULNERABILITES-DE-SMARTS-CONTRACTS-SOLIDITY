from pathlib import Path
import hashlib
import json
import re

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CGT_ROOT = (
    PROJECT_ROOT.parent
    / "smart-contract-cgt"
)

CGT_CSV = (
    CGT_ROOT
    / "consolidated.csv"
)

CGT_SOURCE_DIR = (
    CGT_ROOT
    / "source"
)

V1_ROOT = (
    PROJECT_ROOT.parent
    / "smart-contract-vuln-dataset"
    / "data"
    / "processed"
    / "balanced_stage1_resplit_721"
    / "has_vul_721_stratified_v1"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "cgt_overlap_audit"
)


# ============================================================
# MAPPING CONSERVATEUR
# ============================================================

TARGET_SWC = {
    "101": "arithmetic",
    "104": "unchecked_low_calls",
    "107": "reentrancy",
    "113": "denial_service",
    "128": "denial_service",
    "114": "front_running",
    "116": "time_manipulation",
    "120": "bad_randomness",
}


# ============================================================
# SUPPRESSION DES COMMENTAIRES
# ============================================================

def remove_comments(code):
    """
    Supprime les commentaires // et /* ... */
    sans supprimer le contenu des chaînes.
    """

    result = []

    i = 0
    length = len(code)

    state = "normal"

    while i < length:

        char = code[i]

        next_char = (
            code[i + 1]
            if i + 1 < length
            else ""
        )

        # ----------------------------------------------------
        # Etat normal
        # ----------------------------------------------------

        if state == "normal":

            if (
                char == "/"
                and next_char == "/"
            ):
                state = "line_comment"
                i += 2
                continue

            if (
                char == "/"
                and next_char == "*"
            ):
                state = "block_comment"
                i += 2
                continue

            if char == '"':
                state = "double_string"
                result.append(char)
                i += 1
                continue

            if char == "'":
                state = "single_string"
                result.append(char)
                i += 1
                continue

            result.append(char)
            i += 1

        # ----------------------------------------------------
        # Commentaire ligne
        # ----------------------------------------------------

        elif state == "line_comment":

            if char == "\n":
                result.append("\n")
                state = "normal"

            i += 1

        # ----------------------------------------------------
        # Commentaire bloc
        # ----------------------------------------------------

        elif state == "block_comment":

            if (
                char == "*"
                and next_char == "/"
            ):
                state = "normal"
                i += 2
                continue

            # Conserver les retours à la ligne.
            if char == "\n":
                result.append("\n")

            i += 1

        # ----------------------------------------------------
        # Chaîne double
        # ----------------------------------------------------

        elif state == "double_string":

            result.append(char)

            if char == "\\":
                if i + 1 < length:
                    result.append(
                        code[i + 1]
                    )
                    i += 2
                    continue

            if char == '"':
                state = "normal"

            i += 1

        # ----------------------------------------------------
        # Chaîne simple
        # ----------------------------------------------------

        elif state == "single_string":

            result.append(char)

            if char == "\\":
                if i + 1 < length:
                    result.append(
                        code[i + 1]
                    )
                    i += 2
                    continue

            if char == "'":
                state = "normal"

            i += 1

    return "".join(result)


# ============================================================
# NORMALISATION DU CODE
# ============================================================

def normalize_code(code):

    if not isinstance(code, str):
        return ""

    code = code.replace(
        "\r\n",
        "\n"
    ).replace(
        "\r",
        "\n"
    )

    code = remove_comments(code)

    # Retirer les espaces en fin de ligne.
    lines = [
        line.rstrip()
        for line in code.split("\n")
    ]

    code = "\n".join(lines)

    # Compresser tous les espaces.
    code = re.sub(
        r"\s+",
        " ",
        code
    )

    return code.strip()


def normalized_hash(code):

    normalized = normalize_code(code)

    return hashlib.sha256(
        normalized.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# CHARGEMENT V1
# ============================================================

def load_v1():

    print()
    print("=" * 70)
    print("CHARGEMENT DU DATASET V1")
    print("=" * 70)

    records = []

    for split_name, filename in [
        ("train", "train.json"),
        ("val", "val.json"),
        ("test", "test.json"),
    ]:

        path = V1_ROOT / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Fichier V1 introuvable : {path}"
            )

        print(
            f"Chargement {split_name} : "
            f"{path}"
        )

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            samples = json.load(file)

        print(
            f"  {len(samples):,} échantillons"
        )

        for sample in samples:

            context = sample.get(
                "context",
                []
            )

            if isinstance(
                context,
                list
            ):
                code = "\n".join(
                    str(line)
                    for line in context
                )

            elif isinstance(
                context,
                str
            ):
                code = context

            else:
                code = ""

            code_hash = normalized_hash(
                code
            )

            records.append(
                {
                    "split":
                        split_name,

                    "sample_id":
                        sample.get(
                            "sample_id",
                            ""
                        ),

                    "label":
                        int(
                            sample[
                                "has_vulnerability"
                            ]
                        ),

                    "normalized_hash":
                        code_hash,
                }
            )

    dataframe = pd.DataFrame(
        records
    )

    print()
    print(
        f"Échantillons V1 : "
        f"{len(dataframe):,}"
    )

    print(
        f"Hashes normalisés uniques : "
        f"{dataframe['normalized_hash'].nunique():,}"
    )

    return dataframe


# ============================================================
# CHARGEMENT CGT
# ============================================================

def load_cgt():

    print()
    print("=" * 70)
    print("CHARGEMENT DE CGT")
    print("=" * 70)

    if not CGT_CSV.exists():
        raise FileNotFoundError(
            f"CGT introuvable : {CGT_CSV}"
        )

    df = pd.read_csv(
        CGT_CSV,
        sep=";",
        dtype=str,
        low_memory=False
    )

    df["swc"] = (
        df["swc"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["property_holds"] = (
        df["property_holds"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["fp_sol"] = (
        df["fp_sol"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Seulement les SWC directement compatibles.
    df = df[
        df["swc"].isin(
            TARGET_SWC.keys()
        )
    ].copy()

    # Il faut obligatoirement une source Solidity.
    df = df[
        df["fp_sol"] != ""
    ].copy()

    df["normalized_type"] = (
        df["swc"]
        .map(TARGET_SWC)
    )

    print(
        f"Évaluations CGT pertinentes : "
        f"{len(df):,}"
    )

    print(
        f"Contrats fp_sol concernés : "
        f"{df['fp_sol'].nunique():,}"
    )

    return df


# ============================================================
# CONSTRUCTION DES CANDIDATS CGT
# ============================================================

def build_cgt_candidates(df):

    print()
    print("=" * 70)
    print("CONSTRUCTION DES CANDIDATS CGT")
    print("=" * 70)

    records = []

    groups = df.groupby(
        "fp_sol"
    )

    total_groups = len(groups)

    for index, (
        fp_sol,
        group
    ) in enumerate(groups, start=1):

        source_path = (
            CGT_SOURCE_DIR
            / f"{fp_sol}.sol"
        )

        if not source_path.exists():
            continue

        code = source_path.read_text(
            encoding="utf-8",
            errors="replace"
        )

        if not code.strip():
            continue

        code_hash = normalized_hash(
            code
        )

        positive_rows = group[
            group["property_holds"] == "t"
        ]

        negative_rows = group[
            group["property_holds"] == "f"
        ]

        # ----------------------------------------------------
        # Label binaire
        #
        # 1 : au moins une faiblesse cible confirmée
        # 0 : aucune faiblesse cible positive parmi les
        #     évaluations disponibles.
        # ----------------------------------------------------

        has_vulnerability = (
            1
            if len(positive_rows) > 0
            else 0
        )

        positive_types = sorted(
            positive_rows[
                "normalized_type"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        assessed_types = sorted(
            group[
                "normalized_type"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        assessed_swcs = sorted(
            group["swc"]
            .dropna()
            .unique()
            .tolist()
        )

        source_datasets = sorted(
            group["dataset"]
            .dropna()
            .unique()
            .tolist()
        )

        records.append(
            {
                "fp_sol":
                    fp_sol,

                "normalized_hash":
                    code_hash,

                "label":
                    has_vulnerability,

                "n_assessments":
                    int(len(group)),

                "n_positive":
                    int(
                        len(positive_rows)
                    ),

                "n_negative":
                    int(
                        len(negative_rows)
                    ),

                "n_assessed_types":
                    int(
                        len(assessed_types)
                    ),

                "assessed_types":
                    "|".join(
                        assessed_types
                    ),

                "positive_types":
                    "|".join(
                        positive_types
                    ),

                "assessed_swcs":
                    "|".join(
                        assessed_swcs
                    ),

                "source_datasets":
                    "|".join(
                        source_datasets
                    ),

                "source_path":
                    str(source_path),
            }
        )

        if (
            index % 500 == 0
            or index == total_groups
        ):

            print(
                f"  {index:,}/"
                f"{total_groups:,} contrats analysés"
            )

    candidates = pd.DataFrame(
        records
    )

    print()
    print(
        f"Candidats construits : "
        f"{len(candidates):,}"
    )

    print()

    print(
        "Distribution initiale :"
    )

    print(
        candidates[
            "label"
        ].value_counts().sort_index()
    )

    return candidates


# ============================================================
# AUDIT DES DOUBLONS INTERNES CGT
# ============================================================

def audit_internal_duplicates(
    candidates
):

    print()
    print("=" * 70)
    print("DOUBLONS INTERNES CGT")
    print("=" * 70)

    hash_groups = (
        candidates.groupby(
            "normalized_hash"
        )
    )

    duplicate_groups = 0
    conflicting_groups = 0

    conflict_hashes = set()

    for code_hash, group in hash_groups:

        if len(group) > 1:

            duplicate_groups += 1

            labels = set(
                group["label"].tolist()
            )

            if len(labels) > 1:

                conflicting_groups += 1
                conflict_hashes.add(
                    code_hash
                )

    print(
        f"Hashes contenant plusieurs fp_sol : "
        f"{duplicate_groups:,}"
    )

    print(
        f"Hashes avec labels CGT contradictoires : "
        f"{conflicting_groups:,}"
    )

    return conflict_hashes


# ============================================================
# AUDIT DU CHEVAUCHEMENT V1
# ============================================================

def audit_overlap(
    v1,
    candidates,
    internal_conflict_hashes,
):

    print()
    print("=" * 70)
    print("CHEVAUCHEMENT CGT / V1")
    print("=" * 70)

    # --------------------------------------------------------
    # Hash V1 -> labels / splits
    # --------------------------------------------------------

    v1_hash_info = {}

    for code_hash, group in v1.groupby(
        "normalized_hash"
    ):

        v1_hash_info[
            code_hash
        ] = {
            "labels":
                sorted(
                    set(
                        int(x)
                        for x
                        in group["label"]
                    )
                ),

            "splits":
                sorted(
                    set(
                        group["split"]
                    )
                ),

            "sample_ids":
                group[
                    "sample_id"
                ].tolist(),
        }

    overlap_records = []
    unique_records = []
    conflict_records = []

    for _, row in candidates.iterrows():

        code_hash = row[
            "normalized_hash"
        ]

        cgt_label = int(
            row["label"]
        )

        record = row.to_dict()

        # ----------------------------------------------------
        # Conflit interne CGT
        # ----------------------------------------------------

        if (
            code_hash
            in internal_conflict_hashes
        ):

            record[
                "conflict_reason"
            ] = (
                "same_normalized_code_"
                "different_cgt_labels"
            )

            conflict_records.append(
                record
            )

            continue

        # ----------------------------------------------------
        # Présent dans V1
        # ----------------------------------------------------

        if code_hash in v1_hash_info:

            info = v1_hash_info[
                code_hash
            ]

            record[
                "v1_labels"
            ] = "|".join(
                str(x)
                for x in info["labels"]
            )

            record[
                "v1_splits"
            ] = "|".join(
                info["splits"]
            )

            record[
                "same_label"
            ] = (
                cgt_label
                in info["labels"]
            )

            overlap_records.append(
                record
            )

            if not record[
                "same_label"
            ]:

                record[
                    "conflict_reason"
                ] = (
                    "cgt_v1_label_conflict"
                )

                conflict_records.append(
                    record.copy()
                )

        # ----------------------------------------------------
        # Nouveau
        # ----------------------------------------------------

        else:

            unique_records.append(
                record
            )

    overlaps = pd.DataFrame(
        overlap_records
    )

    unique = pd.DataFrame(
        unique_records
    )

    conflicts = pd.DataFrame(
        conflict_records
    )

    print(
        f"Candidats CGT déjà présents dans V1 : "
        f"{len(overlaps):,}"
    )

    if len(overlaps) > 0:

        same = int(
            overlaps[
                "same_label"
            ].sum()
        )

        different = (
            len(overlaps)
            - same
        )

        print(
            f"  Même label : {same:,}"
        )

        print(
            f"  Label différent : "
            f"{different:,}"
        )

    print(
        f"Candidats CGT réellement nouveaux : "
        f"{len(unique):,}"
    )

    print(
        f"Conflits totaux détectés : "
        f"{len(conflicts):,}"
    )

    return (
        overlaps,
        unique,
        conflicts,
    )


# ============================================================
# DEDUPLICATION DES NOUVEAUX CGT
# ============================================================

def deduplicate_unique_candidates(
    unique
):

    print()
    print("=" * 70)
    print("DEDUPLICATION DES NOUVEAUX CANDIDATS")
    print("=" * 70)

    if unique.empty:

        return unique.copy()

    # --------------------------------------------------------
    # Si plusieurs fp_sol donnent exactement le même code
    # normalisé et le même label, garder un représentant.
    # --------------------------------------------------------

    deduped = (
        unique.sort_values(
            [
                "normalized_hash",
                "n_assessments"
            ],
            ascending=[
                True,
                False
            ]
        )
        .drop_duplicates(
            subset=[
                "normalized_hash"
            ],
            keep="first"
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Avant déduplication : "
        f"{len(unique):,}"
    )

    print(
        f"Après déduplication : "
        f"{len(deduped):,}"
    )

    print()

    print(
        "Distribution finale des labels :"
    )

    print(
        deduped[
            "label"
        ].value_counts().sort_index()
    )

    return deduped


# ============================================================
# STATISTIQUES DES NOUVEAUX CANDIDATS
# ============================================================

def print_unique_statistics(
    unique
):

    print()
    print("=" * 70)
    print("STATISTIQUES DES NOUVEAUX CGT")
    print("=" * 70)

    if unique.empty:

        print(
            "Aucun candidat nouveau."
        )

        return

    total = len(unique)

    vulnerable = int(
        (unique["label"] == 1).sum()
    )

    negative = int(
        (unique["label"] == 0).sum()
    )

    print(
        f"Total : {total:,}"
    )

    print(
        f"Vulnérables : "
        f"{vulnerable:,}"
    )

    print(
        f"Candidats négatifs : "
        f"{negative:,}"
    )

    print()

    print(
        "Couverture par nombre de types évalués :"
    )

    print(
        unique[
            "n_assessed_types"
        ]
        .value_counts()
        .sort_index()
    )

    print()

    # --------------------------------------------------------
    # Vulnérabilités positives
    # --------------------------------------------------------

    vulnerable_rows = unique[
        unique["label"] == 1
    ]

    type_counts = {}

    for value in vulnerable_rows[
        "positive_types"
    ]:

        if not value:
            continue

        for vulnerability_type in (
            str(value).split("|")
        ):

            type_counts[
                vulnerability_type
            ] = (
                type_counts.get(
                    vulnerability_type,
                    0
                )
                + 1
            )

    if type_counts:

        print(
            "Types de vulnérabilités "
            "dans les nouveaux positifs :"
        )

        for (
            vulnerability_type,
            count
        ) in sorted(
            type_counts.items(),
            key=lambda item:
                item[1],
            reverse=True
        ):

            print(
                f"  "
                f"{vulnerability_type:<22}"
                f"{count:>6,}"
            )


# ============================================================
# SAUVEGARDE
# ============================================================

def save_results(
    candidates,
    overlaps,
    unique,
    deduped,
    conflicts,
    v1
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    candidates.to_csv(
        OUTPUT_DIR
        / "cgt_candidates_all.csv",
        index=False,
        encoding="utf-8"
    )

    overlaps.to_csv(
        OUTPUT_DIR
        / "cgt_overlaps_v1.csv",
        index=False,
        encoding="utf-8"
    )

    unique.to_csv(
        OUTPUT_DIR
        / "cgt_unique_before_dedup.csv",
        index=False,
        encoding="utf-8"
    )

    deduped.to_csv(
        OUTPUT_DIR
        / "cgt_unique_final.csv",
        index=False,
        encoding="utf-8"
    )

    conflicts.to_csv(
        OUTPUT_DIR
        / "cgt_conflicts.csv",
        index=False,
        encoding="utf-8"
    )

    summary = {

        "v1_samples":
            int(len(v1)),

        "v1_unique_normalized_hashes":
            int(
                v1[
                    "normalized_hash"
                ].nunique()
            ),

        "cgt_candidates":
            int(
                len(candidates)
            ),

        "cgt_overlaps_v1":
            int(
                len(overlaps)
            ),

        "cgt_unique_before_dedup":
            int(
                len(unique)
            ),

        "cgt_unique_final":
            int(
                len(deduped)
            ),

        "cgt_conflicts":
            int(
                len(conflicts)
            ),

        "cgt_unique_vulnerable":
            int(
                (
                    deduped["label"] == 1
                ).sum()
            )
            if not deduped.empty
            else 0,

        "cgt_unique_negative_candidate":
            int(
                (
                    deduped["label"] == 0
                ).sum()
            )
            if not deduped.empty
            else 0,
    }

    with open(
        OUTPUT_DIR
        / "summary.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False
        )

    print()
    print("=" * 70)
    print("FICHIERS SAUVEGARDES")
    print("=" * 70)

    print(
        OUTPUT_DIR
    )

    print()

    for path in sorted(
        OUTPUT_DIR.glob("*")
    ):

        print(
            f"  {path.name}"
        )


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("=" * 70)
    print("AUDIT DU CHEVAUCHEMENT CGT / DATASET V1")
    print("=" * 70)

    v1 = load_v1()

    cgt = load_cgt()

    candidates = build_cgt_candidates(
        cgt
    )

    internal_conflicts = (
        audit_internal_duplicates(
            candidates
        )
    )

    (
        overlaps,
        unique,
        conflicts,
    ) = audit_overlap(
        v1,
        candidates,
        internal_conflicts,
    )

    deduped = (
        deduplicate_unique_candidates(
            unique
        )
    )

    print_unique_statistics(
        deduped
    )

    save_results(
        candidates,
        overlaps,
        unique,
        deduped,
        conflicts,
        v1,
    )

    print()
    print("=" * 70)
    print("AUDIT DU CHEVAUCHEMENT TERMINE")
    print("=" * 70)


if __name__ == "__main__":
    main()