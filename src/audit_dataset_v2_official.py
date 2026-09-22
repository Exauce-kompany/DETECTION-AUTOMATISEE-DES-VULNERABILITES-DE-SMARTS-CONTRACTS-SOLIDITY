from pathlib import Path
import hashlib
import json
from collections import defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

V2_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "v2"
    / "raw"
)


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


# ============================================================
# CONTEXTE
# ============================================================

def context_to_text(sample):

    context = sample.get(
        "context",
        []
    )

    if isinstance(
        context,
        list
    ):

        return "\n".join(
            str(line)
            for line in context
        )

    if isinstance(
        context,
        str
    ):

        return context

    return ""


def context_sha256(sample):

    code = context_to_text(
        sample
    )

    return hashlib.sha256(
        code.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# NETTOYAGE HASH
# ============================================================

def clean_hash(value):

    if value is None:
        return ""

    value = str(value).strip()

    if value.lower() in {
        "",
        "none",
        "nan",
        "null"
    }:
        return ""

    return value


# ============================================================
# CLES OFFICIELLES
# ============================================================

def get_hashes(sample):

    normalized = clean_hash(
        sample.get(
            "dedup_hash_normalized",
            ""
        )
    )

    raw = clean_hash(
        sample.get(
            "dedup_hash_raw",
            ""
        )
    )

    context_hash = context_sha256(
        sample
    )

    return {
        "normalized":
            normalized,

        "raw":
            raw,

        "context":
            context_hash,
    }


# ============================================================
# STATISTIQUES DES CHAMPS
# ============================================================

def hash_availability(
    samples,
    split_name
):

    normalized = 0
    raw = 0

    for sample in samples:

        hashes = get_hashes(
            sample
        )

        if hashes[
            "normalized"
        ]:
            normalized += 1

        if hashes[
            "raw"
        ]:
            raw += 1

    print(
        f"{split_name:<12} "
        f"dedup_hash_normalized : "
        f"{normalized:,}/{len(samples):,}"
    )

    print(
        f"{split_name:<12} "
        f"dedup_hash_raw        : "
        f"{raw:,}/{len(samples):,}"
    )


# ============================================================
# INDEX
# ============================================================

def build_indexes(samples):

    normalized_index = defaultdict(
        list
    )

    raw_index = defaultdict(
        list
    )

    context_index = defaultdict(
        list
    )

    for position, sample in enumerate(
        samples
    ):

        hashes = get_hashes(
            sample
        )

        record = {
            "position":
                position,

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

            "source_dataset":
                sample.get(
                    "source_dataset",
                    ""
                ),
        }

        if hashes[
            "normalized"
        ]:

            normalized_index[
                hashes[
                    "normalized"
                ]
            ].append(
                record
            )

        if hashes[
            "raw"
        ]:

            raw_index[
                hashes[
                    "raw"
                ]
            ].append(
                record
            )

        context_index[
            hashes[
                "context"
            ]
        ].append(
            record
        )

    return {
        "normalized":
            normalized_index,

        "raw":
            raw_index,

        "context":
            context_index,
    }


# ============================================================
# AUDIT INTERNE
# ============================================================

def audit_internal(
    indexes,
    split_name
):

    print()
    print(
        f"--- {split_name} ---"
    )

    summary = {}
    for hash_type in [
        "normalized",
        "raw",
        "context"
    ]:

        index = indexes[
            hash_type
        ]

        duplicate_hashes = 0
        conflicting_hashes = 0
        duplicated_samples = 0

        for _, records in index.items():

            if len(records) > 1:

                duplicate_hashes += 1

                duplicated_samples += (
                    len(records) - 1
                )

                labels = {
                    record["label"]
                    for record in records
                }

                if len(labels) > 1:
                    conflicting_hashes += 1

        summary[hash_type] = {"unique_hashes": len(index), "duplicate_groups": duplicate_hashes, "duplicated_samples": duplicated_samples, "conflicting_groups": conflicting_hashes}
        print(
            f"{hash_type:<12} : "
            f"{len(index):,} hashes uniques | "
            f"{duplicate_hashes:,} groupes dupliqués | "
            f"{duplicated_samples:,} échantillons supplémentaires | "
            f"{conflicting_hashes:,} conflits labels"
        )


    return summary


# ============================================================
# COMPARAISON DE DEUX SPLITS
# ============================================================

def compare_one_hash_type(
    index_a,
    index_b
):

    common = (
        set(index_a.keys())
        &
        set(index_b.keys())
    )

    label_conflicts = 0

    for hash_value in common:

        labels_a = {
            record["label"]
            for record
            in index_a[
                hash_value
            ]
        }

        labels_b = {
            record["label"]
            for record
            in index_b[
                hash_value
            ]
        }

        if len(labels_a | labels_b) > 1:
            label_conflicts += 1

    return (
        len(common),
        label_conflicts,
    )


def compare_splits(
    indexes_a,
    name_a,
    indexes_b,
    name_b
):

    print()
    print(
        f"{name_a} ↔ {name_b}"
    )

    results = {}

    for hash_type in [
        "normalized",
        "raw",
        "context"
    ]:

        common, conflicts = (
            compare_one_hash_type(
                indexes_a[
                    hash_type
                ],
                indexes_b[
                    hash_type
                ],
            )
        )

        results[
            hash_type
        ] = {
            "common":
                common,

            "conflicts":
                conflicts,
        }

        print(
            f"  {hash_type:<12} : "
            f"{common:,} commun(s) | "
            f"{conflicts:,} conflit(s)"
        )

    return results


# ============================================================
# IDENTIFICATION DES CHEVAUCHEMENTS
# ============================================================

def collect_overlap_examples(
    train_indexes,
    val_indexes,
    test_indexes
):

    output = []

    comparisons = [
        (
            "train",
            train_indexes,
            "validation",
            val_indexes,
        ),
        (
            "train",
            train_indexes,
            "test",
            test_indexes,
        ),
        (
            "validation",
            val_indexes,
            "test",
            test_indexes,
        ),
    ]

    for (
        name_a,
        indexes_a,
        name_b,
        indexes_b,
    ) in comparisons:

        for hash_type in [
            "normalized",
            "raw",
            "context"
        ]:

            common = (
                set(
                    indexes_a[
                        hash_type
                    ].keys()
                )
                &
                set(
                    indexes_b[
                        hash_type
                    ].keys()
                )
            )

            for hash_value in common:

                records_a = (
                    indexes_a[
                        hash_type
                    ][
                        hash_value
                    ]
                )

                records_b = (
                    indexes_b[
                        hash_type
                    ][
                        hash_value
                    ]
                )

                output.append(
                    {
                        "split_a":
                            name_a,

                        "split_b":
                            name_b,

                        "hash_type":
                            hash_type,

                        "hash":
                            hash_value,

                        "samples_a":
                            records_a,

                        "samples_b":
                            records_b,
                    }
                )

    return output


# ============================================================
# SOURCE CGT DANS TRAIN
# ============================================================

def cgt_statistics(train):

    cgt_samples = [
        sample
        for sample in train
        if str(
            sample.get(
                "source_dataset",
                ""
            )
        ).upper() == "CGT"
    ]

    vulnerable = sum(
        1
        for sample in cgt_samples
        if int(
            sample[
                "has_vulnerability"
            ]
        ) == 1
    )

    non_vulnerable = (
        len(cgt_samples)
        - vulnerable
    )

    print()
    print("=" * 70)
    print("CGT PRESENT DANS LE TRAIN V2")
    print("=" * 70)

    print(
        f"CGT total : "
        f"{len(cgt_samples):,}"
    )

    print(
        f"CGT non_vulnerable : "
        f"{non_vulnerable:,}"
    )

    print(
        f"CGT vulnerable : "
        f"{vulnerable:,}"
    )


# ============================================================
# SAUVEGARDE DU RAPPORT
# ============================================================

def save_report(
    report
):

    output_dir = (
        PROJECT_ROOT
        / "results"
        / "v2_official_audit"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir
        / "overlap_report.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(
        f"Rapport sauvegardé : "
        f"{output_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AUDIT V2 AVEC LES HASHES OFFICIELS")
    print("=" * 70)

    train = load_json(
        V2_DIR
        / "train.json"
    )

    val = load_json(
        V2_DIR
        / "val.json"
    )

    test = load_json(
        V2_DIR
        / "test.json"
    )

    print()
    print("=" * 70)
    print("TAILLES")
    print("=" * 70)

    print(
        f"Train      : {len(train):,}"
    )

    print(
        f"Validation : {len(val):,}"
    )

    print(
        f"Test       : {len(test):,}"
    )

    print()
    print("=" * 70)
    print("DISPONIBILITE DES HASHES")
    print("=" * 70)

    hash_availability(
        train,
        "Train"
    )

    hash_availability(
        val,
        "Validation"
    )

    hash_availability(
        test,
        "Test"
    )

    print()
    print("=" * 70)
    print("CONSTRUCTION DES INDEX")
    print("=" * 70)

    train_indexes = build_indexes(
        train
    )

    val_indexes = build_indexes(
        val
    )

    test_indexes = build_indexes(
        test
    )

    print(
        "Index construits."
    )

    print()
    print("=" * 70)
    print("DOUBLONS INTERNES")
    print("=" * 70)

    internal_train = audit_internal(
        train_indexes,
        "TRAIN"
    )

    internal_validation = audit_internal(
        val_indexes,
        "VALIDATION"
    )

    internal_test = audit_internal(
        test_indexes,
        "TEST"
    )

    print()
    print("=" * 70)
    print("CHEVAUCHEMENTS ENTRE SPLITS")
    print("=" * 70)

    train_val = compare_splits(
        train_indexes,
        "TRAIN",
        val_indexes,
        "VALIDATION"
    )

    train_test = compare_splits(
        train_indexes,
        "TRAIN",
        test_indexes,
        "TEST"
    )

    val_test = compare_splits(
        val_indexes,
        "VALIDATION",
        test_indexes,
        "TEST"
    )

    cgt_statistics(
        train
    )

    overlaps = collect_overlap_examples(
        train_indexes,
        val_indexes,
        test_indexes,
    )

    report = {
        "internal": {"train": internal_train, "validation": internal_validation, "test": internal_test},
        "train_samples":
            len(train),

        "validation_samples":
            len(val),

        "test_samples":
            len(test),

        "train_validation":
            train_val,

        "train_test":
            train_test,

        "validation_test":
            val_test,

        "overlap_details":
            overlaps,
    }

    save_report(
        report
    )

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    official_leakage = (
        train_val[
            "normalized"
        ][
            "common"
        ]
        +
        train_test[
            "normalized"
        ][
            "common"
        ]
        +
        val_test[
            "normalized"
        ][
            "common"
        ]
    )

    if official_leakage == 0:

        print(
            "Aucun chevauchement détecté "
            "avec dedup_hash_normalized."
        )

        print(
            "Ce contrôle des empreintes fournies ne prouve pas "
            "l'indépendance des entrées encodées ni l'absence de conflits internes. "
            "Le protocole V3 contrôle aussi ces propriétés."
        )

    else:

        print(
            "Des chevauchements existent aussi "
            "avec dedup_hash_normalized."
        )

        print(
            "Ils devront être analysés avant "
            "la préparation de V2."
        )

    print()
    print("=" * 70)
    print("AUDIT TERMINE")
    print("=" * 70)


if __name__ == "__main__":
    main()
