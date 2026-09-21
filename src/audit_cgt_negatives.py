from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "cgt_overlap_audit"
    / "cgt_unique_final.csv"
)


def main():

    print("=" * 70)
    print("AUDIT DES CANDIDATS NEGATIFS CGT")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        dtype=str
    )

    df["label"] = (
        pd.to_numeric(
            df["label"],
            errors="coerce"
        )
    )

    df["n_assessed_types"] = (
        pd.to_numeric(
            df["n_assessed_types"],
            errors="coerce"
        )
    )

    df["n_assessments"] = (
        pd.to_numeric(
            df["n_assessments"],
            errors="coerce"
        )
    )

    negatives = df[
        df["label"] == 0
    ].copy()

    print()
    print(
        f"Nombre total de candidats négatifs : "
        f"{len(negatives):,}"
    )

    print()
    print("=" * 70)
    print("COUVERTURE PAR NOMBRE DE TYPES")
    print("=" * 70)

    coverage = (
        negatives[
            "n_assessed_types"
        ]
        .value_counts()
        .sort_index()
    )

    print(coverage)

    print()
    print("=" * 70)
    print("SEUILS DE QUALITE POSSIBLES")
    print("=" * 70)

    for threshold in range(1, 7):

        subset = negatives[
            negatives[
                "n_assessed_types"
            ] >= threshold
        ]

        print(
            f"Au moins {threshold} type(s) évalué(s) : "
            f"{len(subset):,}"
        )

    print()
    print("=" * 70)
    print("NOMBRE D'EVALUATIONS PAR CONTRAT")
    print("=" * 70)

    print(
        negatives[
            "n_assessments"
        ].describe()
    )

    print()
    print("=" * 70)
    print("DATASETS SOURCES")
    print("=" * 70)

    source_counts = {}

    for value in negatives[
        "source_datasets"
    ].fillna(""):

        for source in str(value).split("|"):

            source = source.strip()

            if not source:
                continue

            source_counts[source] = (
                source_counts.get(
                    source,
                    0
                )
                + 1
            )

    for source, count in sorted(
        source_counts.items(),
        key=lambda item: item[1],
        reverse=True
    ):

        print(
            f"{source:<25} {count:>6}"
        )

    print()
    print("=" * 70)
    print("COUVERTURE DES TYPES")
    print("=" * 70)

    type_counts = {}

    for value in negatives[
        "assessed_types"
    ].fillna(""):

        for vulnerability_type in str(
            value
        ).split("|"):

            vulnerability_type = (
                vulnerability_type.strip()
            )

            if not vulnerability_type:
                continue

            type_counts[
                vulnerability_type
            ] = (
                type_counts.get(
                    vulnerability_type,
                    0
                )
                + 1
            )

    for vulnerability_type, count in sorted(
        type_counts.items(),
        key=lambda item: item[1],
        reverse=True
    ):

        print(
            f"{vulnerability_type:<25} "
            f"{count:>6}"
        )

    print()
    print("=" * 70)
    print("AUDIT TERMINE")
    print("=" * 70)


if __name__ == "__main__":
    main()
