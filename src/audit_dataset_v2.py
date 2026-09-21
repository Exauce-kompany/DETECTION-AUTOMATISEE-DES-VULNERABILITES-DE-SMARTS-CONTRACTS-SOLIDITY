from pathlib import Path
import hashlib
import json
import re


PROJECT_ROOT = Path(__file__).resolve().parent.parent
V2_DIR = PROJECT_ROOT / "dataset" / "v2" / "raw"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def remove_comments(code):
    pattern = re.compile(
        r"""
        ("(?:\\.|[^"\\])*") |
        ('(?:\\.|[^'\\])*') |
        (//[^\n]*$) |
        (/\*.*?\*/)
        """,
        re.MULTILINE | re.DOTALL | re.VERBOSE,
    )

    def replacer(match):
        if match.group(1) or match.group(2):
            return match.group(0)
        return " "

    return re.sub(pattern, replacer, code)


def normalize_code(code):
    if not isinstance(code, str):
        return ""

    code = code.replace("\r\n", "\n").replace("\r", "\n")
    code = remove_comments(code)
    code = re.sub(r"\s+", " ", code)

    return code.strip()


def code_from_sample(sample):
    context = sample.get("context", [])

    if isinstance(context, list):
        return "\n".join(str(x) for x in context)

    if isinstance(context, str):
        return context

    return ""


def normalized_hash(sample):
    code = normalize_code(
        code_from_sample(sample)
    )

    return hashlib.sha256(
        code.encode("utf-8")
    ).hexdigest()


def build_index(samples, split_name):
    index = {}

    for sample in samples:
        h = normalized_hash(sample)

        label = int(
            sample["has_vulnerability"]
        )

        if h not in index:
            index[h] = {
                "labels": set(),
                "count": 0,
                "sample_ids": [],
                "split": split_name,
            }

        index[h]["labels"].add(label)
        index[h]["count"] += 1
        index[h]["sample_ids"].append(
            sample.get("sample_id", "")
        )

    return index


def count_labels(samples):
    result = {0: 0, 1: 0}

    for sample in samples:
        result[
            int(sample["has_vulnerability"])
        ] += 1

    return result


def audit_internal(index, split_name):
    duplicates = 0
    conflicts = 0

    for _, info in index.items():
        if info["count"] > 1:
            duplicates += 1

        if len(info["labels"]) > 1:
            conflicts += 1

    print(
        f"{split_name:<12} "
        f"hashes uniques : {len(index):,}"
    )

    print(
        f"{split_name:<12} "
        f"doublons internes : {duplicates:,}"
    )

    print(
        f"{split_name:<12} "
        f"conflits labels : {conflicts:,}"
    )

    return duplicates, conflicts


def compare_splits(
    index_a,
    name_a,
    index_b,
    name_b
):
    common = (
        set(index_a.keys())
        &
        set(index_b.keys())
    )

    conflicts = 0

    for h in common:
        labels_a = index_a[h]["labels"]
        labels_b = index_b[h]["labels"]

        if labels_a != labels_b:
            conflicts += 1

    print(
        f"{name_a} ↔ {name_b} : "
        f"{len(common):,} hash(es) commun(s)"
    )

    print(
        f"{name_a} ↔ {name_b} : "
        f"{conflicts:,} conflit(s) de label"
    )

    return len(common), conflicts


def main():
    print("=" * 70)
    print("AUDIT FINAL DU DATASET V2")
    print("=" * 70)

    train = load_json(
        V2_DIR / "train.json"
    )

    val = load_json(
        V2_DIR / "val.json"
    )

    test = load_json(
        V2_DIR / "test.json"
    )

    print()
    print("=" * 70)
    print("TAILLES")
    print("=" * 70)

    print(f"Train      : {len(train):,}")
    print(f"Validation : {len(val):,}")
    print(f"Test       : {len(test):,}")

    print()
    print("=" * 70)
    print("DISTRIBUTIONS")
    print("=" * 70)

    print(
        "Train      :",
        count_labels(train)
    )

    print(
        "Validation :",
        count_labels(val)
    )

    print(
        "Test       :",
        count_labels(test)
    )

    print()
    print("=" * 70)
    print("INDEXATION PAR HASH NORMALISE")
    print("=" * 70)

    train_index = build_index(
        train,
        "train"
    )

    val_index = build_index(
        val,
        "val"
    )

    test_index = build_index(
        test,
        "test"
    )

    print()
    print("=" * 70)
    print("DOUBLONS INTERNES")
    print("=" * 70)

    train_dup, train_conf = audit_internal(
        train_index,
        "Train"
    )

    val_dup, val_conf = audit_internal(
        val_index,
        "Validation"
    )

    test_dup, test_conf = audit_internal(
        test_index,
        "Test"
    )

    print()
    print("=" * 70)
    print("CHEVAUCHEMENT ENTRE SPLITS")
    print("=" * 70)

    train_val, train_val_conf = compare_splits(
        train_index,
        "Train",
        val_index,
        "Validation"
    )

    train_test, train_test_conf = compare_splits(
        train_index,
        "Train",
        test_index,
        "Test"
    )

    val_test, val_test_conf = compare_splits(
        val_index,
        "Validation",
        test_index,
        "Test"
    )

    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)

    problems = (
        train_conf
        + val_conf
        + test_conf
        + train_val
        + train_test
        + val_test
    )

    if problems == 0:
        print(
            "DATASET V2 PROPRE : aucun conflit "
            "et aucun chevauchement entre splits."
        )
    else:
        print(
            "ATTENTION : le dataset V2 contient "
            "encore des doublons ou chevauchements."
        )

    print()
    print("=" * 70)
    print("AUDIT TERMINE")
    print("=" * 70)


if __name__ == "__main__":
    main()