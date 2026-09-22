"""Independently recheck the frozen records, encoded arrays and split isolation."""
import argparse
import json
from pathlib import Path

import numpy as np

from .build_dataset_v3 import audit_splits
from .experiment_v3 import load_arrays, read_jsonl, verify_dataset
from .preprocessing_v3 import ROOT, digest, encode, file_digest, structural_tokens


def audit(directory):
    directory = Path(directory)
    manifest = verify_dataset(directory)
    if manifest["preprocessor_sha256"] != file_digest(Path(__file__).with_name("preprocessing_v3.py")):
        raise ValueError("The preprocessing source differs from the frozen dataset.")
    vocabulary = json.loads((directory / "vocabulary.json").read_text(encoding="utf-8"))
    splits = {}
    for name in ("train", "validation", "calibration", "test", "source_holdout"):
        records = read_jsonl(directory / (name + ".jsonl.gz"))
        arrays = load_arrays(directory, name)
        if len(records) != len(arrays["labels"]) or arrays["offsets"][0] != 0 or np.any(np.diff(arrays["offsets"]) <= 0):
            raise ValueError(f"Invalid dimensions/offsets: {name}")
        for i, record in enumerate(records):
            if digest(record["tokens"]) != record["feature_sha256"] or digest(structural_tokens(record["tokens"])) != record["structural_sha256"]:
                raise ValueError(f"Record fingerprint mismatch: {name}/{i}")
            ids = encode(record["tokens"], vocabulary)
            start, end = arrays["offsets"][i:i + 2]
            if record["label"] != arrays["labels"][i] or not np.array_equal(ids, arrays["tokens"][start:end]):
                raise ValueError(f"Encoding/label mismatch: {name}/{i}")
        splits[name] = records
    result = audit_splits(splits, vocabulary)
    saved = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
    if result["splits"] != saved["splits"] or result["overlaps"] != saved["overlaps"]:
        raise ValueError("Saved audit does not match the independent recomputation.")
    return {"passed": True, "dataset_id": manifest["dataset_id"], "samples": sum(len(rows) for rows in splits.values()), "cross_split_overlaps": 0, "encodings_match_records": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset/v3")
    args = parser.parse_args()
    print(json.dumps(audit(args.dataset), indent=2))
