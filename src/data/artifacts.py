"""Read and verify the immutable datasets shared by every experiment."""

import gzip
import json
from pathlib import Path

import numpy as np

from src.preprocessing import VERSION, digest, file_digest


def read_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def verify_dataset(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    body = {k: v for k, v in manifest.items() if k != "dataset_id"}
    if digest(body) != manifest["dataset_id"] or manifest["preprocessing_version"] != VERSION:
        raise ValueError("Dataset manifest/version mismatch.")
    for name, expected in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or file_digest(path) != expected:
            raise ValueError(f"Dataset artifact mismatch: {name}")
    return manifest


def load_arrays(directory, split):
    with np.load(Path(directory) / (split + ".npz"), allow_pickle=False) as data:
        result = {name: data[name] for name in data.files}
    if len(result["offsets"]) != len(result["labels"]) + 1 or result["offsets"][-1] != len(
        result["tokens"]
    ):
        raise ValueError("Corrupt packed dataset dimensions.")
    return result
