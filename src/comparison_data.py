"""Immutable study inputs, consistent truncation and length-aware batches."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .experiment_v3 import load_arrays, read_jsonl, training_weights, verify_dataset
from .preprocessing_v3 import ROOT, digest, encode

SPLITS = ("train", "validation", "calibration", "test", "source_holdout")


def load_study_config(path=None):
    path = Path(path) if path else ROOT / "config/comparison_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def pack_sequences(sequences, labels):
    lengths = np.asarray([len(row) for row in sequences], dtype=np.int64)
    if np.any(lengths <= 0):
        raise ValueError("Empty encoded contract")
    return {"tokens": np.concatenate(sequences).astype(np.int32), "offsets": np.cumsum(np.r_[0, lengths]), "labels": np.asarray(labels, dtype=np.int32)}


def prepare_inputs(config, destination):
    directory = ROOT / config["dataset_dir"]
    manifest = verify_dataset(directory)
    if manifest["dataset_id"] != config["dataset_id"]:
        raise ValueError("Dataset differs from preregistered study")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / "manifest.json"
    if marker.exists():
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if metadata["dataset_id"] != config["dataset_id"]:
            raise ValueError("Cached dataset mismatch")
        return metadata
    vocabulary = json.loads((directory / "vocabulary.json").read_text(encoding="utf-8"))
    metadata = {"dataset_id": manifest["dataset_id"], "vocabulary_size": len(vocabulary), "truncation_unit": "first_512_normalized_lexical_tokens_before_lossless_encoding", "splits": {}, "cross_split_truncated_collisions": {}}
    collisions = {}
    for split in SPLITS:
        rows = read_jsonl(directory / (split + ".jsonl.gz"))
        short = pack_sequences([encode(row["tokens"][:512], vocabulary) for row in rows], [row["label"] for row in rows])
        np.savez_compressed(destination / (split + "_first512.npz"), **short)
        collisions[split] = defaultdict(list)
        for i, row in enumerate(rows):
            collisions[split][digest(row["tokens"][:512])].append(i)
        metadata["splits"][split] = {"samples": len(rows), "truncated_contracts": sum(len(row["tokens"]) > 512 for row in rows), "full_lexical_tokens": sum(len(row["tokens"]) for row in rows), "truncated_lexical_tokens": sum(min(512, len(row["tokens"])) for row in rows), "sample_ids_sha256": digest([row["sample_id"] for row in rows])}
        save_json(destination / (split + "_metadata.json"), [{k: row[k] for k in ("sample_id", "group_id", "source", "granularity", "label_confidence", "label")} for row in rows])
        if split == "train":
            np.save(destination / "training_weights.npy", training_weights(rows))
    # Truncation can reintroduce identical model inputs despite disjoint full-code
    # families. Do not repartition per model; disclose overlap and evaluate a
    # collision-free diagnostic subset using training/calibration/validation only.
    for split in ("test", "source_holdout"):
        seen = set().union(*(collisions[name] for name in ("train", "validation", "calibration")))
        duplicate_keys = seen.intersection(collisions[split])
        affected = sorted(i for key in duplicate_keys for i in collisions[split][key])
        metadata["cross_split_truncated_collisions"][split] = {"groups": len(duplicate_keys), "affected_indices": affected, "affected_samples": len(affected)}
    save_json(marker, metadata)
    return metadata


def study_arrays(config, cache, split, mode):
    if mode == "full":
        return load_arrays(ROOT / config["dataset_dir"], split)
    with np.load(Path(cache) / (split + "_first512.npz"), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def batches(lengths, config, seed=None):
    """Every record once; bound padding cost while retaining very long contracts."""
    lengths = np.asarray(lengths)
    order = np.argsort(lengths, kind="stable")
    output, current, longest = [], [], 0
    for index in order:
        length = int(lengths[index])
        if current and (len(current) >= config["batch_size"] or (len(current) + 1) * length > config["batch_token_budget"]):
            output.append(np.asarray(current))
            current = []
        current.append(int(index))
        longest = length
    if current:
        output.append(np.asarray(current))
    if seed is not None:
        rng = np.random.default_rng(seed)
        rng.shuffle(output)
        for indices in output:
            rng.shuffle(indices)
    return output


def collate(arrays, indices):
    import torch
    lengths = np.diff(arrays["offsets"])[indices]
    ids = np.zeros((len(indices), int(lengths.max())), dtype=np.int64)
    for n, index in enumerate(indices):
        start, end = arrays["offsets"][index:index + 2]
        ids[n, :end - start] = arrays["tokens"][start:end]
    return torch.from_numpy(ids), torch.from_numpy(lengths), torch.as_tensor(arrays["labels"][indices], dtype=torch.float32)
