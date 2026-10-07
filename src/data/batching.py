"""Training weights and length-aware batches shared across model families."""

from collections import Counter

import numpy as np


def training_weights(records):
    """Balance source/label strata using training examples only; cap rare-stratum weights."""
    counts = Counter((r["source"], r["label"]) for r in records)
    weights = np.asarray(
        [min(3.0, len(records) / (len(counts) * counts[r["source"], r["label"]])) for r in records],
        dtype=np.float32,
    )
    return weights / weights.mean()


def batches(lengths, config, seed=None):
    """Every record once; bound padding cost while retaining very long contracts."""
    lengths = np.asarray(lengths)
    order = np.argsort(lengths, kind="stable")
    output, current = [], []
    for index in order:
        length = int(lengths[index])
        if current and (
            len(current) >= config["batch_size"]
            or (len(current) + 1) * length > config["batch_token_budget"]
        ):
            output.append(np.asarray(current))
            current = []
        current.append(int(index))
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
        start, end = arrays["offsets"][index : index + 2]
        ids[n, : end - start] = arrays["tokens"][start:end]
    return (
        torch.from_numpy(ids),
        torch.from_numpy(lengths),
        torch.as_tensor(arrays["labels"][indices], dtype=torch.float32),
    )


def pack_sequences(sequences, labels):
    lengths = np.asarray([len(row) for row in sequences], dtype=np.int64)
    if np.any(lengths <= 0):
        raise ValueError("Empty encoded contract")
    return {
        "tokens": np.concatenate(sequences).astype(np.int32),
        "offsets": np.cumsum(np.r_[0, lengths]),
        "labels": np.asarray(labels, dtype=np.int32),
    }
