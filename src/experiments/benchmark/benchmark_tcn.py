"""A compact causal TCN trained from scratch on complete encoded contracts.

No sequence is truncated or independently labelled in windows. The default
two-convolution blocks have a 2045-position receptive field in the lossless
encoding, followed by masked global mean/max pooling. Global pooling sees all
positions; interactions beyond the receptive field are not directly modelled.
"""

import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from torch import nn

from src.data.batching import batches, collate
from src.evaluation.metrics import calculate_metrics
from src.preprocessing import ROOT, digest, file_digest
from src.provenance import source_path

FORMAT = "contract-tcn-v1"
ARCHITECTURE_KEYS = (
    "embedding_dim",
    "channels",
    "kernel_size",
    "dilations",
    "convolutions_per_block",
    "dropout",
    "dense_units",
)


def architecture_settings(settings):
    """The exact model topology, independent of inference batch configuration."""
    result = {key: settings[key] for key in ARCHITECTURE_KEYS}
    for key in (
        "embedding_dim",
        "channels",
        "kernel_size",
        "convolutions_per_block",
        "dense_units",
    ):
        value = result[key]
        if isinstance(value, bool) or int(value) != value or int(value) < 1:
            raise ValueError(f"TCN {key} must be a positive integer")
        result[key] = int(value)
    if not result["dilations"] or any(
        isinstance(d, bool) or int(d) != d or int(d) < 1 for d in result["dilations"]
    ):
        raise ValueError("TCN dilations must be positive integers")
    result["dilations"] = [int(d) for d in result["dilations"]]
    result["dropout"] = float(result["dropout"])
    if not 0 <= result["dropout"] < 1:
        raise ValueError("TCN dropout must be in [0, 1)")
    return result


def receptive_field(settings):
    settings = architecture_settings(settings)
    return 1 + settings["convolutions_per_block"] * (settings["kernel_size"] - 1) * sum(
        settings["dilations"]
    )


class CausalConv1d(nn.Module):
    """Explicit left padding: output at t never reads any input after t."""

    def __init__(self, input_channels, output_channels, kernel_size, dilation):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.convolution = nn.Conv1d(
            input_channels, output_channels, kernel_size, dilation=dilation
        )

    def forward(self, values):
        return self.convolution(nn.functional.pad(values, (self.left_padding, 0)))


class ResidualTCNBlock(nn.Module):
    """Causal convolutions/ReLU/dropout with a pointwise residual projection."""

    def __init__(self, input_channels, channels, kernel_size, dilation, convolutions, dropout):
        super().__init__()
        self.convolutions = nn.ModuleList(
            [
                CausalConv1d(
                    input_channels if i == 0 else channels, channels, kernel_size, dilation
                )
                for i in range(convolutions)
            ]
        )
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(convolutions)])
        self.projection = (
            nn.Conv1d(input_channels, channels, 1) if input_channels != channels else nn.Identity()
        )

    def forward(self, values, mask):
        values = values * mask
        residual = self.projection(values) * mask
        for convolution, dropout in zip(self.convolutions, self.dropouts):
            values = convolution(values) * mask
            values = torch.relu(values) * mask
            values = dropout(values) * mask
        return torch.relu(values + residual) * mask


class TCNContractModel(nn.Module):
    """Token embeddings, residual TCN, global masked pooling and binary head."""

    def __init__(self, vocabulary_size, settings):
        super().__init__()
        if int(vocabulary_size) != vocabulary_size or int(vocabulary_size) < 2:
            raise ValueError("TCN vocabulary_size must include PAD and non-PAD tokens")
        settings = architecture_settings(settings)
        self.settings = settings
        self.vocabulary_size = int(vocabulary_size)
        self.receptive_field = receptive_field(settings)
        self.embedding = nn.Embedding(
            self.vocabulary_size, settings["embedding_dim"], padding_idx=0
        )
        self.blocks = nn.ModuleList(
            [
                ResidualTCNBlock(
                    settings["embedding_dim"] if i == 0 else settings["channels"],
                    settings["channels"],
                    settings["kernel_size"],
                    dilation,
                    settings["convolutions_per_block"],
                    settings["dropout"],
                )
                for i, dilation in enumerate(settings["dilations"])
            ]
        )
        self.head = nn.Sequential(
            nn.Dropout(settings["dropout"]),
            nn.Linear(2 * settings["channels"], settings["dense_units"]),
            nn.ReLU(),
            nn.Linear(settings["dense_units"], 1),
        )

    def token_features(self, ids, lengths):
        """Return [batch, channels, time] features and a [batch, 1, time] mask."""
        if ids.ndim != 2 or lengths.ndim != 1 or len(lengths) != len(ids):
            raise ValueError("TCN expects batched IDs and one length per contract")
        if bool(torch.any(lengths <= 0)) or bool(torch.any(lengths > ids.shape[1])):
            raise ValueError("TCN contract lengths must be positive and fit the batch")
        positions = torch.arange(ids.shape[1], device=ids.device)[None, :]
        mask = (positions < lengths[:, None]) & ids.ne(0)
        mask = mask[:, None, :]
        values = self.embedding(ids).transpose(1, 2) * mask
        for block in self.blocks:
            values = block(values, mask)
        return values, mask

    def forward(self, ids, lengths):
        values, mask = self.token_features(ids, lengths)
        mean = (values * mask).sum(-1) / mask.sum(-1).clamp_min(1)
        maximum = values.masked_fill(~mask, -torch.inf).amax(-1)
        return self.head(torch.cat((mean, maximum), dim=-1)).squeeze(-1)


def _source_hashes():
    return {
        name: file_digest(source_path(name))
        for name in (
            "benchmark_tcn.py",
            "batching.py",
            "metrics.py",
            "preprocessing.py",
            "provenance.py",
        )
    }


def _resolve_model_path(model_path):
    path = Path(model_path)
    return path if path.is_absolute() else ROOT / path


def load_tcn(settings, vocabulary_size, model_path=None):
    """Build a fresh trainable TCN, or strictly verify and load its best artifact."""
    model = TCNContractModel(vocabulary_size, settings)
    if model_path is not None:
        artifact = torch.load(
            _resolve_model_path(model_path), map_location="cpu", weights_only=True
        )
        if not isinstance(artifact, dict) or artifact.get("format") != FORMAT:
            raise ValueError("Unrecognised TCN model artifact")
        if (
            artifact.get("architecture") != model.settings
            or artifact.get("vocabulary_size") != model.vocabulary_size
        ):
            raise ValueError(
                "TCN model architecture/vocabulary differ from the configured protocol"
            )
        if artifact.get("sources_sha256") != _source_hashes():
            raise ValueError("TCN model sources differ; use its frozen experiment code")
        state_dict = artifact.get("state_dict")
        if not isinstance(state_dict, dict) or any(
            not torch.isfinite(value).all() for value in state_dict.values()
        ):
            raise ValueError("TCN model contains missing/nonfinite weights")
        model.load_state_dict(state_dict, strict=True)
        model.eval()
    return model.to(torch.device(settings.get("device", "cpu")))


def _validate_arrays(arrays, vocabulary_size, name, allow_empty=False):
    tokens, offsets, labels = (np.asarray(arrays[key]) for key in ("tokens", "offsets", "labels"))
    if any(array.ndim != 1 for array in (tokens, offsets, labels)):
        raise ValueError(f"TCN {name} arrays must be packed one-dimensional arrays")
    if not np.issubdtype(tokens.dtype, np.integer) or not np.issubdtype(offsets.dtype, np.integer):
        raise ValueError(f"TCN {name} IDs and offsets must be integers")
    if (
        len(offsets) != len(labels) + 1
        or not len(offsets)
        or offsets[0] != 0
        or offsets[-1] != len(tokens)
    ):
        raise ValueError(f"TCN {name} packed dimensions are inconsistent")
    lengths = np.diff(offsets)
    if np.any(lengths <= 0) or (not len(labels) and not allow_empty):
        raise ValueError(f"TCN {name} must contain nonempty contracts")
    if np.any(tokens <= 0) or np.any(tokens >= vocabulary_size):
        raise ValueError(f"TCN {name} token IDs must be non-PAD IDs in the training vocabulary")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"TCN {name} expects binary contract labels")
    return lengths


def _validate_batch_settings(settings):
    for key in ("batch_size", "batch_token_budget"):
        value = settings[key]
        if isinstance(value, bool) or int(value) != value or int(value) < 1:
            raise ValueError(f"TCN {key} must be a positive integer")


def predict_tcn(model, arrays, settings):
    """One logit per complete contract, restored to the original input order."""
    _validate_batch_settings(settings)
    lengths = _validate_arrays(arrays, model.vocabulary_size, "prediction", allow_empty=True)
    model.eval()
    output = np.empty(len(lengths), dtype=np.float64)
    device = next(model.parameters()).device
    with torch.inference_mode():
        for indices in batches(lengths, settings):
            ids, sizes, _ = collate(arrays, indices)
            logits = model(ids.to(device), sizes.to(device))
            output[indices] = logits.detach().cpu().numpy()
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite TCN contract logits")
    return output


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_numpy(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value)
    os.replace(temporary, path)


def _rng_state():
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
    }
    if torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    value = state["numpy"]
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _input_fingerprint(config, train_arrays, validation_arrays, weights, seed, vocabulary_size):
    hasher = hashlib.sha256()
    hasher.update(
        digest(
            {
                "config": config,
                "seed": int(seed),
                "vocabulary_size": int(vocabulary_size),
                "sources_sha256": _source_hashes(),
            }
        ).encode()
    )
    for name, arrays in (("train", train_arrays), ("validation", validation_arrays)):
        hasher.update(name.encode())
        for key in ("tokens", "offsets", "labels"):
            value = np.ascontiguousarray(arrays[key])
            hasher.update(
                digest({"name": key, "dtype": value.dtype.str, "shape": value.shape}).encode()
            )
            hasher.update(memoryview(value).cast("B"))
    hasher.update(np.ascontiguousarray(weights, dtype=np.float64).tobytes())
    return hasher.hexdigest()


def _train_batch(model, optimizer, arrays, weights, indices, settings):
    """An optimizer step on mean weighted BCE, with one term per contract."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    ids, sizes, labels = collate(arrays, indices)
    logits = model(ids.to(device), sizes.to(device))
    losses = nn.functional.binary_cross_entropy_with_logits(
        logits, labels.to(device=device, dtype=logits.dtype), reduction="none"
    )
    weighted = losses * torch.as_tensor(weights[indices], dtype=logits.dtype, device=device)
    loss = weighted.mean()
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite TCN training loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip"]))
    if not torch.isfinite(norm):
        raise ValueError("Nonfinite TCN training gradient")
    optimizer.step()
    return float(weighted.detach().sum().cpu())


def _batch_description(indices, lengths):
    longest = int(lengths[indices].max())
    return {
        "contract_indices": indices.tolist(),
        "contracts": len(indices),
        "maximum_encoded_length": longest,
        "encoded_tokens": int(lengths[indices].sum()),
        "padded_encoded_positions": len(indices) * longest,
    }


def _representative_batches(all_batches, lengths):
    maximum_lengths = np.asarray([lengths[indices].max() for indices in all_batches])
    chosen, result = set(), []
    for quantile in (0.5, 0.9):
        order = np.argsort(
            abs(maximum_lengths - np.quantile(maximum_lengths, quantile)), kind="stable"
        )
        index = next((int(i) for i in order if int(i) not in chosen), None)
        if index is None:
            continue
        chosen.add(index)
        result.append((quantile, all_batches[index]))
    return result


def _sync(model):
    if next(model.parameters()).device.type == "cuda":
        torch.cuda.synchronize(next(model.parameters()).device)


def _pilot_epoch_estimate(all_batches, lengths, measured):
    """Interpolate cost per padded position over actual batch maximum lengths."""
    order = sorted(measured, key=lambda row: row["maximum_encoded_length"])
    positions = [row["maximum_encoded_length"] for row in order]
    rates = [row["median_seconds"] / row["padded_encoded_positions"] for row in order]
    return float(
        sum(
            len(indices)
            * int(lengths[indices].max())
            * np.interp(int(lengths[indices].max()), positions, rates)
            for indices in all_batches
        )
    )


def _run_pilot(model, train_arrays, validation_arrays, weights, settings, seed, result_dir, seeds):
    train_lengths, validation_lengths = (
        np.diff(train_arrays["offsets"]),
        np.diff(validation_arrays["offsets"]),
    )
    train_batches, validation_batches = (
        batches(train_lengths, settings),
        batches(validation_lengths, settings),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    measured = {"training": [], "validation": []}
    for phase, all_batches, lengths, arrays in (
        ("training", train_batches, train_lengths, train_arrays),
        ("validation", validation_batches, validation_lengths, validation_arrays),
    ):
        for quantile, indices in _representative_batches(all_batches, lengths):
            description = _batch_description(indices, lengths)
            print(
                json.dumps(
                    {
                        "stage": "tcn_pilot_measurement",
                        "phase": phase,
                        "seed": seed,
                        "representative_quantile": quantile,
                        **description,
                    }
                ),
                flush=True,
            )

            def operation():
                if phase == "training":
                    _train_batch(model, optimizer, arrays, weights, indices, settings)
                else:
                    model.eval()
                    ids, sizes, _ = collate(arrays, indices)
                    device = next(model.parameters()).device
                    with torch.inference_mode():
                        model(ids.to(device), sizes.to(device))

            operation()  # Actual complete-batch warm-up, excluded from timings.
            _sync(model)
            timings = []
            for _ in range(3):
                started = time.perf_counter()
                operation()
                _sync(model)
                timings.append(time.perf_counter() - started)
            row = {
                **description,
                "representative_quantile": quantile,
                "warmup_batches": 1,
                "timed_batches": 3,
                "seconds": timings,
                "median_seconds": float(np.median(timings)),
                "q90_seconds": float(np.quantile(timings, 0.9)),
            }
            measured[phase].append(row)
            print(
                json.dumps(
                    {"stage": "tcn_pilot_batch_complete", "phase": phase, "seed": seed, **row}
                ),
                flush=True,
            )
    training_epoch = _pilot_epoch_estimate(train_batches, train_lengths, measured["training"])
    validation_epoch = _pilot_epoch_estimate(
        validation_batches, validation_lengths, measured["validation"]
    )
    epochs = int(settings["epochs"])
    result = {
        "status": "pilot_complete",
        "seed": seed,
        "architecture": "tcn",
        "final_training_started": False,
        "input_mode": "full",
        "truncation": False,
        "initialization": "from_scratch",
        "receptive_field_encoded_tokens": model.receptive_field,
        "measured_batches": measured,
        "representative_batch_selection": "median_and_q90_of_actual_batch_maximum_encoded_lengths",
        "training_batches_per_epoch": len(train_batches),
        "validation_batches_per_epoch": len(validation_batches),
        "training_contracts": len(train_lengths),
        "validation_contracts": len(validation_lengths),
        "training_encoded_tokens": int(train_lengths.sum()),
        "validation_encoded_tokens": int(validation_lengths.sum()),
        "training_encoded_length_quantiles": {
            str(q): float(np.quantile(train_lengths, q)) for q in (0.5, 0.9, 0.99, 1.0)
        },
        "estimated_training_seconds_per_epoch": training_epoch,
        "estimated_validation_seconds_per_epoch": validation_epoch,
        "maximum_epochs_per_seed": epochs,
        "seeds": list(seeds),
        "estimated_training_and_validation_seconds_per_seed": epochs
        * (training_epoch + validation_epoch),
        "estimated_training_and_validation_seconds_all_seeds": len(seeds)
        * epochs
        * (training_epoch + validation_epoch),
        "estimate_scope": "Representative median/q90 actual batches after warm-up; interpolate time per padded encoded position by batch maximum length. Maximum-epoch estimate excludes checkpoint I/O, metric calculation, test evaluation and pauses; extreme lengths can cost more and early stopping can shorten training.",
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    _atomic_json(result_dir / "pilot.json", result)
    _atomic_json(
        result_dir / "progress.json",
        {
            "status": "pilot_complete",
            "seed": seed,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return result


def train_tcn(
    config,
    train_arrays,
    validation_arrays,
    weights,
    result_dir,
    model_dir,
    seed,
    vocabulary_size,
    pilot=False,
):
    """Fit one seed, resume exactly, and select its best validation epoch only.

    Durable snapshots contain the optimizer, RNG, current batch, history and
    best model together. An interruption during a step replays from the last
    durable snapshot, avoiding partial gradients or consumed dropout RNG.
    """
    settings = config["tcn"]
    architecture_settings(settings)
    _validate_batch_settings(settings)
    for key in ("epochs", "early_stopping_patience", "checkpoint_every_batches", "cpu_threads"):
        value = settings[key]
        if isinstance(value, bool) or int(value) != value or int(value) < 1:
            raise ValueError(f"TCN {key} must be a positive integer")
    for key in ("learning_rate", "gradient_clip"):
        if not np.isfinite(float(settings[key])) or float(settings[key]) <= 0:
            raise ValueError(f"TCN {key} must be positive and finite")
    if not np.isfinite(float(settings["weight_decay"])) or float(settings["weight_decay"]) < 0:
        raise ValueError("TCN weight_decay must be nonnegative and finite")
    train_lengths = _validate_arrays(train_arrays, vocabulary_size, "train")
    _validate_arrays(validation_arrays, vocabulary_size, "validation")
    weights = np.asarray(weights, dtype=np.float64)
    if (
        weights.shape != (len(train_lengths),)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.any(weights > 0)
    ):
        raise ValueError("TCN training weights do not match the contracts")
    result_dir, model_dir = Path(result_dir), Path(model_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    # Relative artifact provenance is intentional: refuse paths outside ROOT
    # before training rather than only at final serialization.
    model_dir.resolve().relative_to(ROOT)
    torch.set_num_threads(int(settings["cpu_threads"]))
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = load_tcn(settings, vocabulary_size)
    fingerprint = _input_fingerprint(
        config, train_arrays, validation_arrays, weights, seed, vocabulary_size
    )
    if pilot:
        report = _run_pilot(
            model,
            train_arrays,
            validation_arrays,
            weights,
            settings,
            seed,
            result_dir,
            config.get("seeds", [seed]),
        )
        report["input_fingerprint"] = fingerprint
        _atomic_json(result_dir / "pilot.json", report)
        return report
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path, checkpoint_path = model_dir / "weights.pt", model_dir / "resume.pt"
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    state = {
        "fingerprint": fingerprint,
        "epoch": 0,
        "batch": 0,
        "optimizer_steps": 0,
        "epoch_loss_sum": 0.0,
        "epoch_seen": 0,
        "history": [],
        "best": None,
        "best_epoch": None,
        "stale": 0,
        "training_seconds": 0.0,
    }
    best_state = None
    resumed = checkpoint_path.exists()
    if resumed:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != FORMAT or checkpoint["state"]["fingerprint"] != fingerprint:
            raise ValueError("TCN checkpoint inputs/settings/sources differ; use a new experiment")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        state, best_state = checkpoint["state"], checkpoint["best_model"]
        _restore_rng(checkpoint["rng"])
        if state["best"] is not None and best_state is None:
            raise ValueError("TCN resume checkpoint is missing its best weights")
    _atomic_json(result_dir / "history.json", state["history"])
    last_clock = time.perf_counter()

    def checkpoint():
        nonlocal last_clock
        state["training_seconds"] += time.perf_counter() - last_clock
        _atomic_torch(
            checkpoint_path,
            {
                "format": FORMAT,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "state": state,
                "best_model": best_state,
                "rng": _rng_state(),
            },
        )
        last_clock = time.perf_counter()
        _atomic_json(
            result_dir / "progress.json",
            {
                **state,
                "seed": seed,
                "status": "training",
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    if not resumed:
        checkpoint()
    print(
        json.dumps(
            {
                "stage": "tcn_training_start",
                "seed": seed,
                "resumed": resumed,
                "epoch": state["epoch"] + 1,
                "next_batch": state["batch"],
                "training_contracts": len(train_lengths),
                "training_encoded_tokens": int(train_lengths.sum()),
                "receptive_field_encoded_tokens": model.receptive_field,
            }
        ),
        flush=True,
    )
    try:
        while state["epoch"] < int(settings["epochs"]) and state["stale"] < int(
            settings["early_stopping_patience"]
        ):
            epoch = state["epoch"]
            epoch_batches = batches(train_lengths, settings, seed=seed + epoch)
            for batch_number in range(state["batch"], len(epoch_batches)):
                indices = epoch_batches[batch_number]
                loss_sum = _train_batch(model, optimizer, train_arrays, weights, indices, settings)
                state["optimizer_steps"] += 1
                state["batch"] = batch_number + 1
                state["epoch_seen"] += len(indices)
                state["epoch_loss_sum"] += loss_sum
                if state["optimizer_steps"] % int(settings["checkpoint_every_batches"]) == 0:
                    checkpoint()
                    print(
                        json.dumps(
                            {
                                "stage": "tcn_training",
                                "seed": seed,
                                "epoch": epoch + 1,
                                "batch": state["batch"],
                                "batches_per_epoch": len(epoch_batches),
                                "contracts_seen": state["epoch_seen"],
                                "optimizer_steps": state["optimizer_steps"],
                            }
                        ),
                        flush=True,
                    )
            if state["epoch_seen"] != len(train_lengths):
                raise ValueError("TCN epoch did not visit every training contract")
            logits = predict_tcn(model, validation_arrays, settings)
            metrics = calculate_metrics(validation_arrays["labels"], expit(logits), 0.5)
            # Stable, unweighted BCE per contract also avoids probability clipping
            # when logits are large. calculate_metrics retains the shared schema.
            value = float(np.mean(np.logaddexp(0, logits) - validation_arrays["labels"] * logits))
            if state["best"] is None or value < state["best"]:
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                state["best"], state["best_epoch"], state["stale"] = value, epoch + 1, 0
            else:
                state["stale"] += 1
            state["history"].append(
                {
                    "epoch": epoch + 1,
                    "loss": state["epoch_loss_sum"] / state["epoch_seen"],
                    "validation_log_loss": value,
                    "validation_f1_macro": metrics["f1_macro"],
                    "validation_roc_auc": metrics["roc_auc"],
                    "contracts_seen": state["epoch_seen"],
                    "optimizer_steps": state["optimizer_steps"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            state["epoch"] += 1
            state["batch"], state["epoch_seen"], state["epoch_loss_sum"] = 0, 0, 0.0
            checkpoint()
            _atomic_json(result_dir / "history.json", state["history"])
            print(
                json.dumps({"stage": "tcn_epoch_complete", "seed": seed, **state["history"][-1]}),
                flush=True,
            )
    except KeyboardInterrupt:
        # A partially executed optimizer step can already have mutated tensors.
        # Keep the last atomic snapshot, whose RNG precedes any replayed steps.
        durable = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["state"]
        _atomic_json(
            result_dir / "progress.json",
            {
                **durable,
                "seed": seed,
                "status": "interrupted",
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise
    model.load_state_dict(best_state, strict=True)
    logits = predict_tcn(model, validation_arrays, settings)
    _atomic_numpy(result_dir / "validation_logits.npy", logits)
    _atomic_torch(
        best_path,
        {
            "format": FORMAT,
            "architecture": model.settings,
            "vocabulary_size": int(vocabulary_size),
            "state_dict": best_state,
            "sources_sha256": _source_hashes(),
            "input_fingerprint": fingerprint,
            "seed": int(seed),
            "best_epoch": state["best_epoch"],
        },
    )
    validation = calculate_metrics(validation_arrays["labels"], expit(logits), 0.5)
    validation["log_loss"] = float(
        np.mean(np.logaddexp(0, logits) - validation_arrays["labels"] * logits)
    )
    summary = {
        "seed": seed,
        "status": "complete",
        "architecture": "tcn",
        "initialization": "from_scratch",
        "input_mode": "full",
        "truncation": False,
        "loss_unit": "contract",
        "pooling": "masked_global_mean_and_max",
        "activation": "relu",
        "padding": "causal_left_padding_and_mask_after_each_sublayer",
        "dropout_placement": "after_each_convolution_relu_and_before_dense_head",
        "receptive_field_encoded_tokens": model.receptive_field,
        "receptive_field_note": "Positions are losslessly encoded vocabulary/byte-escape IDs, not lexical tokens. Global pooling includes every position, but direct temporal interactions are limited to this receptive field.",
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "epochs_executed": len(state["history"]),
        "best_epoch": state["best_epoch"],
        "training_seconds": state["training_seconds"],
        "optimizer_steps": state["optimizer_steps"],
        "training_contracts": len(train_lengths),
        "validation_contracts": len(validation_arrays["labels"]),
        "training_encoded_tokens": int(train_lengths.sum()),
        "validation_encoded_tokens": int(len(validation_arrays["tokens"])),
        "training_weights_mean": float(weights.mean()),
        "training_weights_sha256": digest(weights.tobytes()),
        "validation": validation,
        "model_path": best_path.resolve().relative_to(ROOT).as_posix(),
        "model_sha256": file_digest(best_path),
        "input_fingerprint": fingerprint,
        "sources_sha256": _source_hashes(),
    }
    _atomic_json(result_dir / "run.json", summary)
    _atomic_json(
        result_dir / "progress.json",
        {
            **state,
            "seed": seed,
            "status": "complete",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return summary
