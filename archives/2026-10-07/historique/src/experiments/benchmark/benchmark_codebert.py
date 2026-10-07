"""End-to-end CodeBERT adaptation with a loss and predictions per contract.

Every BPE subtoken is retained. A contract's logit is the mean of its window
CLS logits; its label is never interpreted as a label for each window. With
dropout disabled, two forward passes compute the same BCE gradient as retaining
all window graphs, while retaining only one microbatch's activations at a time.
"""

import hashlib
import json
import math
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from scipy.special import expit
from torch import nn

from src.evaluation.metrics import calculate_metrics
from src.preprocessing import ROOT, digest, file_digest


class CodeBERTContractModel(nn.Module):
    """Learned linear CLS head and fully trainable pretrained encoder."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.config.hidden_size, 1)
        self.pad_token_id = encoder.config.pad_token_id
        self.config = encoder.config
        self.requires_grad_(True)
        self.eval()  # Disables stochastic layers, without disabling autograd.

    def forward(self, input_ids, attention_mask):
        states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(states[:, 0]).squeeze(-1)


def load_encoder(settings):
    """Load only the explicitly pinned revision; import Transformers lazily."""
    if not re.fullmatch(r"[0-9a-fA-F]{40}", settings.get("revision", "")):
        raise ValueError("CodeBERT revision must be a pinned 40-character commit SHA")
    if float(settings.get("dropout", 0)) != 0:
        raise ValueError("Exact two-pass training requires dropout=0")
    from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast

    kwargs = {
        "revision": settings["revision"],
        "cache_dir": str(ROOT / ".cache-comparison/huggingface"),
    }
    tokenizer = RobertaTokenizerFast.from_pretrained(settings["model_id"], **kwargs)
    model_config = RobertaConfig.from_pretrained(settings["model_id"], **kwargs)
    model_config.hidden_dropout_prob = 0.0
    model_config.attention_probs_dropout_prob = 0.0
    model_config.classifier_dropout = 0.0
    encoder = RobertaModel.from_pretrained(
        settings["model_id"],
        config=model_config,
        add_pooling_layer=False,
        attn_implementation="eager",
        **kwargs,
    )
    # The smallest supported position limit also includes RoBERTa's padding offset.
    limit = model_config.max_position_embeddings - model_config.pad_token_id - 1
    if int(settings["chunk_subtokens"]) > limit:
        raise ValueError(f"CodeBERT chunks exceed the encoder's {limit}-token limit")
    model = CodeBERTContractModel(encoder)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    model.to(torch.device(settings.get("device", "cpu")))
    return model, tokenizer


def tokenize_contract(tokens, tokenizer, chunk_subtokens):
    """Return nonoverlapping complete BPE windows, each with its special tokens."""
    specials = int(tokenizer.num_special_tokens_to_add(pair=False))
    capacity = int(chunk_subtokens) - specials
    if capacity < 1:
        raise ValueError("chunk_subtokens must leave room after special tokens")
    ids = tokenizer.encode(" ".join(tokens), add_special_tokens=False, truncation=False)
    if not ids:
        raise ValueError("Empty CodeBERT input")
    chunks = [
        tokenizer.build_inputs_with_special_tokens(ids[start : start + capacity])
        for start in range(0, len(ids), capacity)
    ]
    if any(len(chunk) > chunk_subtokens for chunk in chunks):
        raise ValueError("Tokenizer produced a chunk exceeding its configured limit")
    return chunks


def _chunk_logits(model, chunks):
    if not chunks:
        raise ValueError("Empty window batch")
    device = next(model.parameters()).device
    pad = getattr(model, "pad_token_id", 1)
    longest = max(len(chunk) for chunk in chunks)
    ids = torch.full((len(chunks), longest), int(pad), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for row, chunk in enumerate(chunks):
        ids[row, : len(chunk)] = torch.as_tensor(chunk, dtype=torch.long, device=device)
        mask[row, : len(chunk)] = 1
    logits = model(input_ids=ids, attention_mask=mask)
    if hasattr(logits, "logits"):
        logits = logits.logits
    logits = logits.reshape(-1)
    if logits.numel() != len(chunks) or not torch.isfinite(logits).all():
        raise ValueError("Invalid CodeBERT window logits")
    return logits


def _contract_logit(model, chunks, chunk_batch_size):
    if not chunks or chunk_batch_size < 1:
        raise ValueError("A contract must contain windows and chunk_batch_size must be positive")
    total, dtype = None, None
    for start in range(0, len(chunks), chunk_batch_size):
        logits = _chunk_logits(model, chunks[start : start + chunk_batch_size])
        dtype = logits.dtype
        value = logits.double().sum()
        total = value if total is None else total + value
    return (total / len(chunks)).to(dtype=dtype)


def predict_contracts(model, encoded, chunk_batch_size):
    """Predict one logit per contract in input order without retaining graphs."""
    model.eval()
    output = np.empty(len(encoded), dtype=np.float64)
    with torch.no_grad():
        for index, chunks in enumerate(encoded):
            output[index] = float(_contract_logit(model, chunks, chunk_batch_size))
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite CodeBERT predictions")
    return output


def backward_contract(model, chunks, label, weight, batch_contracts, chunk_batch_size):
    """Accumulate the exact weighted contract BCE gradient, not window losses.

    The caller zeroes gradients before the effective batch and steps afterwards.
    It must not update any weights between this contract's two passes.
    """
    if batch_contracts < 1 or label not in (0, 1) or not math.isfinite(float(weight)) or weight < 0:
        raise ValueError("Invalid contract label, weight or effective batch size")
    if any(isinstance(module, nn.Dropout) and module.p != 0 for module in model.modules()):
        raise ValueError("Exact two-pass gradients require dropout=0")
    model.eval()
    with torch.no_grad():
        logit = _contract_logit(model, chunks, chunk_batch_size)
        target = torch.as_tensor(label, dtype=logit.dtype, device=logit.device)
        loss = nn.functional.binary_cross_entropy_with_logits(logit, target)
        coefficient = (
            (torch.sigmoid(logit) - target) * float(weight) / (batch_contracts * len(chunks))
        )
    for start in range(0, len(chunks), chunk_batch_size):
        logits = _chunk_logits(model, chunks[start : start + chunk_batch_size])
        (logits.sum() * coefficient).backward()
    return float(loss) * float(weight), float(logit)


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


def _input_fingerprint(config, train_rows, validation_rows, weights, seed):
    # Hash rows incrementally; avoid constructing another copy of a large dataset.
    hasher = hashlib.sha256()
    hasher.update(digest({"config": config, "seed": seed}).encode())
    for name, rows in (("train", train_rows), ("validation", validation_rows)):
        hasher.update(name.encode())
        for row in rows:
            hasher.update(
                digest(
                    {
                        "tokens": row["tokens"],
                        "label": row["label"],
                        "sample_id": row.get("sample_id"),
                    }
                ).encode()
            )
    hasher.update(np.asarray(weights, dtype=np.float64).tobytes())
    return hasher.hexdigest()


def _run_pilot(
    model, encoded_train, encoded_validation, weights, labels, settings, seed, result_dir, seeds
):
    """Measure representative contracts; do not produce a trained final model."""
    counts = np.asarray([len(chunks) for chunks in encoded_train])
    indices = []
    for quantile in (0.5, 0.9):
        candidates = np.argsort(abs(counts - np.quantile(counts, quantile)), kind="stable")
        chosen = next((int(index) for index in candidates if int(index) not in indices), None)
        if chosen is not None:
            indices.append(chosen)
    print(
        json.dumps(
            {
                "stage": "codebert_pilot_measurement",
                "seed": seed,
                "representative_contract_indices": indices,
                "representative_window_counts": counts[indices].tolist(),
                "training_windows": int(counts.sum()),
            }
        ),
        flush=True,
    )
    # Small deterministic warm-up excludes initial graph/kernel setup.
    predict_contracts(model, [[encoded_train[indices[0]][0]]], settings["chunk_batch_size"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    for index in indices:
        backward_contract(
            model,
            encoded_train[index],
            labels[index],
            weights[index],
            len(indices),
            settings["chunk_batch_size"],
        )
        print(
            json.dumps(
                {
                    "stage": "codebert_pilot_contract_complete",
                    "seed": seed,
                    "contract_index": index,
                    "windows": int(counts[index]),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    training_seconds = time.perf_counter() - started
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    predict_contracts(
        model, [encoded_train[index] for index in indices], settings["chunk_batch_size"]
    )
    forward_seconds = time.perf_counter() - started
    measured_windows = int(counts[indices].sum())
    total_train_windows = int(counts.sum())
    total_validation_windows = sum(map(len, encoded_validation))
    epochs = int(settings["epochs"])
    per_seed = epochs * (
        training_seconds / measured_windows * total_train_windows
        + forward_seconds / measured_windows * total_validation_windows
    )
    result = {
        "status": "pilot_complete",
        "seed": seed,
        "final_training_started": False,
        "representative_contract_indices": indices,
        "representative_window_counts": counts[indices].tolist(),
        "measured_training_seconds": training_seconds,
        "measured_forward_seconds": forward_seconds,
        "measured_training_seconds_per_window": training_seconds / measured_windows,
        "measured_forward_seconds_per_window": forward_seconds / measured_windows,
        "training_windows": total_train_windows,
        "validation_windows": total_validation_windows,
        "training_window_count_quantiles": {
            str(q): float(np.quantile(counts, q)) for q in (0.5, 0.9, 0.99)
        },
        "estimated_training_and_validation_seconds_per_seed": per_seed,
        "estimated_training_and_validation_seconds_all_seeds": per_seed * len(seeds),
        "estimate_scope": "linear window-count extrapolation; excludes download, tokenization, checkpoint I/O, test evaluation and pauses; early stopping can shorten training",
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    _atomic_json(result_dir / "pilot.json", result)
    return result


def train_codebert(
    config, train_rows, validation_rows, weights, result_dir, model_dir, seed, pilot=False
):
    """Train/restore one seed, select using validation only and persist logits.

    Checkpoints are written only at completed optimizer steps. A resumed process
    repeats at most checkpoint_every_steps completed batches; it restores the
    exact optimizer/RNG and deterministic permutation for the saved position.
    The model directory contains a best-weights checkpoint and a rolling resume
    checkpoint; both are local artifacts and can be large for a full encoder.
    The default protocol runs on CPU; another device requires explicit settings
    in a separate experiment configuration.
    """
    settings = config["codebert"]
    if float(settings.get("dropout", 0)) != 0:
        raise ValueError("Exact two-pass training requires dropout=0")
    integer_settings = (
        "chunk_subtokens",
        "chunk_batch_size",
        "effective_batch_size",
        "epochs",
        "early_stopping_patience",
        "cpu_threads",
        "checkpoint_every_steps",
    )
    if any(int(settings[key]) < 1 for key in integer_settings):
        raise ValueError("All CodeBERT size/epoch/checkpoint settings must be positive")
    if not train_rows or not validation_rows:
        raise ValueError("CodeBERT requires nonempty training and validation partitions")
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(train_rows),) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Training weights do not match the contracts")
    if any(row["label"] not in (0, 1) for rows in (train_rows, validation_rows) for row in rows):
        raise ValueError("CodeBERT expects binary labels")
    result_dir, model_dir = Path(result_dir), Path(model_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(settings["cpu_threads"]))
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model, tokenizer = load_encoder(settings)
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("CodeBERT encoder and head must both be trainable")
    configured_device = torch.device(settings.get("device", "cpu"))
    actual_device = next(model.parameters()).device
    if actual_device.type != configured_device.type or (
        configured_device.index is not None and actual_device.index != configured_device.index
    ):
        raise ValueError("CodeBERT model device differs from its configured protocol")
    print(
        json.dumps(
            {
                "stage": "codebert_tokenization",
                "seed": seed,
                "contracts": len(train_rows) + len(validation_rows),
            }
        ),
        flush=True,
    )
    encoded_train = [
        tokenize_contract(row["tokens"], tokenizer, settings["chunk_subtokens"])
        for row in train_rows
    ]
    encoded_validation = [
        tokenize_contract(row["tokens"], tokenizer, settings["chunk_subtokens"])
        for row in validation_rows
    ]
    labels = np.asarray([row["label"] for row in train_rows])
    validation_labels = np.asarray([row["label"] for row in validation_rows])
    if pilot:
        return _run_pilot(
            model,
            encoded_train,
            encoded_validation,
            weights,
            labels,
            settings,
            seed,
            result_dir,
            config.get("seeds", [seed]),
        )
    fingerprint = _input_fingerprint(config, train_rows, validation_rows, weights, seed)
    best_path = model_dir / f"seed-{seed}.pt"
    checkpoint_path = model_dir / f"seed-{seed}.resume.pt"
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    state = {
        "fingerprint": fingerprint,
        "epoch": 0,
        "position": 0,
        "optimizer_steps": 0,
        "epoch_loss_sum": 0.0,
        "epoch_seen": 0,
        "history": [],
        "best": None,
        "best_epoch": None,
        "stale": 0,
        "training_seconds": 0.0,
    }
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["state"]["fingerprint"] != fingerprint:
            raise ValueError("CodeBERT checkpoint inputs/settings differ; use a new experiment")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        state = checkpoint["state"]
        _restore_rng(checkpoint["rng"])
        if state["best"] is not None and not best_path.exists():
            raise ValueError("CodeBERT resume checkpoint is missing its best weights")
    last_clock = time.perf_counter()

    def checkpoint():
        nonlocal last_clock
        now = time.perf_counter()
        state["training_seconds"] += now - last_clock
        _atomic_torch(
            checkpoint_path,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "state": state,
                "rng": _rng_state(),
            },
        )
        # Exclude checkpoint I/O from active compute time.
        last_clock = time.perf_counter()
        _atomic_json(
            result_dir / "progress.json",
            {**state, "updated_utc": datetime.now(timezone.utc).isoformat(), "status": "training"},
        )

    try:
        while state["epoch"] < int(settings["epochs"]) and state["stale"] < int(
            settings["early_stopping_patience"]
        ):
            epoch = state["epoch"]
            order = np.random.default_rng(seed + epoch).permutation(len(train_rows))
            for start in range(
                state["position"], len(order), int(settings["effective_batch_size"])
            ):
                indices = order[start : start + int(settings["effective_batch_size"])]
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.0
                for index in indices:
                    loss, _ = backward_contract(
                        model,
                        encoded_train[index],
                        int(labels[index]),
                        weights[index],
                        len(indices),
                        int(settings["chunk_batch_size"]),
                    )
                    loss_sum += loss
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(norm):
                    raise ValueError("Nonfinite CodeBERT gradient")
                optimizer.step()
                state["optimizer_steps"] += 1
                state["position"] = int(start + len(indices))
                state["epoch_seen"] += len(indices)
                state["epoch_loss_sum"] += loss_sum
                if state["optimizer_steps"] % int(settings["checkpoint_every_steps"]) == 0:
                    checkpoint()
                    print(
                        json.dumps(
                            {
                                "stage": "codebert_training",
                                "seed": seed,
                                "epoch": epoch + 1,
                                "contracts_seen": state["epoch_seen"],
                                "optimizer_steps": state["optimizer_steps"],
                            }
                        ),
                        flush=True,
                    )
            if state["epoch_seen"] != len(train_rows):
                raise ValueError("CodeBERT epoch did not visit every training contract")
            logits = predict_contracts(model, encoded_validation, int(settings["chunk_batch_size"]))
            metrics = calculate_metrics(validation_labels, expit(logits), 0.5)
            value = float(metrics["log_loss"])
            if state["best"] is None or value < state["best"] - 1e-8:
                _atomic_torch(best_path, model.state_dict())
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
            state["position"], state["epoch_seen"], state["epoch_loss_sum"] = 0, 0, 0.0
            _atomic_json(result_dir / "history.json", state["history"])
            checkpoint()
            print(
                json.dumps(
                    {"stage": "codebert_epoch_complete", "seed": seed, **state["history"][-1]}
                ),
                flush=True,
            )
    except KeyboardInterrupt:
        optimizer.zero_grad(set_to_none=True)
        checkpoint()
        raise
    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
    logits = predict_contracts(model, encoded_validation, int(settings["chunk_batch_size"]))
    _atomic_numpy(result_dir / "validation_logits.npy", logits)
    summary = {
        "seed": seed,
        "status": "complete",
        "architecture": "codebert_finetuned",
        "input_mode": "full",
        "encoder_frozen": False,
        "encoder_revision": settings["revision"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "epochs_executed": len(state["history"]),
        "best_epoch": state["best_epoch"],
        "training_seconds": state["training_seconds"],
        "optimizer_steps": state["optimizer_steps"],
        "training_windows": sum(map(len, encoded_train)),
        "validation_windows": sum(map(len, encoded_validation)),
        "validation": calculate_metrics(validation_labels, expit(logits), 0.5),
        "model_path": best_path.resolve().relative_to(ROOT).as_posix(),
        "model_sha256": file_digest(best_path),
        "contract_pooling": "mean_of_all_window_CLS_logits",
        "gradient_method": "two_exact_passes_with_dropout_zero_and_contract_BCE",
        "input_fingerprint": fingerprint,
    }
    _atomic_json(result_dir / "run.json", summary)
    _atomic_json(
        result_dir / "progress.json",
        {
            **state,
            "status": "complete",
            "seed": seed,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return summary
