"""Frozen, revision-pinned CodeT5 features, covering every input subtoken."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")
import numpy as np
import torch
from .comparison_data import SPLITS, load_study_config, save_json
from .experiment_v3 import read_jsonl
from .preprocessing_v3 import ROOT, file_digest, digest


class FrozenCodeEncoder:
    def __init__(self, config):
        from huggingface_hub import HfApi
        from transformers import RobertaTokenizer, T5EncoderModel
        self.config = config["pretrained"]
        self.directory = ROOT / ".cache-comparison/codet5"
        self.directory.mkdir(parents=True, exist_ok=True)
        manifest_path = self.directory / "manifest.json"
        previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        revision_path = self.directory / "revision.json"
        if revision_path.exists():
            revision = json.loads(revision_path.read_text())["revision"]
        else:
            revision = HfApi().model_info(self.config["model_id"]).sha
            save_json(revision_path, {"model_id": self.config["model_id"], "revision": revision})
        kwargs = {"revision": revision, "cache_dir": str(ROOT / ".cache-comparison/huggingface")}
        self.tokenizer = RobertaTokenizer.from_pretrained(self.config["model_id"], **kwargs)
        self.model = T5EncoderModel.from_pretrained(self.config["model_id"], **kwargs).eval()
        self.model.requires_grad_(False)
        self.dimension = self.model.config.d_model
        self.capacity = self.config["chunk_subtokens"] - self.tokenizer.num_special_tokens_to_add(pair=False)
        manifest = {"model_id": self.config["model_id"], "revision": revision, "dataset_id": config["dataset_id"], "strategy": self.config["strategy"], "config": self.config, "encoder_parameters": sum(p.numel() for p in self.model.parameters()), "feature_dimension": self.dimension * 2, "input": "space-joined normalized V3 lexical tokens, first512 cut before BPE", "special_tokens_excluded_from_pooling": True, "no_trainable_encoder_parameters": True}
        if previous and previous != manifest:
            raise ValueError("Pretrained cache does not match this protocol")
        if not previous:
            save_json(manifest_path, manifest)
        self.manifest = manifest

    def chunks(self, lexical_tokens):
        text = " ".join(lexical_tokens)
        ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        if not ids:
            raise ValueError("Empty pretrained input")
        return [tuple(ids[i:i + self.capacity]) for i in range(0, len(ids), self.capacity)]

    def chunk_features(self, chunks):
        """Return sums/maxima/counts; pooling is invariant to batch padding."""
        inputs = [self.tokenizer.build_inputs_with_special_tokens(list(x)) for x in chunks]
        encoded = self.tokenizer.pad({"input_ids": inputs}, padding=True, return_tensors="pt")
        masks = torch.zeros_like(encoded["attention_mask"], dtype=torch.bool)
        for i, (chunk, ids) in enumerate(zip(chunks, inputs)):
            special = self.tokenizer.get_special_tokens_mask(list(chunk), already_has_special_tokens=False)
            masks[i, :len(ids)] = torch.as_tensor([not value for value in special])
        with torch.inference_mode():
            states = self.model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]).last_hidden_state
            sums = (states * masks.unsqueeze(-1)).sum(1).numpy()
            maxima = states.masked_fill(~masks.unsqueeze(-1), -torch.inf).amax(1).numpy()
        counts = masks.sum(1).numpy()
        return sums, maxima, counts

    def encode_rows(self, rows, modes):
        # Shared chunks are computed once across full/first512 for each contract.
        chunks, owners = [], []
        sums = np.zeros((len(rows), len(modes), self.dimension), dtype=np.float64)
        maxima = np.full_like(sums, -np.inf)
        counts = np.zeros((len(rows), len(modes)), dtype=np.int64)
        for row_index, row in enumerate(rows):
            local = {}
            for mode_index, mode in enumerate(modes):
                tokens = row["tokens"] if mode == "full" else row["tokens"][:512]
                for chunk in self.chunks(tokens):
                    if chunk not in local:
                        local[chunk] = len(chunks)
                        chunks.append(chunk)
                        owners.append([])
                    owners[local[chunk]].append((row_index, mode_index))
        batch_size = self.config["batch_size"]
        for start in range(0, len(chunks), batch_size):
            s, m, n = self.chunk_features(chunks[start:start + batch_size])
            for offset in range(len(s)):
                for row_index, mode_index in owners[start + offset]:
                    sums[row_index, mode_index] += s[offset]
                    maxima[row_index, mode_index] = np.maximum(maxima[row_index, mode_index], m[offset])
                    counts[row_index, mode_index] += n[offset]
        if np.any(counts <= 0):
            raise ValueError("Missing pretrained subtokens")
        features = np.concatenate((sums / counts[..., None], maxima), axis=-1).astype(np.float32)
        return features, counts, len(chunks)


def extract_split(encoder, config, split):
    directory = encoder.directory
    modes = config["pretrained"]["input_modes"]
    marker = directory / f"{split}_extraction.json"
    if marker.exists():
        done = json.loads(marker.read_text())
        for name, sha in done["sha256"].items():
            if file_digest(directory / name) != sha:
                raise ValueError(f"Corrupted feature cache: {name}")
        return
    rows = read_jsonl(ROOT / config["dataset_dir"] / (split + ".jsonl.gz"))
    state_path = directory / (split + "_progress.json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {"completed": 0, "seconds": 0.0, "computed_chunks": 0}
    maps = {mode: np.lib.format.open_memmap(directory / f"{split}_{mode}.partial.npy", mode="r+" if state_path.exists() else "w+", dtype=np.float32, shape=(len(rows), encoder.dimension * 2)) for mode in modes}
    token_counts = np.lib.format.open_memmap(directory / f"{split}_counts.partial.npy", mode="r+" if state_path.exists() else "w+", dtype=np.int64, shape=(len(rows), len(modes)))
    for start in range(state["completed"], len(rows), 32):
        clock = time.perf_counter()
        stop = min(start + 32, len(rows))
        features, counts, chunks = encoder.encode_rows(rows[start:stop], modes)
        for index, mode in enumerate(modes):
            maps[mode][start:stop] = features[:, index]
            maps[mode].flush()
        token_counts[start:stop] = counts
        token_counts.flush()
        state.update(completed=stop, seconds=state["seconds"] + time.perf_counter() - clock, computed_chunks=state["computed_chunks"] + chunks)
        save_json(state_path, state)
        print(json.dumps({"extraction": split, **state, "total": len(rows)}), flush=True)
    for mode in modes:
        maps[mode]._mmap.close()
        (directory / f"{split}_{mode}.partial.npy").replace(directory / f"{split}_{mode}.npy")
    token_counts._mmap.close()
    (directory / f"{split}_counts.partial.npy").replace(directory / f"{split}_counts.npy")
    names = [f"{split}_{mode}.npy" for mode in modes] + [f"{split}_counts.npy"]
    save_json(marker, {**state, "sample_ids_sha256": digest([row["sample_id"] for row in rows]), "manifest_sha256": file_digest(directory / "manifest.json"), "sha256": {name: file_digest(directory / name) for name in names}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=SPLITS)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    config = load_study_config(args.config)
    torch.set_num_threads(config["cpu_threads"])
    torch.set_num_interop_threads(1)
    encoder = FrozenCodeEncoder(config)
    print(json.dumps(encoder.manifest), flush=True)
    if not args.download_only:
        for split in args.splits:
            extract_split(encoder, config, split)


if __name__ == "__main__":
    main()
