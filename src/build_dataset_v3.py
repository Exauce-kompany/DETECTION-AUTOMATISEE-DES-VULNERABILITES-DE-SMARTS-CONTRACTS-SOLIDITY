"""Build a quarantined, grouped benchmark without altering the V1/V2 evidence."""
import argparse
from collections import Counter, defaultdict
import gzip
import itertools
import json
from pathlib import Path
import re

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

try:
    from .preprocessing_v3 import ROOT, VERSION, build_vocabulary, code_from_sample, digest, encode, file_digest, load_config, structural_tokens, tokenize
except ImportError:
    from preprocessing_v3 import ROOT, VERSION, build_vocabulary, code_from_sample, digest, encode, file_digest, load_config, structural_tokens, tokenize


class UnionFind:
    def __init__(self, size):
        self.parents = list(range(size))

    def find(self, item):
        while self.parents[item] != item:
            self.parents[item] = self.parents[self.parents[item]]
            item = self.parents[item]
        return item

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parents[max(a, b)] = min(a, b)


def source_name(sample):
    if sample.get("source_dataset"):
        return sample["source_dataset"]
    label = str(sample.get("label", "unknown"))
    return "zeus" if label.startswith("zeus_") else label


def lineage_key(sample, source):
    path = str(sample.get("source_path") or sample.get("contract") or "").replace("\\", "/")
    if not path:
        return None
    if source == "dappscan":
        parts = path.split("/")
        return "project:" + "/".join(parts[:3]).lower().replace(" ", "_")
    if source == "solidifi_benchmark":
        return "solidifi:" + Path(path).stem.lower()
    stem = Path(path).stem.lower()
    address = re.search(r"(?:0x)?([0-9a-f]{40})(?:_ext)?$", stem)
    return "address:" + address.group(1) if address else source + ":" + path.lower()


def normalize_sample(sample, split, index, config):
    label = sample.get("has_vulnerability")
    if not isinstance(label, (int, bool)) or label not in (0, 1):
        raise ValueError("invalid_binary_label")
    code = code_from_sample(sample)
    raw_tokens = tokenize(code, normalize_identifiers=False)
    tokens = tokenize(code)
    if not tokens:
        raise ValueError("empty_source_after_comment_removal")
    source = source_name(sample)
    granularity = sample.get("granularity") or "contract_inferred"
    if granularity == "function":
        raise ValueError("function_without_parent_contract_context")
    metadata = sample.get("metadata") or {}
    assessed = sorted(filter(None, str(metadata.get("assessed_types", "")).split("|")))
    if source == "CGT" and label == 0 and not set(config["target_types"]).issubset(assessed):
        raise ValueError("negative_with_incomplete_target_coverage")
    canonical = digest(raw_tokens)
    record = {
        "sample_id": f"{source}:{canonical}",
        "source": source, "sources": [source], "label": int(label),
        "granularity": granularity,
        "label_scope": "no_annotation_in_source_scope" if label == 0 else "positive_source_annotation",
        "label_origin": sample.get("label_origin") or str(sample.get("label", "legacy")),
        "label_confidence": sample.get("label_confidence", "unverified_legacy"),
        "assessed_types": assessed,
        "vulnerability_types": sorted({v.get("type", "unknown") if isinstance(v, dict) else str(v) for v in sample.get("vulnerabilities", [])}),
        "source_refs": [{"split": split, "index": index, "original_id": sample.get("sample_id"), "contract": sample.get("contract")}],
        "raw_sha256": digest(code.encode("utf-8")),
        "canonical_sha256": canonical, "feature_sha256": digest(tokens),
        "structural_sha256": digest(structural_tokens(tokens)),
        "lineage_keys": [key] if (key := lineage_key(sample, source)) else [],
        "tokens": tokens,
    }
    return record


def quarantine_conflicts_and_deduplicate(records):
    canonical = defaultdict(list)
    features = defaultdict(list)
    for i, record in enumerate(records):
        canonical[record["canonical_sha256"]].append(i)
        features[record["feature_sha256"]].append(i)
    rejected = set()
    for groups in (canonical, features):
        for indices in groups.values():
            if len({records[i]["label"] for i in indices}) > 1:
                rejected.update(indices)
    quarantine = [{"reason": "contradictory_labels_for_identical_code_or_model_representation", **{k: v for k, v in records[i].items() if k != "tokens"}} for i in sorted(rejected)]
    kept, duplicates = [], []
    for indices in features.values():
        indices = [i for i in indices if i not in rejected]
        if not indices:
            continue
        representative = dict(records[indices[0]])
        representative["sources"] = sorted({s for i in indices for s in records[i]["sources"]})
        representative["source_refs"] = [ref for i in indices for ref in records[i]["source_refs"]]
        representative["lineage_keys"] = sorted({k for i in indices for k in records[i]["lineage_keys"]})
        kept.append(representative)
        for i in indices[1:]:
            duplicates.append({"kept_id": representative["sample_id"], "removed_ref": records[i]["source_refs"], "removed_canonical_sha256": records[i]["canonical_sha256"], "reason": "identical_clean_model_tokens_with_consistent_label"})
    return kept, quarantine, duplicates


def assign_groups(records):
    union = UnionFind(len(records))
    first = {}
    for i, record in enumerate(records):
        keys = ["feature:" + record["feature_sha256"], "structure:" + record["structural_sha256"]]
        keys += ["lineage:" + key for key in record["lineage_keys"]]
        for key in keys:
            if key in first:
                union.union(i, first[key])
            else:
                first[key] = i
    members = defaultdict(list)
    for i in range(len(records)):
        members[union.find(i)].append(i)
    for indices in members.values():
        group_id = digest(sorted(records[i]["canonical_sha256"] for i in indices))
        for i in indices:
            records[i]["group_id"] = group_id


def split_records(records, config):
    holdout_groups = {r["group_id"] for r in records if config["holdout_source"] in r["sources"]}
    splits = {name: [] for name in ("train", "validation", "calibration", "test", "source_holdout")}
    pool = []
    for record in records:
        (splits["source_holdout"] if record["group_id"] in holdout_groups else pool).append(record)
    pool.sort(key=lambda r: r["sample_id"])
    strata = [r["source"] + ":" + str(r["label"]) for r in pool]
    counts = Counter(strata)
    common = {label: max((s for s in counts if s.endswith(":" + str(label))), key=counts.get) for label in (0, 1)}
    strata = [s if counts[s] >= config["folds"] else common[r["label"]] for s, r in zip(strata, pool)]
    cv = StratifiedGroupKFold(n_splits=config["folds"], shuffle=True, random_state=config["split_seed"])
    for fold, (_, selected) in enumerate(cv.split(np.zeros(len(pool)), strata, [r["group_id"] for r in pool])):
        name = {0: "test", 1: "validation", 2: "calibration"}.get(fold, "train")
        splits[name].extend(pool[i] for i in selected)
    for name, rows in splits.items():
        rows.sort(key=lambda r: r["sample_id"])
        if name != "source_holdout" and {r["label"] for r in rows} != {0, 1}:
            raise ValueError(f"Partition {name} does not contain both classes; review grouped split configuration.")
    return splits


def audit_splits(splits, vocabulary):
    indexes, summary = {}, {}
    for name, rows in splits.items():
        kinds = {kind: defaultdict(set) for kind in ("sample_id", "group_id", "canonical_sha256", "feature_sha256", "structural_sha256", "encoded_sha256")}
        for row in rows:
            row["encoded_sha256"] = digest(encode(row["tokens"], vocabulary).astype("<i4").tobytes())
            for kind, index in kinds.items():
                index[row[kind]].add(row["label"])
        conflicts = {kind: sum(len(labels) > 1 for labels in index.values()) for kind, index in kinds.items() if kind not in ("group_id", "structural_sha256")}
        if any(conflicts.values()):
            raise ValueError(f"Conflicting effective representations in {name}: {conflicts}")
        indexes[name] = kinds
        summary[name] = {
            "samples": len(rows), "groups": len(kinds["group_id"]),
            "labels": dict(Counter(str(r["label"]) for r in rows)),
            "sources": dict(Counter(r["source"] for r in rows)),
            "source_labels": {source: dict(Counter(str(r["label"]) for r in rows if r["source"] == source)) for source in sorted({r["source"] for r in rows})},
            "granularities": dict(Counter(r["granularity"] for r in rows)),
            "conflicting_groups": conflicts, "tokens": sum(len(r["tokens"]) for r in rows),
        }
    overlaps = {}
    for a, b in itertools.combinations(splits, 2):
        overlaps[a + ":" + b] = {kind: len(indexes[a][kind].keys() & indexes[b][kind].keys()) for kind in indexes[a]}
        if any(overlaps[a + ":" + b].values()):
            raise ValueError(f"Partitions overlap: {a}/{b}: {overlaps[a + ':' + b]}")
    return {"passed": True, "splits": summary, "overlaps": overlaps, "limitations": ["Shared library fragments can occur across otherwise distinct complete inputs.", "Grouping does not prove absence of every approximate clone.", "Labels describe source annotations; negative labels do not certify security."]}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            for row in rows:
                stream.write((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))


def build(config, output=None):
    destination = Path(output) if output else ROOT / config["dataset_dir"]
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to replace dataset artifacts: {destination}. Use a new --output directory.")
    records, quarantine, source_hashes = [], [], {}
    for split in ("train", "val", "test"):
        path = ROOT / config["input_dir"] / (split + ".json")
        source_hashes[str(path.relative_to(ROOT))] = file_digest(path)
        samples = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(samples, list):
            raise ValueError(f"Invalid input partition: {path}")
        for index, sample in enumerate(samples):
            try:
                records.append(normalize_sample(sample, split, index, config))
            except ValueError as exc:
                quarantine.append({"reason": str(exc), "source_split": split, "source_index": index, "original_id": sample.get("sample_id"), "source": source_name(sample), "label": sample.get("has_vulnerability")})
        print(f"Loaded {split}: {len(samples)} source records", flush=True)
        del samples
    records, conflicts, duplicates = quarantine_conflicts_and_deduplicate(records)
    quarantine.extend(conflicts)
    assign_groups(records)
    splits = split_records(records, config)
    vocabulary = build_vocabulary((r["tokens"] for r in splits["train"]), config["vocabulary_size"])
    audit = audit_splits(splits, vocabulary)
    audit["quarantine"] = dict(Counter(r["reason"] for r in quarantine))
    audit["removed_duplicate_samples"] = len(duplicates)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "config.json", config)
    write_json(destination / "vocabulary.json", vocabulary)
    write_json(destination / "audit.json", audit)
    write_jsonl(destination / "quarantine.jsonl.gz", quarantine)
    write_jsonl(destination / "duplicates.jsonl.gz", duplicates)
    for name, rows in splits.items():
        sequences = [encode(row["tokens"], vocabulary) for row in rows]
        offsets = np.cumsum([0] + [len(ids) for ids in sequences], dtype=np.int64)
        np.savez_compressed(destination / (name + ".npz"), tokens=np.concatenate(sequences) if sequences else np.array([], dtype=np.int32), offsets=offsets, labels=np.asarray([r["label"] for r in rows], dtype=np.int32))
        write_jsonl(destination / (name + ".jsonl.gz"), rows)
        print(f"{name}: {len(rows)} records / {audit['splits'][name]['groups']} independent groups", flush=True)
    files = {p.name: file_digest(p) for p in sorted(destination.iterdir()) if p.is_file()}
    manifest = {"version": "v3", "preprocessing_version": VERSION, "source_files": source_hashes, "files": files, "config": config, "builder_sha256": file_digest(__file__), "preprocessor_sha256": file_digest(Path(__file__).with_name('preprocessing_v3.py'))}
    manifest["dataset_id"] = digest(manifest)
    write_json(destination / "manifest.json", manifest)
    print(json.dumps({"dataset_id": manifest["dataset_id"], "quarantine": audit["quarantine"], "duplicates_removed": len(duplicates)}, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    build(load_config(args.config), args.output)
