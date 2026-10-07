"""Resolve relocated raw sources without rewriting frozen experiment manifests."""

import hashlib
import json
from pathlib import Path

from src.preprocessing import ROOT


def resolve_source_path(relative_path, root=ROOT):
    root = Path(root).resolve()
    logical = str(relative_path).replace("\\", "/")
    path = (root / logical).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Source path outside the project")
    if path.is_file():
        return path
    relocation = root / "dataset/sources/manifest.json"
    if relocation.is_file():
        for entry in json.loads(relocation.read_text(encoding="utf-8"))["files"]:
            if entry["original_path"] != logical:
                continue
            destination = (root / entry["current_path"]).resolve()
            if not destination.is_relative_to(root):
                raise ValueError("Relocated source outside the project")
            with destination.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != entry["sha256"]:
                raise ValueError(f"Relocated source fingerprint mismatch: {logical}")
            return destination
    raise FileNotFoundError(path)
