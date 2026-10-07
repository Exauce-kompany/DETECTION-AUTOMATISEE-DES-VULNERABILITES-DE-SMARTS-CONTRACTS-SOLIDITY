import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.data.source_paths import resolve_source_path


class SourceRelocationTests(unittest.TestCase):
    def test_relocation_checks_fingerprint_and_preserves_logical_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "dataset/sources/train.json"
            source.parent.mkdir(parents=True)
            source.write_bytes(b'[{"context":"contract C {}"}]')
            original = "dataset/v2/raw/train.json"
            manifest = {
                "files": [
                    {
                        "original_path": original,
                        "current_path": "dataset/sources/train.json",
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    }
                ]
            }
            (source.parent / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(resolve_source_path(original.replace("/", "\\"), root), source)
            source.write_bytes(b"[]")
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                resolve_source_path(original, root)

    def test_explicit_existing_sources_work_and_missing_sources_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "train.json").write_text("[]")
            self.assertEqual(resolve_source_path("train.json", root), root / "train.json")
            with self.assertRaises(FileNotFoundError):
                resolve_source_path("missing.json", root)
            with self.assertRaises(ValueError):
                resolve_source_path("../outside.json", root)
