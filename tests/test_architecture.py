"""Protect module boundaries and frozen experiment provenance after relocation."""

import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.provenance import ROOT, SOURCE_FILES, source_path


def imports(path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


class ModuleBoundaryTests(unittest.TestCase):
    def test_scientific_modules_do_not_depend_on_the_web_application(self):
        for path in (ROOT / "src").rglob("*.py"):
            with self.subTest(module=path.relative_to(ROOT)):
                dependencies = {name.split(".")[0] for name in imports(path)}
                self.assertTrue(dependencies.isdisjoint({"webapp", "fastapi", "flask"}))

    def test_dataset_and_metric_modules_do_not_depend_on_experiment_runners(self):
        for directory in ("data", "evaluation"):
            for path in (ROOT / "src" / directory).glob("*.py"):
                with self.subTest(module=path.relative_to(ROOT)):
                    self.assertFalse(
                        any(
                            name.startswith(("src.experiments", "src.training"))
                            for name in imports(path)
                        )
                    )

    def test_shared_imports_do_not_load_model_frameworks(self):
        code = """
import importlib
import sys
for name in (
    'src.data.artifacts', 'src.data.batching', 'src.data.source_paths',
    'src.evaluation.metrics', 'src.evaluation.comparison_metrics',
    'src.training.datasets', 'src.inference.predictor',
):
    importlib.import_module(name)
assert {'tensorflow', 'torch', 'transformers', 'fastapi', 'matplotlib'}.isdisjoint(sys.modules)
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class SourceProvenanceTests(unittest.TestCase):
    def test_registered_sources_resolve_to_existing_project_files(self):
        for name, relative in SOURCE_FILES.items():
            with self.subTest(source=name):
                path = source_path(name).resolve()
                self.assertTrue(path.is_relative_to(ROOT / "src"))
                self.assertTrue(path.is_file())
                self.assertEqual(path.name, name)
                self.assertEqual(path, ROOT / relative)
        with self.assertRaises(KeyError):
            source_path("../preprocessing.py")

    def test_benchmark_resume_rejects_changed_shared_source(self):
        from src.experiments.benchmark import run_model_benchmark as benchmark

        config = {
            "dataset_dir": "dataset/processed",
            "dataset_id": "fixture",
            "models": ["tcn"],
            "cnn_bilstm": {"manifest": "models/active.json"},
        }
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / config["cnn_bilstm"]["manifest"]
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}", encoding="utf-8")
            with (
                patch.object(benchmark, "ROOT", root),
                patch.object(benchmark, "verify_dataset", return_value={"dataset_id": "fixture"}),
                patch.object(benchmark.metadata, "version", return_value="fixture"),
            ):
                result, _ = benchmark.initialize(config, "new-study")
                benchmark.initialize(config, "new-study")
                protocol = json.loads((result / "protocol.json").read_text(encoding="utf-8"))
                for name in (
                    "artifacts.py",
                    "batching.py",
                    "datasets.py",
                    "metrics.py",
                    "provenance.py",
                ):
                    with self.subTest(source=name):
                        self.assertIn(name, protocol["sources_sha256"])
                        original_digest = benchmark.file_digest

                        def changed_digest(path):
                            return (
                                "changed"
                                if Path(path) == source_path(name)
                                else original_digest(path)
                            )

                        with patch.object(benchmark, "file_digest", side_effect=changed_digest):
                            with self.assertRaisesRegex(ValueError, "choose a new run-id"):
                                benchmark.initialize(config, "new-study")
                self.assertEqual(
                    json.loads((result / "protocol.json").read_text(encoding="utf-8")), protocol
                )


if __name__ == "__main__":
    unittest.main()
