"""Verify that partial plots preserve evaluation and calibration boundaries."""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.plot_model_benchmark import MODELS, METRICS, completion_gate, evaluated_models, load_complete_test


class AvailableModelPlotTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.result = self.root / "results"
        self.config = {"dataset_dir": "dataset", "dataset_id": "test-snapshot"}
        self.summary = {"complete": False, "models": {"tcn": {"status": "paused_by_user", "metrics": None}}}
        for name in MODELS[:2]:
            selected = {"seed": 42}
            point = {key: 0.5 for key in METRICS}
            point.update({"threshold": 0.5, "samples": 2, "confusion_matrix": [[1, 0], [0, 1]]})
            calibration = {"fit_split": "calibration", "temperature": 1.0, "threshold": 0.5}
            self.summary["models"][name] = {"status": "evaluated", "selected": selected, "metrics": point}
            self.save(self.result / name / "evaluation.json", {"model": name, "selected": selected,
                      "calibration": calibration, "splits": {"test": {"calibrated": point}}})
            self.save(self.result / name / "selection.json", {"fit_split": "validation", "selected": selected})
            self.save(self.result / name / "calibration.json", calibration)
            self.save(self.result / name / "test_sample_ids.json", ["first", "second"])
            np.save(self.result / name / "test_logits.npy", np.asarray([-1.0, 1.0]))
        compressed = gzip.compress(b'{"sample_id":"first","label":0}\n{"sample_id":"second","label":1}\n')
        dataset = self.root / "dataset"
        dataset.mkdir()
        (dataset / "test.jsonl.gz").write_bytes(compressed)
        self.save(dataset / "manifest.json", {"dataset_id": self.config["dataset_id"],
                  "files": {"test.jsonl.gz": hashlib.sha256(compressed).hexdigest()}})

    @staticmethod
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_default_still_requires_all_three_evaluations(self):
        self.assertTrue(completion_gate(self.summary, self.result))
        models = evaluated_models(self.summary)
        self.assertEqual(models, MODELS[:2])
        self.assertEqual(completion_gate(self.summary, self.result, models, require_complete=False), [])

    def test_eligibility_does_not_select_the_highest_test_score(self):
        self.summary["models"]["tcn"]["metrics"] = {"f1_macro": 1.0}
        self.assertEqual(evaluated_models(self.summary), MODELS[:2])

    def test_declared_evaluated_model_with_missing_calibration_is_rejected(self):
        (self.result / "xgboost/calibration.json").unlink()
        reasons = completion_gate(self.summary, self.result, evaluated_models(self.summary), require_complete=False)
        self.assertTrue(any("xgboost/calibration.json" in reason for reason in reasons))

    def test_calibration_mismatch_is_rejected_before_test_is_opened(self):
        self.save(self.result / "xgboost/calibration.json", {"fit_split": "test", "temperature": 1.0, "threshold": 0.5})
        (self.root / "dataset/test.jsonl.gz").unlink()
        with patch("src.plot_model_benchmark.ROOT", self.root):
            with self.assertRaisesRegex(ValueError, "Calibration incohérente"):
                load_complete_test(self.config, self.result, self.summary, {}, {}, MODELS[:2])

    def test_permuted_test_ids_are_rejected_despite_same_sample_count(self):
        self.save(self.result / "cnn_bilstm/test_sample_ids.json", ["second", "first"])
        with patch("src.plot_model_benchmark.ROOT", self.root):
            with self.assertRaisesRegex(ValueError, "Ordre des identifiants"):
                load_complete_test(self.config, self.result, self.summary, {}, {}, MODELS[:2])


if __name__ == "__main__":
    unittest.main()
