"""Prevent leakage and fabricated rankings in the three-model comparison."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.run_model_benchmark import import_baselines, load_config, paired_comparisons, save_json, select_seed, vectorizer, write_report
from src.preprocessing import ROOT


class ModelBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "config/benchmark_models.json")

    def test_validation_tokens_never_enter_tfidf_vocabulary(self):
        self.config["tfidf"]["min_df"] = 1
        extractor = vectorizer(self.config)
        train = extractor.fit_transform(["contract id_train transfer", "contract id_train call"])
        before = dict(extractor.vocabulary_)
        validation = extractor.transform(["contract token_only_in_validation"])
        self.assertEqual(before, extractor.vocabulary_)
        self.assertNotIn("token_only_in_validation", before)
        self.assertEqual(train.shape[1], validation.shape[1])

    def test_selection_ignores_higher_test_score(self):
        runs = [{"seed": 42, "validation": {"log_loss": 0.4}, "test_f1": 0.99},
                {"seed": 73, "validation": {"log_loss": 0.3}, "test_f1": 0.60}]
        self.assertEqual(select_seed(runs)["seed"], 73)

    def test_failed_json_serialization_preserves_previous_completion_marker(self):
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / "selection.json"
            save_json(marker, {"selected_seed": 73})
            with self.assertRaises(ValueError):
                save_json(marker, {"selected_seed": float("nan")})
            self.assertEqual(json.loads(marker.read_text(encoding="utf-8")), {"selected_seed": 73})

    def test_pending_models_have_no_fake_metrics_or_final_ranking(self):
        with tempfile.TemporaryDirectory() as folder:
            result = Path(folder)
            summary = write_report(self.config, result)
            self.assertFalse(summary["complete"])
            self.assertEqual(set(summary["models"]), {"cnn_bilstm", "xgboost", "tcn"})
            for run in summary["models"].values():
                self.assertIsNone(run["metrics"])
            self.assertIn("aucun classement final", (result / "report.md").read_text(encoding="utf-8"))

    def test_import_rejects_different_feature_protocol_before_copying_results(self):
        with tempfile.TemporaryDirectory() as folder:
            source, destination = Path(folder) / "old", Path(folder) / "new"
            previous = json.loads(json.dumps(self.config))
            previous["tfidf"]["max_features"] = 500
            save_json(source / "protocol.json", {"config": previous})
            with self.assertRaisesRegex(ValueError, "do not match"):
                import_baselines(self.config, destination, source)
            self.assertFalse(destination.exists())

    def test_paired_comparison_rejects_different_contract_order(self):
        with tempfile.TemporaryDirectory() as folder:
            result = Path(folder)
            save_json(result / "tcn/evaluation.json", {})
            save_json(result / "tcn/test_sample_ids.json", ["wrong-contract"])
            rows = [{"sample_id": "correct-contract", "group_id": "family", "label": 1}]
            with patch("src.run_model_benchmark.records", return_value=rows):
                with self.assertRaisesRegex(ValueError, "same test order"):
                    paired_comparisons(self.config, result, {"tcn": {}})
            self.assertFalse((result / "paired_comparisons.json").exists())


if __name__ == "__main__":
    unittest.main()
