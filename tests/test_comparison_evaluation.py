import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

try:
    import torch
except ModuleNotFoundError:
    raise unittest.SkipTest("Optional comparison dependencies are not installed")

from src.comparison_data import load_study_config, save_json
from src.comparison_metrics import evaluate_logits, fit_operating_points
from src.evaluate_comparison import aggregate, evaluate_run, write_report
from src.plot_comparison import plot_comparison


class EvaluationTests(unittest.TestCase):
    def test_calibration_is_saved_before_any_test_predictions(self):
        config = load_study_config()
        labels = np.array([0, 1, 0, 1])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            destination = root / "cnn_full/seed-42"
            calls = []
            for split in ("test", "source_holdout"):
                save_json(cache / (split + "_metadata.json"), [{"label": int(label), "sample_id": str(i), "group_id": str(i), "source": "toy"} for i, label in enumerate(labels)])
            run = {"variant": "cnn_full", "architecture": "cnn", "input_mode": "full", "seed": 42, "model_sha256": "toy-checkpoint"}
            save_json(destination / "run.json", run)
            class ToyPredictor:
                def __init__(self, *args):
                    pass
                def predict(self, split):
                    calls.append(split)
                    if split != "calibration":
                        assert (destination / "calibration.json").exists()
                    return np.array([-2., 1., -.5, 3.])
            inputs = {"cross_split_truncated_collisions": {split: {"affected_indices": [0]} for split in ("test", "source_holdout")}}
            with patch("src.evaluate_comparison.Predictor", ToyPredictor), patch("src.evaluate_comparison.study_arrays", return_value={"labels": labels}):
                evaluate_run(destination / "run.json", config, cache, inputs)
                self.assertEqual(calls, ["calibration", "test", "source_holdout"])
                # Resume must reuse frozen calibration/evaluation, not refit.
                calls.clear()
                evaluate_run(destination / "run.json", config, cache, inputs)
                self.assertEqual(calls, [])
            result = json.loads((destination / "evaluation.json").read_text())
            self.assertEqual(result["splits"]["test"]["collision_free"]["samples"], 3)
            with np.load(destination / "test_predictions.npz") as predictions:
                np.testing.assert_array_equal(predictions["sample_ids"], ["0", "1", "2", "3"])

    def test_aggregation_report_and_figure_cover_all_variants(self):
        config = load_study_config()
        config["bootstrap_repetitions"] = 10
        labels = np.array([0, 1, 0, 1])
        logits = np.array([-2., 1., -.5, 3.])
        calibration = fit_operating_points(labels, logits, config)
        metrics = evaluate_logits(labels, logits, calibration)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            save_json(cache / "test_metadata.json", [{"label": int(label), "sample_id": str(i), "group_id": str(i)} for i, label in enumerate(labels)])
            candidates = []
            for architecture in config["architectures"] + ["codet5_frozen", "tfidf"]:
                for mode in config["input_modes"]:
                    variant = architecture + "_" + mode
                    directory = root / variant if architecture == "tfidf" else root / variant / "seed-42"
                    save_json(directory / "run.json", {"parameters": 30, "trainable_parameters": 20, "training_seconds": 1020.})
                    save_json(directory / "history.json", [{"elapsed_seconds": n} for n in (10., 20., 1020.)])
                    save_json(directory / "calibration.json", calibration)
                    save_json(directory / "evaluation.json", {"splits": {"test": {"all": metrics, "collision_free": metrics}}})
                    save_json(root / variant / "latency.json", {"median_ms": 2., "p95_ms": 4.})
                    np.savez(directory / "test_predictions.npz", probabilities=np.array([.1, .8, .3, .9]))
                    candidates.append({"variant": variant, "runs": [str((directory / "run.json").relative_to(root))]})
            selection = {"selected_variant": "cnn_full", "candidates": candidates}
            rows, comparisons = aggregate(config, root, selection, cache)
            self.assertEqual(len(rows), 10)
            self.assertTrue(all(row["timing_has_large_pause"] for row in rows))
            self.assertTrue(all(row["epoch_wall_seconds_median"] == 10. for row in rows))
            inputs = {"splits": {"test": {"samples": 4}}, "cross_split_truncated_collisions": {"test": {"affected_samples": 1}}}
            write_report(config, root, selection, rows, comparisons, inputs)
            plot_comparison(rows, root)
            for name in ("summary.csv", "summary.json", "paired_comparisons.json", "report.md", "comparison.png", "comparison.svg"):
                self.assertGreater((root / name).stat().st_size, 100)


if __name__ == "__main__":
    unittest.main()
