import tempfile
from pathlib import Path
import unittest
import numpy as np
from scipy.special import expit
try:
    import torch
except ModuleNotFoundError:
    raise unittest.SkipTest("Optional comparison dependencies are not installed")

from src.comparison_data import batches, collate, load_study_config, pack_sequences
from src.comparison_metrics import bootstrap_delta, evaluate_logits, fit_operating_points
from src.comparison_models import ContractClassifier
from src.comparison_pretrained import FrozenCodeEncoder
from src.compare_models import fit_one, predict_neural


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = load_study_config()
        cls.config.update(window_size=16, embedding_dim=8, convolution_filters=4, lstm_units=4, dense_units=8, batch_size=3, batch_token_budget=50)

    def test_batches_cover_every_contract_without_truncation(self):
        lengths = [3, 2, 999, 4, 31, 300, 9, 8]
        arrays = pack_sequences([np.full(n, i + 1) for i, n in enumerate(lengths)], np.arange(len(lengths)) % 2)
        for seed in (42, 73):
            seen = []
            for indices in batches(lengths, self.config, seed):
                x, n, labels = collate(arrays, indices)
                seen.extend(indices)
                for row, index in enumerate(indices):
                    self.assertEqual(n[row], lengths[index])
                    self.assertEqual(int(x[row].count_nonzero()), lengths[index])
            self.assertEqual(sorted(seen), list(range(len(lengths))))

    def test_padding_and_checkpoint_invariance_for_all_architectures(self):
        ids = torch.tensor([[2, 3, 4, 5, 6], [8, 9, 0, 0, 0]])
        lengths = torch.tensor([5, 2])
        for architecture in self.config["architectures"]:
            with self.subTest(architecture=architecture):
                torch.manual_seed(4)
                model = ContractClassifier(architecture, 300, self.config).eval()
                with torch.inference_mode():
                    expected = model(ids, lengths)
                    padded = model(torch.nn.functional.pad(ids, (0, 33)), lengths)
                np.testing.assert_allclose(expected, padded, atol=1e-6)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "model.pt"
                    torch.save(model.state_dict(), path)
                    restored = ContractClassifier(architecture, 300, self.config).eval()
                    restored.load_state_dict(torch.load(path, weights_only=True))
                    with torch.inference_mode():
                        np.testing.assert_allclose(restored(ids, lengths), expected, atol=1e-7)

    def test_tail_beyond_512_receives_gradient(self):
        for architecture in self.config["architectures"]:
            with self.subTest(architecture=architecture):
                torch.manual_seed(42)
                model = ContractClassifier(architecture, 300, self.config).eval()
                ids = torch.full((1, 600), 2)
                ids[0, -1] = 299
                model(ids, torch.tensor([600])).sum().backward()
                self.assertGreater(float(model.embedding.weight.grad[299].abs().sum()), 0)

    def test_calibration_fpr_is_conservative_in_presence_of_ties(self):
        labels = np.array([0] * 20 + [1] * 20)
        logits = np.array([0.] * 20 + [1.] * 20)
        fit = fit_operating_points(labels, logits, self.config)
        p = expit(logits / fit["temperature"])
        self.assertLessEqual(float(np.mean(p[labels == 0] >= fit["fpr_threshold"])), self.config["target_fpr"])
        self.assertEqual(fit["fit_split"], "calibration")
        self.assertIsNone(evaluate_logits([1, 1], np.array([0., 1.]), fit)["roc_recall_at_fpr_0_10"])

    def test_paired_bootstrap_identical_predictions_have_zero_difference(self):
        probabilities = [np.array([.1, .2, .8, .9])]
        result = bootstrap_delta([0, 0, 1, 1], ["a", "a", "b", "c"], probabilities, [.5], probabilities, [.5], 30)
        for score in result.values():
            self.assertEqual(score, {"difference": 0.0, "ci95": [0.0, 0.0]})

    def test_pretrained_all_chunks_counted_even_when_repeated(self):
        encoder = object.__new__(FrozenCodeEncoder)
        encoder.dimension = 1
        encoder.config = {"batch_size": 2}
        encoder.chunks = lambda tokens: [tuple(tokens[i:i + 100]) for i in range(0, len(tokens), 100)]
        def features(chunks):
            return (np.array([[sum(c)] for c in chunks]), np.array([[max(c)] for c in chunks]), np.array([len(c) for c in chunks]))
        encoder.chunk_features = features
        rows = [{"tokens": [1] * 512 + [9] * 88}, {"tokens": [3, 7]}]
        values, counts, computed = encoder.encode_rows(rows, ["full", "first512"])
        np.testing.assert_array_equal(counts, [[600, 512], [2, 2]])
        np.testing.assert_allclose(values[0, 0], [(512 + 88 * 9) / 600, 9])
        np.testing.assert_allclose(values[0, 1], [1, 1])
        np.testing.assert_allclose(values[1], [[5, 7], [5, 7]])

    def test_training_restores_best_checkpoint_and_prediction_order(self):
        config = {**self.config, "epochs": 2}
        arrays = pack_sequences([np.full(n, 2 + i % 2) for i, n in enumerate([17, 3, 80, 7, 33, 5])], [0, 1, 0, 1, 0, 1])
        model = ContractClassifier("cnn", 300, config)
        # Use a project-local temporary directory: model paths in provenance are
        # deliberately relative to the repository.
        from src.preprocessing_v3 import ROOT
        (ROOT / ".cache-comparison").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache-comparison") as directory:
            path = Path(directory)
            summary = fit_one(model, arrays, arrays, np.ones(6, dtype=np.float32), config, 42, path, path / "model.pt")
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["epochs_executed"], 2)
            batched = predict_neural(model, arrays, config)
            single = []
            with torch.inference_mode():
                for index in range(6):
                    x, n, _ = collate(arrays, np.array([index]))
                    single.append(float(model(x, n)[0]))
            np.testing.assert_allclose(batched, single, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
