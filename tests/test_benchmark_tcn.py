"""TCN temporal semantics, contract-level loss, provenance and exact resume."""
import copy
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

from src import benchmark_tcn as benchmark
from src.comparison_data import batches, collate, pack_sequences
from src.preprocessing import ROOT, file_digest


class TCNBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        (ROOT / ".cache-comparison").mkdir(exist_ok=True)

    def _settings(self, **updates):
        settings = {"embedding_dim": 6, "channels": 4, "kernel_size": 3,
                    "dilations": [1, 2, 4], "convolutions_per_block": 2,
                    "dropout": 0.2, "dense_units": 5, "batch_size": 2,
                    "batch_token_budget": 80, "epochs": 3, "early_stopping_patience": 3,
                    "learning_rate": 0.01, "weight_decay": 0.001, "gradient_clip": 1.0,
                    "checkpoint_every_batches": 1, "cpu_threads": 1}
        return {**settings, **updates}

    def _config_and_arrays(self):
        config = {"dataset_id": "test-tcn-dataset", "seeds": [42, 73, 101], "tcn": self._settings()}
        sequences = [np.asarray([2 + i % 2, 4 + i] * (2 + i), dtype=np.int32) for i in range(6)]
        arrays = pack_sequences(sequences, [i % 2 for i in range(6)])
        return config, arrays

    def test_padding_batch_composition_and_prediction_order_invariance(self):
        settings = self._settings()
        torch.manual_seed(42)
        model = benchmark.load_tcn(settings, 300).double().eval()
        sequences = [np.array([2, 3, 4, 5, 6]), np.array([7, 8]), np.arange(10, 49)]
        arrays = pack_sequences(sequences, [0, 1, 0])
        ids, lengths, _ = collate(arrays, np.arange(3))
        with torch.inference_mode():
            batched = model(ids, lengths)
            extra_padding = model(torch.nn.functional.pad(ids, (0, 67)), lengths)
            alone = torch.stack([model(torch.as_tensor(row[None, :]), torch.tensor([len(row)]))[0] for row in sequences])
        torch.testing.assert_close(extra_padding, batched, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(alone, batched, atol=1e-12, rtol=1e-12)
        logits = benchmark.predict_tcn(model, arrays, settings)
        np.testing.assert_allclose(logits, alone.numpy(), atol=1e-12, rtol=1e-12)
        single_batch = benchmark.predict_tcn(model, arrays, {**settings, "batch_size": 8, "batch_token_budget": 1000})
        np.testing.assert_allclose(logits, single_batch, atol=1e-12, rtol=1e-12)
        values, mask = model.token_features(torch.nn.functional.pad(ids, (0, 11)), lengths)
        self.assertEqual(float(values.detach().masked_select(~mask.expand_as(values)).abs().sum()), 0)
        self.assertIsInstance(model.blocks[0].projection, torch.nn.Conv1d)
        self.assertEqual(model.blocks[0].projection.kernel_size, (1,))

    def test_causal_convolution_residual_and_full_tcn_cannot_read_future(self):
        torch.manual_seed(17)
        original = torch.randn(2, 6, 37, dtype=torch.float64)
        changed = original.clone()
        changed[:, :, 19:] += 100
        convolution = benchmark.CausalConv1d(6, 4, 3, 4).double()
        torch.testing.assert_close(convolution(original)[:, :, :19], convolution(changed)[:, :, :19], atol=0, rtol=0)
        block = benchmark.ResidualTCNBlock(6, 4, 3, 4, 2, 0).double().eval()
        mask = torch.ones(2, 1, 37, dtype=torch.bool)
        torch.testing.assert_close(block(original, mask)[:, :, :19], block(changed, mask)[:, :, :19], atol=0, rtol=0)
        model = benchmark.load_tcn(self._settings(dropout=0), 300).double().eval()
        ids = torch.full((2, 37), 2)
        altered = ids.clone()
        altered[:, 19:] = 299
        lengths = torch.tensor([37, 37])
        expected, _ = model.token_features(ids, lengths)
        actual, _ = model.token_features(altered, lengths)
        torch.testing.assert_close(actual[:, :, :19], expected[:, :, :19], atol=0, rtol=0)

    def test_tail_after_512_receives_gradient_and_changes_contract_prediction(self):
        model = benchmark.load_tcn(self._settings(dropout=0), 300).double().eval()
        # Positive weights keep all ReLUs open, proving a path from a unique
        # tail token rather than relying on a fortunate random initialization.
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.fill_(0.01 if "bias" not in name else 0.02)
            model.embedding.weight[0].zero_()
            model.embedding.weight[299].fill_(0.2)
        ids = torch.full((1, 700), 2)
        ids[0, -1] = 299
        logit = model(ids, torch.tensor([700]))
        logit.sum().backward()
        self.assertGreater(float(model.embedding.weight.grad[299].abs().sum()), 0)
        self.assertEqual(float(model.embedding.weight.grad[0].abs().sum()), 0)
        with torch.inference_mode():
            without_tail = ids.clone()
            without_tail[0, -1] = 2
            self.assertNotEqual(float(logit.detach()[0]), float(model(without_tail, torch.tensor([700]))[0]))

    def test_default_receptive_field_counts_encoded_positions(self):
        settings = self._settings(dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256])
        self.assertEqual(benchmark.receptive_field(settings), 2045)
        self.assertEqual(benchmark.load_tcn(settings, 300).receptive_field, 2045)

    def test_length_batches_retain_all_long_contracts_deterministically(self):
        lengths = np.array([3, 12, 2, 1901, 39, 58, 7, 8])
        arrays = pack_sequences([np.full(n, i + 1) for i, n in enumerate(lengths)], np.arange(len(lengths)) % 2)
        settings = self._settings(batch_size=3, batch_token_budget=80)
        first = batches(lengths, settings, 42)
        second = batches(lengths, settings, 42)
        self.assertEqual([row.tolist() for row in first], [row.tolist() for row in second])
        self.assertEqual(sorted(np.concatenate(first).tolist()), list(range(len(lengths))))
        for indices in first:
            ids, sizes, _ = collate(arrays, indices)
            self.assertTrue(len(indices) == 1 or ids.numel() <= settings["batch_token_budget"])
            for row, index in enumerate(indices):
                self.assertEqual(int(sizes[row]), int(lengths[index]))
                self.assertEqual(int(ids[row].count_nonzero()), int(lengths[index]))
        self.assertNotEqual([row.tolist() for row in first], [row.tolist() for row in batches(lengths, settings, 43)])

    def test_weighted_BCE_is_per_contract_and_tail_is_not_a_labelled_window(self):
        settings = self._settings(dropout=0, gradient_clip=1e10, weight_decay=0)
        torch.manual_seed(23)
        reference = benchmark.load_tcn(settings, 300).double()
        candidate = copy.deepcopy(reference)
        arrays = pack_sequences([np.full(n, i + 2) for i, n in enumerate([3, 53, 700])], [0, 1, 1])
        indices = np.arange(3)
        weights = np.array([0.4, 1.7, 2.2], dtype=np.float64)
        ids, sizes, labels = collate(arrays, indices)
        logits = reference(ids, sizes)
        losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.double(), reduction="none")
        expected_loss = (losses * torch.from_numpy(weights)).mean()
        expected_loss.backward()
        optimizer = torch.optim.SGD(candidate.parameters(), lr=0)
        observed_sum = benchmark._train_batch(candidate, optimizer, arrays, weights, indices, settings)
        self.assertAlmostEqual(observed_sum / 3, float(expected_loss.detach()), places=12)
        for (name, expected), (_, actual) in zip(reference.named_parameters(), candidate.named_parameters()):
            with self.subTest(parameter=name):
                self.assertIsNotNone(actual.grad)
                torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-12, atol=1e-12)

    def test_interrupted_resume_matches_exact_weights_history_and_best_logits(self):
        config, arrays = self._config_and_arrays()
        weights = np.array([0.4, 1.6, 0.8, 1.2, 0.9, 1.1])
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache-comparison") as directory:
            base = Path(directory)
            expected = benchmark.train_tcn(config, arrays, arrays, weights, base / "whole", base / "whole-model", 42, 300)
            calls = 0
            train_batch = benchmark._train_batch

            def interrupt_after_mutating_second_step(*args, **kwargs):
                nonlocal calls
                calls += 1
                value = train_batch(*args, **kwargs)
                if calls == 2:
                    raise KeyboardInterrupt()
                return value

            with patch.object(benchmark, "_train_batch", side_effect=interrupt_after_mutating_second_step):
                with self.assertRaises(KeyboardInterrupt):
                    benchmark.train_tcn(config, arrays, arrays, weights, base / "resume", base / "resume-model", 42, 300)
            progress = json.loads((base / "resume/progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["status"], "interrupted")
            self.assertEqual(progress["optimizer_steps"], 1)
            actual = benchmark.train_tcn(config, arrays, arrays, weights, base / "resume", base / "resume-model", 42, 300)
            self.assertEqual(actual["status"], "complete")
            for key in ("best_epoch", "optimizer_steps", "epochs_executed", "validation", "parameters", "input_fingerprint"):
                self.assertEqual(actual[key], expected[key])
            self.assertEqual(json.loads((base / "resume/history.json").read_text(encoding="utf-8")),
                             json.loads((base / "whole/history.json").read_text(encoding="utf-8")))
            np.testing.assert_array_equal(np.load(base / "resume/validation_logits.npy"), np.load(base / "whole/validation_logits.npy"))
            expected_state = torch.load(ROOT / expected["model_path"], weights_only=True)["state_dict"]
            actual_state = torch.load(ROOT / actual["model_path"], weights_only=True)["state_dict"]
            for name in expected_state:
                torch.testing.assert_close(actual_state[name], expected_state[name], rtol=0, atol=0)
            model = benchmark.load_tcn(config["tcn"], 300, actual["model_path"])
            np.testing.assert_array_equal(benchmark.predict_tcn(model, arrays, config["tcn"]), np.load(base / "resume/validation_logits.npy"))
            self.assertEqual(actual["model_sha256"], file_digest(ROOT / actual["model_path"]))
            self.assertEqual(actual["parameters"], actual["trainable_parameters"])
            self.assertEqual(actual["training_encoded_tokens"], len(arrays["tokens"]))
            # Completion is idempotent when the caller has not yet written its
            # training.json marker; the final epoch is not trained again.
            completed = benchmark.train_tcn(config, arrays, arrays, weights, base / "resume", base / "resume-model", 42, 300)
            self.assertEqual(completed["optimizer_steps"], actual["optimizer_steps"])

    def test_resume_rejects_changed_inputs_weights_config_or_source(self):
        config, arrays = self._config_and_arrays()
        weights = np.ones(6)
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache-comparison") as directory:
            base = Path(directory)
            benchmark.train_tcn(config, arrays, arrays, weights, base / "run", base / "model", 42, 300)
            changed_arrays = {key: value.copy() for key, value in arrays.items()}
            changed_arrays["tokens"][-1] += 1
            changed_config = copy.deepcopy(config)
            changed_config["tcn"]["learning_rate"] *= 2
            changed_weights = weights.copy()
            changed_weights[0] = 2
            for candidate_config, candidate_arrays, candidate_weights in (
                    (changed_config, arrays, weights), (config, changed_arrays, weights), (config, arrays, changed_weights)):
                with self.assertRaisesRegex(ValueError, "differ"):
                    benchmark.train_tcn(candidate_config, candidate_arrays, arrays, candidate_weights, base / "run", base / "model", 42, 300)
            sources = benchmark._source_hashes()
            sources["benchmark_tcn.py"] = "changed"
            with patch.object(benchmark, "_source_hashes", return_value=sources):
                with self.assertRaisesRegex(ValueError, "differ"):
                    benchmark.train_tcn(config, arrays, arrays, weights, base / "run", base / "model", 42, 300)

    def test_validation_early_stopping_saves_best_epoch_instead_of_final_epoch(self):
        config, arrays = self._config_and_arrays()
        config["tcn"]["epochs"] = 6
        config["tcn"]["early_stopping_patience"] = 1
        labels = arrays["labels"]
        best_logits = np.where(labels == 1, 2.0, -2.0)
        final_logits = -best_logits
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache-comparison") as directory:
            base = Path(directory)
            # Training deliberately continues to epoch 2 before validation
            # degrades. The final artifact must contain epoch 1's parameters.
            with patch.object(benchmark, "predict_tcn", side_effect=[best_logits, final_logits, best_logits]):
                result = benchmark.train_tcn(config, arrays, arrays, np.ones(6), base / "run", base / "model", 42, 300)
            checkpoint = torch.load(base / "model/resume.pt", weights_only=True)
            selected = torch.load(base / "model/weights.pt", weights_only=True)["state_dict"]
            self.assertEqual(result["best_epoch"], 1)
            self.assertEqual(result["epochs_executed"], 2)
            self.assertEqual(checkpoint["state"]["stale"], 1)
            for name in selected:
                torch.testing.assert_close(selected[name], checkpoint["best_model"][name], rtol=0, atol=0)
            self.assertTrue(any(not torch.equal(selected[name], checkpoint["model"][name]) for name in selected))
            np.testing.assert_array_equal(np.load(base / "run/validation_logits.npy"), best_logits)

    def test_pilot_times_real_batches_after_warmup_without_final_scores_or_model(self):
        config, arrays = self._config_and_arrays()
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache-comparison") as directory:
            base = Path(directory)
            result = benchmark.train_tcn(config, arrays, arrays, np.ones(6), base / "pilot", base / "models", 42, 300, pilot=True)
            self.assertEqual(result["status"], "pilot_complete")
            self.assertFalse(result["final_training_started"])
            self.assertNotIn("validation", result)
            self.assertNotIn("model_path", result)
            self.assertNotIn("best_epoch", result)
            self.assertGreater(result["estimated_training_and_validation_seconds_per_seed"], 0)
            self.assertAlmostEqual(result["estimated_training_and_validation_seconds_all_seeds"],
                                   result["estimated_training_and_validation_seconds_per_seed"] * 3, places=12)
            for phase in ("training", "validation"):
                rows = result["measured_batches"][phase]
                self.assertEqual([row["representative_quantile"] for row in rows], [0.5, 0.9])
                for row in rows:
                    self.assertEqual(row["warmup_batches"], 1)
                    self.assertEqual(row["timed_batches"], 3)
                    self.assertGreater(row["median_seconds"], 0)
                    self.assertGreaterEqual(row["q90_seconds"], row["median_seconds"])
                    self.assertEqual(row["contracts"], len(row["contract_indices"]))
            self.assertFalse((base / "models").exists())
            self.assertFalse((base / "pilot/validation_logits.npy").exists())
            self.assertFalse((base / "pilot/training.json").exists())

    def test_invalid_arrays_and_incompatible_artifacts_are_rejected(self):
        config, arrays = self._config_and_arrays()
        corrupted = {key: value.copy() for key, value in arrays.items()}
        corrupted["tokens"][0] = 0
        model = benchmark.load_tcn(config["tcn"], 300)
        with self.assertRaisesRegex(ValueError, "non-PAD"):
            benchmark.predict_tcn(model, corrupted, config["tcn"])
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache-comparison") as directory:
            base = Path(directory)
            result = benchmark.train_tcn(config, arrays, arrays, np.ones(6), base / "run", base / "model", 42, 300)
            with self.assertRaisesRegex(ValueError, "architecture/vocabulary"):
                benchmark.load_tcn(config["tcn"], 299, result["model_path"])
            changed = {**config["tcn"], "channels": 7}
            with self.assertRaisesRegex(ValueError, "architecture/vocabulary"):
                benchmark.load_tcn(changed, 300, result["model_path"])
            # Changing inference batching preserves topology and is supported.
            loaded = benchmark.load_tcn({**config["tcn"], "batch_size": 1}, 300, result["model_path"])
            self.assertFalse(loaded.training)


if __name__ == "__main__":
    unittest.main()
