"""Verify contract-level gradients and resumability without network downloads."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    raise unittest.SkipTest("Optional comparison dependencies are not installed")

import src.experiments.benchmark.benchmark_codebert as benchmark
from src.preprocessing import ROOT, file_digest


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def encode(self, text, add_special_tokens=False, truncation=False):
        self.calls.append((add_special_tokens, truncation))
        return [int(token) + 3 for token in text.split()]

    def build_inputs_with_special_tokens(self, ids):
        return [0, *ids, 2]


class FakeEncoder(torch.nn.Module):
    """Tiny contextual encoder: CLS depends on every unmasked input token."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=5, pad_token_id=1)
        self.embedding = torch.nn.Embedding(128, 5)
        self.projection = torch.nn.Linear(5, 5)

    def forward(self, input_ids, attention_mask):
        embeddings = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1)
        context = (embeddings * mask).sum(1) / mask.sum(1)
        states = torch.tanh(self.projection(embeddings + context.unsqueeze(1)))
        return SimpleNamespace(last_hidden_state=states)


def fake_loader(settings):
    return benchmark.CodeBERTContractModel(FakeEncoder()), FakeTokenizer()


class CodeBERTBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_tokenization_retains_every_subtoken_and_reserves_special_tokens(self):
        tokenizer = FakeTokenizer()
        chunks = benchmark.tokenize_contract([str(index) for index in range(15)], tokenizer, 6)
        self.assertEqual([token for chunk in chunks for token in chunk[1:-1]], list(range(3, 18)))
        self.assertEqual([len(chunk) for chunk in chunks], [6, 6, 6, 5])
        self.assertTrue(all(chunk[0] == 0 and chunk[-1] == 2 for chunk in chunks))
        self.assertEqual(tokenizer.calls, [(False, False)])
        with self.assertRaises(ValueError):
            benchmark.tokenize_contract([], tokenizer, 6)
        with self.assertRaises(ValueError):
            benchmark.tokenize_contract(["1"], tokenizer, 2)

    def test_two_pass_gradient_matches_monolithic_weighted_contract_BCE(self):
        torch.manual_seed(23)
        reference = benchmark.CodeBERTContractModel(FakeEncoder()).double()
        candidate = copy.deepcopy(reference)
        encoded = [[[0, 3, 4, 2], [0, 8, 2]], [[0, 5, 2]], [[0, 6, 7, 2], [0, 12, 2], [0, 9, 2]]]
        labels = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64)
        weights = torch.tensor([0.4, 1.7, 2.2], dtype=torch.float64)
        logits = torch.stack(
            [benchmark._chunk_logits(reference, chunks).mean() for chunks in encoded]
        )
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            * weights
        ).mean()
        loss.backward()
        losses = []
        for chunks, label, weight in zip(encoded, labels, weights):
            value, _ = benchmark.backward_contract(
                candidate, chunks, int(label), float(weight), len(encoded), 1
            )
            losses.append(value)
        self.assertAlmostEqual(float(loss.detach()), float(np.mean(losses)), places=12)
        for (name, expected), (_, actual) in zip(
            reference.named_parameters(), candidate.named_parameters()
        ):
            with self.subTest(parameter=name):
                self.assertIsNotNone(actual.grad)
                torch.testing.assert_close(actual.grad, expected.grad, atol=1e-12, rtol=1e-12)
        self.assertGreater(float(candidate.encoder.embedding.weight.grad.abs().sum()), 0)
        self.assertGreater(float(candidate.head.weight.grad.abs().sum()), 0)

    def test_prediction_order_padding_invariance_and_tail_gradient(self):
        torch.manual_seed(42)
        model = benchmark.CodeBERTContractModel(FakeEncoder()).double()
        long_chunks = [[0, *([3] * 100), 2] for _ in range(7)] + [[0, 119, 2]]
        encoded = [long_chunks, [[0, 5, 2]], [[0, 8, 9, 2], [0, 7, 2]]]
        small = benchmark.predict_contracts(model, encoded, 1)
        large = benchmark.predict_contracts(model, encoded, 4)
        np.testing.assert_allclose(small, large, atol=1e-12, rtol=1e-12)
        independently = [benchmark.predict_contracts(model, [chunks], 3)[0] for chunks in encoded]
        np.testing.assert_allclose(small, independently, atol=1e-12, rtol=1e-12)
        benchmark.backward_contract(model, long_chunks, 1, 1.0, 1, 3)
        self.assertGreater(float(model.encoder.embedding.weight.grad[119].abs().sum()), 0)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertFalse(model.training)

    def test_stochastic_training_and_unpinned_revision_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "pinned"):
            benchmark.load_encoder({"revision": "main"})
        with self.assertRaisesRegex(ValueError, "dropout"):
            benchmark.load_encoder({"revision": "a" * 40, "dropout": 0.1})
        model = benchmark.CodeBERTContractModel(FakeEncoder())
        model.unused_dropout = torch.nn.Dropout(0.3)
        with self.assertRaisesRegex(ValueError, "dropout"):
            benchmark.backward_contract(model, [[0, 3, 2]], 1, 1.0, 1, 1)

    def _config_and_rows(self):
        config = {
            "dataset_id": "test-dataset",
            "seeds": [42, 73],
            "codebert": {
                "model_id": "fake/codebert",
                "revision": "a" * 40,
                "chunk_subtokens": 6,
                "chunk_batch_size": 2,
                "effective_batch_size": 2,
                "epochs": 2,
                "early_stopping_patience": 2,
                "learning_rate": 0.02,
                "weight_decay": 0.01,
                "dropout": 0,
                "cpu_threads": 1,
                "checkpoint_every_steps": 1,
            },
        }
        rows = [
            {"tokens": [str(i)] * (3 + i), "label": i % 2, "sample_id": f"sample-{i}"}
            for i in range(6)
        ]
        return config, rows

    def test_interrupted_resume_matches_uninterrupted_training_and_best_logits(self):
        config, rows = self._config_and_rows()
        cache = ROOT / ".cache-comparison"
        cache.mkdir(exist_ok=True)
        with (
            tempfile.TemporaryDirectory(dir=cache) as directory,
            patch.object(benchmark, "load_encoder", side_effect=fake_loader),
        ):
            base = Path(directory)
            expected = benchmark.train_codebert(
                config, rows, rows, np.ones(6), base / "whole", base / "whole-model", 42
            )
            calls = 0
            actual_backward = benchmark.backward_contract

            def interrupt_during_second_batch(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 4:
                    raise KeyboardInterrupt()
                return actual_backward(*args, **kwargs)

            with patch.object(
                benchmark, "backward_contract", side_effect=interrupt_during_second_batch
            ):
                with self.assertRaises(KeyboardInterrupt):
                    benchmark.train_codebert(
                        config, rows, rows, np.ones(6), base / "resume", base / "resume-model", 42
                    )
            actual = benchmark.train_codebert(
                config, rows, rows, np.ones(6), base / "resume", base / "resume-model", 42
            )
            self.assertEqual(actual["status"], "complete")
            self.assertEqual(actual["optimizer_steps"], expected["optimizer_steps"])
            self.assertEqual(actual["best_epoch"], expected["best_epoch"])
            np.testing.assert_array_equal(
                np.load(base / "resume/validation_logits.npy"),
                np.load(base / "whole/validation_logits.npy"),
            )
            expected_state = torch.load(ROOT / expected["model_path"], weights_only=True)
            actual_state = torch.load(ROOT / actual["model_path"], weights_only=True)
            for name in expected_state:
                torch.testing.assert_close(actual_state[name], expected_state[name], rtol=0, atol=0)
            self.assertEqual(actual["model_sha256"], file_digest(ROOT / actual["model_path"]))
            self.assertEqual(actual["parameters"], actual["trainable_parameters"])
            self.assertTrue(actual["training_windows"] > len(rows))
            changed = copy.deepcopy(config)
            changed["codebert"]["learning_rate"] *= 2
            with self.assertRaisesRegex(ValueError, "differ"):
                benchmark.train_codebert(
                    changed, rows, rows, np.ones(6), base / "resume", base / "resume-model", 42
                )

    def test_pilot_reports_estimate_without_saving_final_model(self):
        config, rows = self._config_and_rows()
        with (
            tempfile.TemporaryDirectory(dir=ROOT / ".cache-comparison") as directory,
            patch.object(benchmark, "load_encoder", side_effect=fake_loader),
        ):
            base = Path(directory)
            result = benchmark.train_codebert(
                config, rows, rows, np.ones(6), base / "pilot", base / "models", 42, pilot=True
            )
            self.assertEqual(result["status"], "pilot_complete")
            self.assertFalse(result["final_training_started"])
            self.assertEqual(len(result["representative_contract_indices"]), 2)
            self.assertGreater(result["estimated_training_and_validation_seconds_per_seed"], 0)
            self.assertEqual(
                result["estimated_training_and_validation_seconds_all_seeds"],
                result["estimated_training_and_validation_seconds_per_seed"] * 2,
            )
            self.assertFalse(list((base / "models").glob("*.pt")))
            self.assertFalse((base / "pilot/validation_logits.npy").exists())


if __name__ == "__main__":
    unittest.main()
