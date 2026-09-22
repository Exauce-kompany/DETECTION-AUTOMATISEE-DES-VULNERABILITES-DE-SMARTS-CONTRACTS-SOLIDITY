from pathlib import Path
import tempfile
import unittest

import numpy as np
import tensorflow as tf

from src.experiment_v3 import calculate_metrics, fit_calibration, make_dataset
from src.model_v3 import build_model
from src.preprocessing_v3 import load_config


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tf.keras.utils.set_random_seed(42)
        cls.config = load_config()
        cls.config.update(window_size=16, embedding_dim=8, convolution_filters=4, lstm_units=4, batch_size=2)
        cls.model = build_model(cls.config, 300)

    def test_padding_windows_do_not_change_prediction(self):
        inputs = np.zeros((1, 2, 16), dtype=np.int32)
        inputs[0, 0, :8] = np.arange(2, 10)
        inputs[0, 1, :2] = [4, 6]
        padded = np.pad(inputs, ((0, 0), (0, 3), (0, 0)))
        np.testing.assert_allclose(self.model(inputs, training=False), self.model(padded, training=False), atol=1e-6)

    def test_saved_model_reproduces_inference(self):
        inputs = np.full((2, 3, 16), 2, dtype=np.int32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.keras"
            self.model.save(path)
            restored = tf.keras.models.load_model(path, compile=False)
            np.testing.assert_allclose(restored(inputs, training=False), self.model(inputs, training=False), atol=1e-6)

    def test_windows_share_one_contract_label(self):
        arrays = {"tokens": np.arange(2, 42, dtype=np.int32), "offsets": np.asarray([0, 33, 40]), "labels": np.asarray([1, 0])}
        x, y = next(iter(make_dataset(arrays, self.config)))
        self.assertEqual(tuple(x.shape), (2, 3, 16))
        np.testing.assert_array_equal(y, [[1], [0]])
        np.testing.assert_array_equal(x[0].numpy().ravel()[:33], arrays["tokens"][:33])

    def test_bucketed_training_visits_every_contract_each_epoch(self):
        lengths = [5, 45, 200, 1100, 9, 78, 130, 3, 1300]
        arrays = {"tokens": np.full(sum(lengths), 2, dtype=np.int32), "offsets": np.cumsum([0] + lengths), "labels": np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 1])}
        dataset = make_dataset(arrays, self.config, training=True)
        for _ in range(2):
            seen = [int(np.count_nonzero(row)) for x, _ in dataset for row in x.numpy()]
            self.assertEqual(sorted(seen), sorted(lengths))


class CalibrationTests(unittest.TestCase):
    def test_calibration_temperature_and_threshold_are_fitted_on_labels(self):
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
        result = fit_calibration(labels, np.asarray([-8, -5, -2, 3, -1, 2, 5, 9]), 0.9)
        self.assertGreater(result["temperature"], 0)
        self.assertGreaterEqual(result["calibrated"]["recall_vulnerable"], 0.9)
        self.assertEqual(result["fit_split"], "calibration")
        self.assertLessEqual(result["calibrated"]["log_loss"], result["raw"]["log_loss"] + 1e-6)

    def test_single_class_holdout_does_not_report_fake_auc(self):
        result = calculate_metrics([1, 1], [0.2, 0.8], 0.5)
        self.assertIsNone(result["roc_auc"])
        self.assertTrue(result["single_class_subset"])

    def test_single_class_calibration_is_rejected(self):
        with self.assertRaises(ValueError):
            fit_calibration([1, 1], [1, 2])


if __name__ == "__main__":
    unittest.main()
