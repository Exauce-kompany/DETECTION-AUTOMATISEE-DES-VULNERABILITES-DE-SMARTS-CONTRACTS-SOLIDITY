"""Exercise the deployed model through a real HTTP server and isolated history."""

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import requests

from src.data.artifacts import read_jsonl
from src.data.source_paths import resolve_source_path
from src.preprocessing import ROOT, code_from_sample


@unittest.skipUnless((ROOT / "models/active_model.json").exists(), "Train SMART BUG first")
class ProductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            cls.port = sock.getsockname()[1]
        env = dict(
            os.environ,
            SMARTBUG_DATABASE_PATH=str(Path(cls.temporary.name) / "test.db"),
            TF_NUM_INTRAOP_THREADS="2",
            TF_NUM_INTEROP_THREADS="2",
            PYTHONDONTWRITEBYTECODE="1",
        )
        cls.server = subprocess.Popen(
            [
                sys.executable,
                "-B",
                "-m",
                "uvicorn",
                "webapp.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(cls.port),
            ],
            cwd=ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        cls.url = f"http://127.0.0.1:{cls.port}"
        for _ in range(120):
            try:
                if requests.get(cls.url + "/api/status", timeout=1).ok:
                    return
            except requests.RequestException:
                pass
            if cls.server.poll() is not None:
                break
            time.sleep(0.25)
        cls.server.terminate()
        cls.server.wait(timeout=10)
        cls.temporary.cleanup()
        raise RuntimeError("Test API did not start")

    @classmethod
    def tearDownClass(cls):
        cls.server.terminate()
        cls.server.wait(timeout=15)
        cls.temporary.cleanup()

    def analyze(self, code, filename="contract.sol"):
        return requests.post(
            self.url + "/api/analyze",
            files={"file": (filename, code.encode("utf-8"), "text/plain")},
            timeout=120,
        )

    def test_reports_and_dashboard_use_active_experiment(self):
        bundle = json.loads((ROOT / "models/active_model.json").read_text(encoding="utf-8"))
        model = requests.get(self.url + "/api/model-info", timeout=10).json()
        self.assertEqual(model["run_id"], bundle["run_id"])
        self.assertEqual(model["experiment"]["status"], "complete")
        self.assertTrue(requests.get(self.url + "/api/dataset-info", timeout=10).json()["passed"])
        html = requests.get(self.url, timeout=10).text
        self.assertIn('data-model-metric="accuracy"', html)
        self.assertNotIn("85.87%", html)

    def test_http_prediction_reproduces_frozen_test_prediction(self):
        bundle = json.loads((ROOT / "models/active_model.json").read_text(encoding="utf-8"))
        record = read_jsonl(ROOT / bundle["dataset_path"] / "test.jsonl.gz")[0]
        reference = record["source_refs"][0]
        raw = json.loads(
            resolve_source_path(
                Path(bundle["config"]["input_dir"]) / (reference["split"] + ".json")
            ).read_text(encoding="utf-8")
        )[reference["index"]]
        expected = json.loads(
            (ROOT / bundle["results_path"] / "test_predictions.json").read_text(encoding="utf-8")
        )[0]
        code = code_from_sample(raw)
        response = self.analyze(code)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertAlmostEqual(
            result["probability_vulnerable"], expected["probability_vulnerable"], places=5
        )
        self.assertEqual(result["predicted_label"], expected["predicted_label"])
        self.assertFalse(result["preprocessing"]["truncated"])
        self.assertEqual(
            result["preprocessing"]["tokens_used"], result["preprocessing"]["tokens_detected"]
        )
        self.assertFalse(result["risk_analysis"]["combined_analysis"]["available"])
        self.assertIsNone(result["combined_risk_score"])
        annotated = self.analyze("// @vulnerable_at_lines: 1 SWC-107\n" + code).json()
        self.assertAlmostEqual(
            annotated["probability_vulnerable"], result["probability_vulnerable"], places=6
        )

    def test_invalid_uploads_have_client_errors(self):
        for code, filename in (
            ("", "empty.sol"),
            ("// only comments", "comment.sol"),
            ("contract C {}", "bad.txt"),
        ):
            with self.subTest(filename=filename):
                self.assertEqual(self.analyze(code, filename).status_code, 400)


if __name__ == "__main__":
    unittest.main()
