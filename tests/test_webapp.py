"""API boundary regressions without model loading or a database server."""

import asyncio
import sqlite3
import threading
import unittest
from io import BytesIO
from unittest.mock import patch

from fastapi import HTTPException, UploadFile

from webapp.routes.analysis import MAX_FILE_SIZE_BYTES, analyze_contract
from webapp.services import analysis


class PredictionContractTests(unittest.TestCase):
    def test_zero_and_small_percentages_are_not_missing_values(self):
        for value in (0, 0.5, 1, 100):
            self.assertEqual(
                analysis.prediction_percentages(
                    {
                        "probability_vulnerable_percent": value,
                        "probability_non_vulnerable_percent": 100 - value,
                    }
                ),
                (value, 100 - value),
            )

    def test_invalid_predictions_cannot_become_a_zero_risk_result(self):
        invalid = [None, {}, {"probabilities": {"vulnerable": 0.5}}]
        invalid += [
            {"probability_vulnerable_percent": value, "probability_non_vulnerable_percent": 50}
            for value in (None, True, "50", -1, 101, float("nan"), float("inf"), 20)
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(analysis.InvalidPrediction):
                analysis.prediction_percentages(value)

    def test_optional_engine_failures_remain_unavailable(self):
        with (
            patch.object(analysis, "analyze_contract_risks", side_effect=RuntimeError("broken")),
            patch.object(
                analysis, "analyze_contract_performance", side_effect=RuntimeError("broken")
            ),
            self.assertLogs(analysis.logger, level="ERROR"),
        ):
            risks, performance = analysis.heuristic_results("contract C {}", 0.5)
        self.assertFalse(risks["success"])
        self.assertIsNone(risks["static_analysis"]["score"])
        self.assertIsNone(performance["summary"]["efficiency_score"])

    def test_database_failure_preserves_prediction(self):
        result = {
            "predicted_label": 0,
            "verdict": "Absent",
            "confidence_percent": 99.5,
            "probability_vulnerable_percent": 0.5,
            "probability_non_vulnerable_percent": 99.5,
            "risk_score": 0,
            "risk_level": "faible",
            "preprocessing": {
                "tokens_detected": 3,
                "tokens_used": 3,
                "unknown_tokens": 0,
                "unknown_rate": 0,
                "truncated": False,
            },
        }
        with (
            patch.object(analysis, "save_analysis", side_effect=sqlite3.OperationalError("locked")),
            self.assertLogs(analysis.logger, level="ERROR"),
        ):
            analysis.save_history(result, "contract C {}", "c.sol", 13)
        self.assertFalse(result["history_saved"])
        self.assertEqual(result["probability_vulnerable_percent"], 0.5)


class UploadTests(unittest.IsolatedAsyncioTestCase):
    async def test_oversize_upload_reads_only_limit_plus_one(self):
        stream = BytesIO(b"x" * (MAX_FILE_SIZE_BYTES + 100))
        with self.assertRaises(HTTPException) as caught:
            await analyze_contract(UploadFile(stream, filename="large.sol"))
        self.assertEqual(caught.exception.status_code, 413)
        self.assertEqual(stream.tell(), MAX_FILE_SIZE_BYTES + 1)

    async def test_synchronous_analysis_does_not_block_event_loop(self):
        started, release = threading.Event(), threading.Event()

        def slow_analysis(*args):
            started.set()
            release.wait(timeout=3)
            return {"success": True}

        upload = UploadFile(BytesIO(b"contract C {}"), filename="c.sol")
        with patch("webapp.routes.analysis.analyze", side_effect=slow_analysis):
            task = asyncio.create_task(analyze_contract(upload))
            try:
                for _ in range(100):
                    await asyncio.sleep(0.01)
                    if started.is_set():
                        break
                self.assertTrue(started.is_set())
                self.assertFalse(task.done(), "Synchronous work blocked the event loop")
            finally:
                release.set()
                await task

    async def test_invalid_model_response_is_server_error(self):
        upload = UploadFile(BytesIO(b"contract C {}"), filename="c.sol")
        with (
            patch(
                "webapp.routes.analysis.analyze",
                side_effect=analysis.InvalidPrediction("missing score"),
            ),
            self.assertLogs("webapp.routes.analysis", level="ERROR"),
            self.assertRaises(HTTPException) as caught,
        ):
            await analyze_contract(upload)
        self.assertEqual(caught.exception.status_code, 500)
