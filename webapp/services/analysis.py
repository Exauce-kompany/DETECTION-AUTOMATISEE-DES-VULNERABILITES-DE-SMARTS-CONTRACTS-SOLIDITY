"""Compose model and heuristic results, then persist their summary."""

import logging
import math
import sqlite3

from src.analysis.common import get_risk_level
from src.analysis.performance_analyzer import analyze_contract_performance
from src.analysis.risk_analyzer import analyze_contract_risks
from webapp.persistence.database import save_analysis
from webapp.predictor import predictor
from webapp.services.failures import unavailable_performance, unavailable_risks

logger = logging.getLogger(__name__)


class InvalidPrediction(Exception):
    """The inference engine violated its output contract."""


def prediction_percentages(result: dict) -> tuple[float, float]:
    """Read explicit percentage fields, without guessing units or filling missing scores."""
    if not isinstance(result, dict):
        raise InvalidPrediction("Expected a prediction object")
    values = []
    for label in ("vulnerable", "non_vulnerable"):
        value = result.get(f"probability_{label}_percent")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidPrediction(f"Missing or nonnumeric probability: {label}")
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise InvalidPrediction(f"Probability outside 0..100: {label}")
        values.append(float(value))
    if not math.isclose(sum(values), 100, abs_tol=0.02):
        raise InvalidPrediction("Complementary probabilities must sum to 100")
    return values[0], values[1]


def heuristic_results(code: str, probability: float) -> tuple[dict, dict]:
    # An optional engine failure is visible and does not discard a valid ML result.
    try:
        risks = analyze_contract_risks(code, probability_vulnerable=probability)
    except Exception:
        logger.exception("Static risk analysis failed")
        risks = unavailable_risks(probability)
    try:
        performance = analyze_contract_performance(code)
    except Exception:
        logger.exception("Performance analysis failed")
        performance = unavailable_performance()
    return risks, performance


def save_history(result: dict, code: str, filename: str, size: int) -> None:
    preprocessing = result["preprocessing"]
    try:
        result["analysis_id"] = save_analysis(
            filename=filename,
            file_size_bytes=size,
            predicted_label=result["predicted_label"],
            verdict=result["verdict"],
            confidence=result["confidence_percent"],
            probability_vulnerable=result["probability_vulnerable_percent"],
            probability_non_vulnerable=result["probability_non_vulnerable_percent"],
            risk_score=result["risk_score"],
            risk_level=result["risk_level"],
            tokens_detected=preprocessing["tokens_detected"],
            tokens_used=preprocessing["tokens_used"],
            unknown_tokens=preprocessing["unknown_tokens"],
            unknown_rate=preprocessing["unknown_rate"],
            truncated=preprocessing["truncated"],
            code=code,
        )
    except (sqlite3.Error, OSError):
        logger.exception("Analysis history could not be saved")
        result.update(
            analysis_id=None, history_saved=False, history_error="Historique indisponible."
        )
    else:
        result["history_saved"] = True


def analyze(code: str, filename: str, size: int) -> dict:
    prediction = predictor.predict(code)
    vulnerable, non_vulnerable = prediction_percentages(prediction)
    result = dict(prediction)
    risks, performance = heuristic_results(code, vulnerable)
    score = round(vulnerable)
    level = get_risk_level(score)
    summary = performance["summary"]
    result.update(
        probability_vulnerable_percent=vulnerable,
        probability_non_vulnerable_percent=non_vulnerable,
        risk_analysis=risks,
        performance_analysis=performance,
        ml_risk_score=score,
        ml_risk_level=level,
        static_risk_score=risks["static_analysis"]["score"],
        static_risk_level=risks["static_analysis"]["level"],
        combined_risk_score=None,
        combined_risk_level=None,
        risk_score=score,
        risk_level=level,
        risk_score_method="calibrated_ml_probability",
        performance_efficiency_score=summary["efficiency_score"],
        performance_efficiency_level=summary["efficiency_level"],
        performance_complexity_score=summary["complexity_score"],
        performance_complexity_level=summary["complexity_level"],
        performance_cost_pressure_score=summary["cost_pressure_score"],
        performance_cost_pressure_level=summary["cost_pressure_level"],
        file={
            "filename": filename,
            "size_bytes": size,
            "size_kb": round(size / 1024, 2),
            "extension": ".sol",
        },
        code_preview=code,
        analysis_capabilities={
            "ai_model": {
                "name": "CNN + BiLSTM hiérarchique SMART BUG",
                "type": "binary_classification",
                "classes": ["non_vulnerable", "vulnerable"],
                "detects_exact_vulnerability_type": False,
            },
            "static_analysis": {
                "engine": "SMART BUG Static Risk Analyzer v1",
                "heuristic": True,
                "findings_are_model_predictions": False,
            },
            "performance_analysis": {
                "engine": "SMART BUG Contract Performance Analyzer v1",
                "heuristic": True,
                "exact_gas_measurement": False,
                "compiler_used": False,
                "purpose": "analyse de complexité, efficacité et indicateurs potentiels de coût",
            },
        },
    )
    save_history(result, code, filename, size)
    return result
