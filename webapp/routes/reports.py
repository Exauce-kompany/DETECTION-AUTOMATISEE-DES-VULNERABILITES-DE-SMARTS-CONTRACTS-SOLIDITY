"""Read-only model, dataset and application reports."""

from fastapi import APIRouter, HTTPException

from webapp.predictor import predictor

router = APIRouter(prefix="/api", tags=["reports"])


@router.get("/status")
def api_status():
    return {
        "success": True,
        "application": "SMART BUG",
        "status": "online",
        "model": "CNN + BiLSTM hiérarchique SMART BUG",
        "model_ready": predictor.manifest_path.is_file(),
        "risk_engine": "SMART BUG Static Risk Analyzer v1",
        "performance_engine": "SMART BUG Contract Performance Analyzer v1",
        "database": "SQLite",
        "classification": "binary",
        "classes": ["non_vulnerable", "vulnerable"],
    }


@router.get("/model-info")
def model_info():
    try:
        return predictor.model_report()
    except (RuntimeError, OSError, ValueError) as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/dataset-info")
def dataset_info():
    try:
        return predictor.dataset_report()
    except (RuntimeError, OSError, ValueError) as exc:
        raise HTTPException(503, str(exc)) from exc
