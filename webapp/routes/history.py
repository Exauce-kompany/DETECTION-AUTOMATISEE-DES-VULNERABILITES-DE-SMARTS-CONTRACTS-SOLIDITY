"""History endpoints backed by the SQLite repository."""

from fastapi import APIRouter, HTTPException

from webapp.persistence import database

router = APIRouter(prefix="/api", tags=["history"])


@router.get("/history")
def history():
    analyses = database.get_all_analyses()
    return {"success": True, "count": len(analyses), "analyses": analyses}


@router.get("/history-statistics")
def history_statistics():
    return {"success": True, "statistics": database.get_analysis_statistics()}


@router.get("/history/{analysis_id}")
def history_detail(analysis_id: int):
    analysis = database.get_analysis_by_id(analysis_id)
    if analysis is None:
        raise HTTPException(404, "Analyse introuvable.")
    return {"success": True, "analysis": analysis}


@router.delete("/history/{analysis_id}")
def remove_history_item(analysis_id: int):
    if not database.delete_analysis(analysis_id):
        raise HTTPException(404, "Analyse introuvable.")
    return {
        "success": True,
        "message": "Analyse supprimée de l'historique.",
        "analysis_id": analysis_id,
    }


@router.delete("/history")
def remove_all_history():
    database.clear_history()
    return {"success": True, "message": "Historique SMART BUG supprimé."}
