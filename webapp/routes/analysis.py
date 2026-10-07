"""Upload validation and HTTP error translation."""

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from webapp.services.analysis import InvalidPrediction, analyze

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["analysis"])
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024


@router.post("/analyze")
async def analyze_contract(file: UploadFile = File(...)):
    filename = file.filename or "contract.sol"
    if not filename.lower().endswith(".sol"):
        raise HTTPException(400, "Le fichier doit avoir l'extension .sol.")
    # Bound the application buffer even when the multipart upload was spooled to disk.
    content = await file.read(MAX_FILE_SIZE_BYTES + 1)
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(413, "Le fichier est trop volumineux. Taille maximale : 2 Mo.")
    if not content:
        raise HTTPException(400, "Le fichier Solidity est vide.")
    try:
        code = content.decode("utf-8")
    except UnicodeDecodeError:
        code = content.decode("latin-1")
    if not code.strip():
        raise HTTPException(400, "Le fichier Solidity ne contient aucun code exploitable.")
    try:
        # TensorFlow, heuristics and SQLite are synchronous; keep the event loop free.
        return await run_in_threadpool(analyze, code, filename, len(content))
    except InvalidPrediction as exc:
        logger.exception("Invalid model response")
        raise HTTPException(500, "Le moteur IA a retourné une réponse invalide.") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        logger.exception("Contract analysis failed")
        raise HTTPException(500, "Erreur pendant l'analyse par le modèle IA.") from exc
