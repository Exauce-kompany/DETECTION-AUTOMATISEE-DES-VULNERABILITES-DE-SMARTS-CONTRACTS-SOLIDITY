from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from webapp.database import (
    clear_history,
    delete_analysis,
    get_all_analyses,
    get_analysis_by_id,
    get_analysis_statistics,
    init_database,
    save_analysis,
)
from webapp.predictor import predictor
from webapp.risk_analyzer import analyze_contract_risks
from webapp.performance_analyzer import analyze_contract_performance


# ============================================================
# SMART BUG - FASTAPI APPLICATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024  # 2 Mo


app = FastAPI(
    title="SMART BUG",
    description=(
        "Plateforme intelligente d'analyse de sécurité "
        "des Smart Contracts Solidity."
    ),
    version="2.2.0",
)


app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)


templates = Jinja2Templates(
    directory=str(TEMPLATES_DIR)
)


# Initialise la base SQLite au démarrage du module.
init_database()


# ============================================================
# UTILITAIRES
# ============================================================

def clamp(value: float, minimum: float = 0, maximum: float = 100) -> float:
    return max(minimum, min(maximum, value))


def normalize_probability_percent(result: dict, label: str) -> float:
    """
    Extrait une probabilité en pourcentage depuis différentes
    structures possibles retournées par predictor.py.

    Exemples supportés :
    - probability_vulnerable_percent
    - probabilities["vulnerable_percent"]
    - probabilities["vulnerable"] en 0..1 ou 0..100
    """

    flat_key = f"probability_{label}_percent"

    if flat_key in result:
        try:
            return float(
                clamp(
                    float(result[flat_key])
                )
            )
        except (TypeError, ValueError):
            pass


    probabilities = result.get(
        "probabilities",
        {}
    )


    percent_key = f"{label}_percent"

    if isinstance(probabilities, dict):

        if percent_key in probabilities:
            try:
                return float(
                    clamp(
                        float(
                            probabilities[
                                percent_key
                            ]
                        )
                    )
                )
            except (TypeError, ValueError):
                pass


        if label in probabilities:
            try:
                value = float(
                    probabilities[label]
                )

                if 0 <= value <= 1:
                    value *= 100

                return float(
                    clamp(value)
                )

            except (TypeError, ValueError):
                pass


    return 0.0


def calculate_probability_risk_score(
    probability_vulnerable: float
) -> int:
    """
    Score de risque basé uniquement sur la probabilité
    vulnérable du modèle IA.
    """

    return int(
        round(
            clamp(
                probability_vulnerable
            )
        )
    )


def get_risk_level(score: float) -> str:

    if score >= 85:
        return "critique"

    if score >= 65:
        return "élevé"

    if score >= 40:
        return "modéré"

    return "faible"


def safe_float(
    value,
    default: float = 0.0
) -> float:

    try:
        return float(value)

    except (TypeError, ValueError):
        return default


def safe_int(
    value,
    default: int = 0
) -> int:

    try:
        return int(value)

    except (TypeError, ValueError):
        return default


def decode_solidity_file(
    content: bytes
) -> str:
    """
    Décodage UTF-8 avec repli Latin-1.
    """

    try:
        return content.decode(
            "utf-8"
        )

    except UnicodeDecodeError:
        try:
            return content.decode(
                "latin-1"
            )

        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Impossible de décoder le fichier Solidity."
                ),
            ) from exc


# ============================================================
# PAGE PRINCIPALE
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def home(
    request: Request
):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# ============================================================
# STATUS
# ============================================================

@app.get(
    "/api/status"
)
async def api_status():

    return {
        "success": True,
        "application": "SMART BUG",
        "status": "online",
        "model": "BiLSTM V2",
        "risk_engine": (
            "SMART BUG Static Risk Analyzer v1"
        ),
        "performance_engine": (
            "SMART BUG Contract Performance Analyzer v1"
        ),
        "database": "SQLite",
        "classification": "binary",
        "classes": [
            "non_vulnerable",
            "vulnerable",
        ],
    }


# ============================================================
# ANALYSE D'UN SMART CONTRACT
# ============================================================

@app.post(
    "/api/analyze"
)
async def analyze_contract(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validation extension
    # --------------------------------------------------------

    filename = (
        file.filename
        or
        "contract.sol"
    )


    if not filename.lower().endswith(
        ".sol"
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Le fichier doit avoir "
                "l'extension .sol."
            ),
        )


    # --------------------------------------------------------
    # Lecture du fichier
    # --------------------------------------------------------

    content = await file.read()


    if not content:

        raise HTTPException(
            status_code=400,
            detail=(
                "Le fichier Solidity est vide."
            ),
        )


    if len(content) > MAX_FILE_SIZE_BYTES:

        raise HTTPException(
            status_code=413,
            detail=(
                "Le fichier est trop volumineux. "
                "Taille maximale : 2 Mo."
            ),
        )


    code = decode_solidity_file(
        content
    )


    if not code.strip():

        raise HTTPException(
            status_code=400,
            detail=(
                "Le fichier Solidity ne contient "
                "aucun code exploitable."
            ),
        )


    # ========================================================
    # 1. PRÉDICTION IA BILSTM V2
    # ========================================================

    try:

        prediction_result = (
            predictor.predict(
                code
            )
        )

    except Exception as exc:

        print(
            "Erreur prédiction IA :",
            exc
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Erreur pendant l'analyse "
                "par le modèle IA."
            ),
        ) from exc


    if not isinstance(
        prediction_result,
        dict
    ):

        raise HTTPException(
            status_code=500,
            detail=(
                "Le moteur IA a retourné "
                "une réponse invalide."
            ),
        )


    result = dict(
        prediction_result
    )


    # ========================================================
    # 2. NORMALISATION DES PROBABILITÉS
    # ========================================================

    vulnerable_probability = (
        normalize_probability_percent(
            result,
            "vulnerable",
        )
    )


    safe_probability = (
        normalize_probability_percent(
            result,
            "non_vulnerable",
        )
    )


    # Si une seule probabilité est exploitable,
    # on reconstitue l'autre.

    if (
        vulnerable_probability == 0
        and
        safe_probability > 0
    ):

        vulnerable_probability = (
            100
            -
            safe_probability
        )


    if (
        safe_probability == 0
        and
        vulnerable_probability > 0
    ):

        safe_probability = (
            100
            -
            vulnerable_probability
        )


    vulnerable_probability = float(
        clamp(
            vulnerable_probability
        )
    )


    safe_probability = float(
        clamp(
            safe_probability
        )
    )


    result[
        "probability_vulnerable_percent"
    ] = vulnerable_probability


    result[
        "probability_non_vulnerable_percent"
    ] = safe_probability


    # ========================================================
    # 3. ANALYSE STATIQUE DES RISQUES
    # ========================================================

    try:

        risk_analysis = (
            analyze_contract_risks(
                code,
                probability_vulnerable=(
                    vulnerable_probability
                ),
            )
        )

    except Exception as exc:

        print(
            "Erreur analyse statique :",
            exc
        )

        # L'analyse IA reste disponible même si
        # le moteur heuristique rencontre une erreur.

        risk_analysis = {
            "success": False,
            "engine": (
                "SMART BUG Static Risk Analyzer v1"
            ),
            "error": str(exc),
            "static_analysis": {
                "score": 0,
                "level": "indisponible",
                "findings_count": 0,
                "severity_counts": {
                    "critical": 0,
                    "high": 0,
                    "medium": 0,
                    "low": 0,
                    "info": 0,
                },
                "findings": [],
            },
            "combined_analysis": {
                "available": False,
                "model_probability_vulnerable": (
                    vulnerable_probability
                ),
                "score": None,
                "level": None,
                "formula": (
                    "70% probabilité IA + "
                    "30% score heuristique statique"
                ),
            },
            "metrics": {},
            "disclaimer": (
                "L'analyse statique n'a pas pu "
                "être exécutée."
            ),
        }


    # ========================================================
    # 4. ANALYSE HEURISTIQUE DES PERFORMANCES
    # ========================================================

    try:

        performance_analysis = (
            analyze_contract_performance(
                code
            )
        )

    except Exception as exc:

        print(
            "Erreur analyse performances :",
            exc
        )

        performance_analysis = {
            "success": False,
            "engine": (
                "SMART BUG Contract Performance Analyzer v1"
            ),
            "analysis_type": (
                "heuristic_static_performance_analysis"
            ),
            "error": str(exc),
            "metrics": {},
            "complexity": {
                "score": 0,
                "level": "indisponible",
            },
            "efficiency": {
                "score": 0,
                "level": "indisponible",
            },
            "cost_profile": {
                "overall_pressure": {
                    "score": 0,
                    "level": "indisponible",
                }
            },
            "summary": {
                "efficiency_score": 0,
                "efficiency_level": "indisponible",
                "complexity_score": 0,
                "complexity_level": "indisponible",
                "cost_pressure_score": 0,
                "cost_pressure_level": "indisponible",
                "finding_count": 0,
                "severity_counts": {
                    "critical": 0,
                    "high": 0,
                    "medium": 0,
                    "low": 0,
                    "info": 0,
                },
                "main_metrics": {},
                "recommendations": [],
            },
            "findings": [],
            "limitations": {
                "exact_gas": False,
                "compiler_used": False,
                "bytecode_analyzed": False,
                "statement": (
                    "Le moteur heuristique de performances "
                    "n'a pas pu être exécuté."
                ),
            },
        }


    # ========================================================
    # 5. SCORES DE RISQUE
    # ========================================================

    ml_risk_score = (
        calculate_probability_risk_score(
            vulnerable_probability
        )
    )


    ml_risk_level = (
        get_risk_level(
            ml_risk_score
        )
    )


    combined = (
        risk_analysis.get(
            "combined_analysis",
            {}
        )
    )


    if (
        combined.get(
            "available"
        )
        and
        combined.get(
            "score"
        )
        is not None
    ):

        final_risk_score = safe_int(
            combined.get(
                "score"
            ),
            ml_risk_score,
        )


        final_risk_level = str(
            combined.get(
                "level"
            )
            or
            get_risk_level(
                final_risk_score
            )
        )

    else:

        final_risk_score = (
            ml_risk_score
        )

        final_risk_level = (
            ml_risk_level
        )


    final_risk_score = int(
        clamp(
            final_risk_score
        )
    )


    # ========================================================
    # 6. INFORMATIONS GLOBALES DU RÉSULTAT
    # ========================================================

    result[
        "performance_analysis"
    ] = performance_analysis


    performance_summary = (
        performance_analysis.get(
            "summary",
            {}
        )
        or
        {}
    )


    result[
        "performance_efficiency_score"
    ] = safe_float(
        performance_summary.get(
            "efficiency_score"
        ),
        0.0,
    )


    result[
        "performance_efficiency_level"
    ] = str(
        performance_summary.get(
            "efficiency_level"
        )
        or
        "indisponible"
    )


    result[
        "performance_complexity_score"
    ] = safe_float(
        performance_summary.get(
            "complexity_score"
        ),
        0.0,
    )


    result[
        "performance_complexity_level"
    ] = str(
        performance_summary.get(
            "complexity_level"
        )
        or
        "indisponible"
    )


    result[
        "performance_cost_pressure_score"
    ] = safe_float(
        performance_summary.get(
            "cost_pressure_score"
        ),
        0.0,
    )


    result[
        "performance_cost_pressure_level"
    ] = str(
        performance_summary.get(
            "cost_pressure_level"
        )
        or
        "indisponible"
    )


    result[
        "risk_analysis"
    ] = risk_analysis


    result[
        "ml_risk_score"
    ] = ml_risk_score


    result[
        "ml_risk_level"
    ] = ml_risk_level


    result[
        "static_risk_score"
    ] = safe_int(
        risk_analysis
            .get(
                "static_analysis",
                {}
            )
            .get(
                "score"
            ),
        0,
    )


    result[
        "static_risk_level"
    ] = (
        risk_analysis
            .get(
                "static_analysis",
                {}
            )
            .get(
                "level"
            )
        or
        "indisponible"
    )


    result[
        "combined_risk_score"
    ] = final_risk_score


    result[
        "combined_risk_level"
    ] = final_risk_level


    # Le frontend actuel utilise déjà risk_score/risk_level.
    # Ils représentent désormais le score combiné quand
    # l'analyse statique est disponible.

    result[
        "risk_score"
    ] = final_risk_score


    result[
        "risk_level"
    ] = final_risk_level


    result[
        "risk_score_method"
    ] = (
        "combined_ml_static"
        if combined.get(
            "available"
        )
        else "ml_probability_only"
    )


    result[
        "file"
    ] = {
        "filename":
            filename,

        "size_bytes":
            len(content),

        "size_kb":
            round(
                len(content)
                /
                1024,
                2,
            ),

        "extension":
            ".sol",
    }


    result[
        "code_preview"
    ] = code


    result[
        "analysis_capabilities"
    ] = {
        "ai_model": {
            "name": "BiLSTM V2",
            "type": "binary_classification",
            "classes": [
                "non_vulnerable",
                "vulnerable",
            ],
            "detects_exact_vulnerability_type": False,
        },
        "static_analysis": {
            "engine": (
                "SMART BUG Static Risk Analyzer v1"
            ),
            "heuristic": True,
            "findings_are_model_predictions": False,
        },
        "performance_analysis": {
            "engine": (
                "SMART BUG Contract Performance Analyzer v1"
            ),
            "heuristic": True,
            "exact_gas_measurement": False,
            "compiler_used": False,
            "purpose": (
                "analyse de complexité, efficacité et "
                "indicateurs potentiels de coût"
            ),
        },
    }


    # ========================================================
    # 7. PRÉPARATION POUR SQLITE
    # ========================================================

    preprocessing = (
        result.get(
            "preprocessing",
            {}
        )
        or
        {}
    )


    predicted_label = safe_int(
        result.get(
            "predicted_label"
        ),
        0,
    )


    verdict = str(
        result.get(
            "verdict"
        )
        or
        (
            "VULNÉRABLE"
            if predicted_label == 1
            else "NON VULNÉRABLE"
        )
    )


    confidence_percent = safe_float(
        result.get(
            "confidence_percent"
        ),
        0.0,
    )


    # ========================================================
    # 8. SAUVEGARDE HISTORIQUE
    # ========================================================

    try:

        analysis_id = save_analysis(

            filename=
                filename,

            file_size_bytes=
                len(content),

            predicted_label=
                predicted_label,

            verdict=
                verdict,

            confidence=
                confidence_percent,

            probability_vulnerable=
                vulnerable_probability,

            probability_non_vulnerable=
                safe_probability,

            risk_score=
                final_risk_score,

            risk_level=
                final_risk_level,

            tokens_detected=
                safe_int(
                    preprocessing.get(
                        "tokens_detected"
                    ),
                    0,
                ),

            tokens_used=
                safe_int(
                    preprocessing.get(
                        "tokens_used"
                    ),
                    0,
                ),

            unknown_tokens=
                safe_int(
                    preprocessing.get(
                        "unknown_tokens"
                    ),
                    0,
                ),

            unknown_rate=
                safe_float(
                    preprocessing.get(
                        "unknown_rate"
                    ),
                    0.0,
                ),

            truncated=
                bool(
                    preprocessing.get(
                        "truncated",
                        False,
                    )
                ),

            code=
                code,
        )


        result[
            "analysis_id"
        ] = analysis_id


    except Exception as exc:

        # Une erreur de base ne doit pas supprimer
        # le résultat de l'analyse.

        print(
            "Erreur sauvegarde SQLite :",
            exc
        )


        result[
            "analysis_id"
        ] = None


        result[
            "history_saved"
        ] = False


        result[
            "history_error"
        ] = str(exc)


    else:

        result[
            "history_saved"
        ] = True


    # ========================================================
    # 9. RETOUR API
    # ========================================================

    return result


# ============================================================
# HISTORIQUE COMPLET
# ============================================================

@app.get(
    "/api/history"
)
async def history():

    analyses = (
        get_all_analyses()
    )

    return {
        "success": True,
        "count": len(
            analyses
        ),
        "analyses": analyses,
    }


# ============================================================
# STATISTIQUES HISTORIQUE
# ============================================================

@app.get(
    "/api/history-statistics"
)
async def history_statistics():

    statistics = (
        get_analysis_statistics()
    )

    return {
        "success": True,
        "statistics": statistics,
    }


# ============================================================
# DÉTAIL D'UNE ANALYSE
# ============================================================

@app.get(
    "/api/history/{analysis_id}"
)
async def history_detail(
    analysis_id: int
):

    analysis = (
        get_analysis_by_id(
            analysis_id
        )
    )


    if analysis is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "Analyse introuvable."
            ),
        )


    return {
        "success": True,
        "analysis": analysis,
    }


# ============================================================
# SUPPRESSION D'UNE ANALYSE
# ============================================================

@app.delete(
    "/api/history/{analysis_id}"
)
async def remove_history_item(
    analysis_id: int
):

    existing = (
        get_analysis_by_id(
            analysis_id
        )
    )


    if existing is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "Analyse introuvable."
            ),
        )


    delete_analysis(
        analysis_id
    )


    return {
        "success": True,
        "message": (
            "Analyse supprimée "
            "de l'historique."
        ),
        "analysis_id": analysis_id,
    }


# ============================================================
# SUPPRESSION DE TOUT L'HISTORIQUE
# ============================================================

@app.delete(
    "/api/history"
)
async def remove_all_history():

    clear_history()


    return {
        "success": True,
        "message": (
            "Historique SMART BUG supprimé."
        ),
    }
