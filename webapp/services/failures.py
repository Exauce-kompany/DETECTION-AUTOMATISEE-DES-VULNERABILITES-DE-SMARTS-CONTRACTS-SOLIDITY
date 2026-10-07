"""Explicit unavailable states for the two optional heuristic engines."""


def unavailable_risks(probability: float) -> dict:
    return {
        "success": False,
        "engine": "SMART BUG Static Risk Analyzer v1",
        "error": "Analyse statique indisponible.",
        "static_analysis": {
            "score": None,
            "level": "indisponible",
            "findings_count": 0,
            "severity_counts": dict.fromkeys(("critical", "high", "medium", "low", "info"), 0),
            "findings": [],
        },
        "combined_analysis": {
            "available": False,
            "model_probability_vulnerable": probability,
            "score": None,
            "level": None,
            "formula": "Aucune combinaison des scores",
        },
        "metrics": {},
        "disclaimer": "L'analyse statique n'a pas pu être exécutée.",
    }


def unavailable_performance() -> dict:
    return {
        "success": False,
        "engine": "SMART BUG Contract Performance Analyzer v1",
        "analysis_type": "heuristic_static_performance_analysis",
        "error": "Analyse de performances indisponible.",
        "metrics": {},
        "complexity": {"score": None, "level": "indisponible"},
        "efficiency": {"score": None, "level": "indisponible"},
        "cost_profile": {"overall_pressure": {"score": None, "level": "indisponible"}},
        "summary": {
            "efficiency_score": None,
            "efficiency_level": "indisponible",
            "complexity_score": None,
            "complexity_level": "indisponible",
            "cost_pressure_score": None,
            "cost_pressure_level": "indisponible",
            "finding_count": 0,
            "severity_counts": dict.fromkeys(("critical", "high", "medium", "low", "info"), 0),
            "main_metrics": {},
            "recommendations": [],
        },
        "findings": [],
        "limitations": {
            "exact_gas": False,
            "compiler_used": False,
            "bytecode_analyzed": False,
            "statement": "Le moteur heuristique de performances n'a pas pu être exécuté.",
        },
    }
