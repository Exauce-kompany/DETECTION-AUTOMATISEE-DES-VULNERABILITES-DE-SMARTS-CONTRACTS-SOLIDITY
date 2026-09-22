
"""
SMART BUG - Contract Performance Analyzer
=========================================

Analyse heuristique des performances d'un Smart Contract Solidity.

IMPORTANT :
- ce module ne compile pas le contrat ;
- il ne calcule pas le gas exact ;
- il fournit des indicateurs statiques utiles pour identifier
  des zones potentiellement coûteuses ou complexes.

Le moteur est indépendant du modèle CNN + BiLSTM V3.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any


ENGINE_NAME = "SMART BUG Contract Performance Analyzer v1"


SEVERITY_WEIGHTS = {
    "critical": 20,
    "high": 12,
    "medium": 7,
    "low": 3,
    "info": 0,
}


@dataclass
class PerformanceFinding:
    id: str
    category: str
    severity: str
    title: str
    description: str
    recommendation: str
    line: int | None = None
    evidence: str | None = None
    confidence: str = "heuristique"
    source: str = "analyse statique"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clamp(value: float, minimum: float = 0, maximum: float = 100) -> float:
    return max(minimum, min(maximum, value))


def strip_comments(code: str) -> str:
    code = re.sub(
        r"/\*.*?\*/",
        lambda match: "\n" * match.group(0).count("\n"),
        code,
        flags=re.DOTALL,
    )
    return re.sub(r"//[^\n]*", "", code)


def count_pattern(pattern: str, code: str, flags: int = 0) -> int:
    return len(re.findall(pattern, code, flags))


def line_number(code: str, position: int) -> int:
    return code.count("\n", 0, position) + 1


def compact(text: str, limit: int = 180) -> str:
    value = " ".join(text.strip().split())
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def efficiency_level(score: float) -> str:
    if score >= 85:
        return "excellent"
    if score >= 70:
        return "bon"
    if score >= 50:
        return "moyen"
    return "faible"


def complexity_level(score: float) -> str:
    if score >= 75:
        return "élevée"
    if score >= 45:
        return "modérée"
    return "faible"


def pressure_level(score: float) -> str:
    if score >= 70:
        return "élevée"
    if score >= 40:
        return "modérée"
    return "faible"


def estimate_state_writes(code: str) -> int:
    assignments = count_pattern(
        r"(?<![=!<>])\b[A-Za-z_][A-Za-z0-9_]*"
        r"(?:\[[^\]]+\])?"
        r"\s*(?:=|\+=|-=|\*=|/=|%=|\+\+|--)",
        code,
    )
    deletes = count_pattern(
        r"\bdelete\s+[A-Za-z_][A-Za-z0-9_]*",
        code,
    )
    return assignments + deletes


def estimate_state_variables(code: str) -> int:
    pattern = re.compile(
        r"^\s*"
        r"(?:mapping\s*\([^;]+\)|"
        r"(?:u?int(?:8|16|32|64|128|256)?|address|bool|string|bytes(?:1|2|4|8|16|32)?|"
        r"[A-Za-z_][A-Za-z0-9_]*(?:\[\])?))"
        r"\s+"
        r"(?:public|private|internal|constant|immutable|\s)*"
        r"[A-Za-z_][A-Za-z0-9_]*"
        r"(?:\s*=\s*[^;]+)?"
        r"\s*;",
        re.MULTILINE,
    )
    return len(list(pattern.finditer(code)))


def extract_metrics(code: str) -> dict[str, Any]:
    clean = strip_comments(code)
    lines = code.splitlines()
    non_empty_lines = [line for line in lines if line.strip()]

    metrics = {
        "lines_total": len(lines),
        "lines_non_empty": len(non_empty_lines),
        "characters": len(code),

        "functions": count_pattern(
            r"\bfunction\s+[A-Za-z_][A-Za-z0-9_]*\s*\(",
            clean,
        ),
        "constructors": count_pattern(
            r"\bconstructor\s*\(",
            clean,
        ),
        "modifiers": count_pattern(
            r"\bmodifier\s+[A-Za-z_][A-Za-z0-9_]*",
            clean,
        ),
        "events": count_pattern(
            r"\bevent\s+[A-Za-z_][A-Za-z0-9_]*",
            clean,
        ),
        "structs": count_pattern(
            r"\bstruct\s+[A-Za-z_][A-Za-z0-9_]*",
            clean,
        ),
        "mappings": count_pattern(
            r"\bmapping\s*\(",
            clean,
        ),

        "for_loops": count_pattern(
            r"\bfor\s*\(",
            clean,
        ),
        "while_loops": count_pattern(
            r"\bwhile\s*\(",
            clean,
        ),
        "do_while_loops": count_pattern(
            r"\bdo\s*\{",
            clean,
        ),

        "conditionals": count_pattern(
            r"\bif\s*\(",
            clean,
        ),
        "require_calls": count_pattern(
            r"\brequire\s*\(",
            clean,
        ),
        "revert_calls": count_pattern(
            r"\brevert\b",
            clean,
        ),
        "assert_calls": count_pattern(
            r"\bassert\s*\(",
            clean,
        ),

        "low_level_calls": (
            count_pattern(r"\.call\s*[\{\(]", clean)
            + count_pattern(r"\.delegatecall\s*[\{\(]", clean)
            + count_pattern(r"\.staticcall\s*[\{\(]", clean)
        ),
        "transfer_calls": count_pattern(
            r"\.transfer\s*\(",
            clean,
        ),
        "send_calls": count_pattern(
            r"\.send\s*\(",
            clean,
        ),

        "assembly_blocks": count_pattern(
            r"\bassembly\s*\{",
            clean,
        ),
        "keccak_calls": count_pattern(
            r"\bkeccak256\s*\(",
            clean,
        ),
        "abi_encode_calls": (
            count_pattern(r"\babi\.encode\s*\(", clean)
            + count_pattern(r"\babi\.encodePacked\s*\(", clean)
        ),

        "storage_keywords": count_pattern(
            r"\bstorage\b",
            clean,
        ),
        "memory_keywords": count_pattern(
            r"\bmemory\b",
            clean,
        ),
        "calldata_keywords": count_pattern(
            r"\bcalldata\b",
            clean,
        ),

        "dynamic_array_operations": (
            count_pattern(r"\.push\s*\(", clean)
            + count_pattern(r"\.pop\s*\(", clean)
        ),
        "estimated_state_variables": estimate_state_variables(clean),
        "estimated_state_writes": estimate_state_writes(clean),
    }

    metrics["loops"] = (
        metrics["for_loops"]
        + metrics["while_loops"]
        + metrics["do_while_loops"]
    )

    metrics["external_calls"] = (
        metrics["low_level_calls"]
        + metrics["transfer_calls"]
        + metrics["send_calls"]
    )

    return metrics


def calculate_complexity(metrics: dict[str, Any]) -> dict[str, Any]:
    score = 0.0

    score += min(metrics["functions"] * 2.0, 24)
    score += min(metrics["conditionals"] * 2.5, 20)
    score += min(metrics["loops"] * 7.0, 28)
    score += min(metrics["external_calls"] * 2.0, 12)
    score += min(metrics["modifiers"] * 1.5, 6)

    if metrics["lines_non_empty"] > 400:
        score += 10
    elif metrics["lines_non_empty"] > 250:
        score += 7
    elif metrics["lines_non_empty"] > 120:
        score += 4

    score = round(clamp(score), 2)

    return {
        "score": score,
        "level": complexity_level(score),
        "method": (
            "heuristique structurelle basée sur fonctions, conditions, "
            "boucles, appels externes et taille du code"
        ),
    }


def build_findings(
    code: str,
    metrics: dict[str, Any],
) -> list[PerformanceFinding]:

    clean = strip_comments(code)
    findings: list[PerformanceFinding] = []

    def add(
        finding_id: str,
        category: str,
        severity: str,
        title: str,
        description: str,
        recommendation: str,
        match=None,
        evidence: str | None = None,
    ):
        line = None

        if match is not None:
            line = line_number(clean, match.start())
            if evidence is None:
                evidence = compact(match.group(0))

        findings.append(
            PerformanceFinding(
                id=finding_id,
                category=category,
                severity=severity,
                title=title,
                description=description,
                recommendation=recommendation,
                line=line,
                evidence=evidence,
            )
        )

    if metrics["loops"] >= 4:
        match = re.search(r"\b(for|while)\s*\(", clean)
        add(
            "PERF-LOOP-001",
            "loops",
            "high",
            "Nombre important de boucles",
            (
                "Le contrat contient plusieurs boucles. "
                "Le coût peut augmenter fortement avec le nombre d'itérations."
            ),
            (
                "Limiter les itérations, paginer les traitements volumineux "
                "et éviter les boucles non bornées sur des données de stockage."
            ),
            match,
        )

    elif metrics["loops"] > 0:
        match = re.search(r"\b(for|while)\s*\(", clean)
        add(
            "PERF-LOOP-002",
            "loops",
            "medium",
            "Boucle à surveiller",
            (
                "Une boucle peut augmenter le coût d'exécution "
                "selon le nombre d'itérations."
            ),
            (
                "Vérifier que la borne reste raisonnable et limiter les "
                "écritures de stockage ou appels externes dans la boucle."
            ),
            match,
        )

    if metrics["external_calls"] >= 5:
        match = re.search(
            r"\.(?:call|delegatecall|staticcall|transfer|send)\s*[\{\(]",
            clean,
        )
        add(
            "PERF-CALL-001",
            "external_calls",
            "high",
            "Nombre élevé d'appels externes",
            (
                "Les appels externes ajoutent du coût et peuvent rendre "
                "l'exécution dépendante d'autres contrats."
            ),
            (
                "Supprimer les appels redondants et éviter de les placer "
                "dans des boucles lorsque cela est possible."
            ),
            match,
        )

    elif metrics["external_calls"] > 0:
        match = re.search(
            r"\.(?:call|delegatecall|staticcall|transfer|send)\s*[\{\(]",
            clean,
        )
        add(
            "PERF-CALL-002",
            "external_calls",
            "low",
            "Appel externe détecté",
            (
                "Un appel externe peut augmenter le coût et la variabilité "
                "de l'exécution."
            ),
            (
                "Vérifier que cet appel est réellement nécessaire "
                "et qu'il n'est pas répété inutilement."
            ),
            match,
        )

    loop_call_match = re.search(
        r"\b(?:for|while)\s*\([^)]*\)\s*\{"
        r"[\s\S]{0,1200}?"
        r"\.(?:call|delegatecall|staticcall|transfer|send)\s*[\{\(]",
        clean,
    )

    if loop_call_match:
        add(
            "PERF-LOOP-CALL-001",
            "external_calls",
            "high",
            "Appel externe dans une boucle",
            (
                "Un appel externe répété à chaque itération peut augmenter "
                "fortement le coût total."
            ),
            (
                "Utiliser si possible une architecture pull, un traitement "
                "par lots ou une pagination."
            ),
            loop_call_match,
        )

    if metrics["estimated_state_writes"] >= 12:
        add(
            "PERF-STORAGE-001",
            "storage",
            "high",
            "Nombre élevé d'écritures potentielles de stockage",
            (
                "Les écritures persistantes sont généralement parmi les "
                "opérations les plus coûteuses sur Ethereum."
            ),
            (
                "Réduire les écritures redondantes et mettre en cache en mémoire "
                "les valeurs réutilisées lorsque la logique le permet."
            ),
            evidence=(
                f"{metrics['estimated_state_writes']} écritures potentielles détectées"
            ),
        )

    elif metrics["estimated_state_writes"] >= 5:
        add(
            "PERF-STORAGE-002",
            "storage",
            "medium",
            "Plusieurs écritures potentielles de stockage",
            (
                "Le contrat semble modifier plusieurs fois des données "
                "persistantes."
            ),
            (
                "Vérifier les écritures répétitives et regrouper certaines "
                "mises à jour si possible."
            ),
            evidence=(
                f"{metrics['estimated_state_writes']} écritures potentielles détectées"
            ),
        )

    if metrics["dynamic_array_operations"] >= 4:
        match = re.search(r"\.(?:push|pop)\s*\(", clean)
        add(
            "PERF-ARRAY-001",
            "dynamic_arrays",
            "medium",
            "Opérations fréquentes sur tableaux dynamiques",
            (
                "Les modifications répétées de tableaux dynamiques en stockage "
                "peuvent augmenter le coût d'exécution."
            ),
            (
                "Évaluer si une mapping ou une structure d'indexation différente "
                "serait plus adaptée."
            ),
            match,
        )

    if metrics["assembly_blocks"] > 0:
        match = re.search(r"\bassembly\s*\{", clean)
        add(
            "PERF-ASM-001",
            "assembly",
            "medium",
            "Bloc assembly détecté",
            (
                "L'assembly peut permettre certaines optimisations, "
                "mais augmente la complexité et le risque d'erreur."
            ),
            (
                "Conserver l'assembly uniquement lorsqu'un gain mesurable "
                "justifie son utilisation et documenter son objectif."
            ),
            match,
        )

    if (
        metrics["keccak_calls"]
        + metrics["abi_encode_calls"]
        >= 8
    ):
        match = re.search(
            r"\b(?:keccak256|abi\.encode|abi\.encodePacked)\s*\(",
            clean,
        )
        add(
            "PERF-HASH-001",
            "encoding_hashing",
            "low",
            "Encodages ou hachages fréquents",
            (
                "Des encodages et hachages répétés peuvent ajouter "
                "un coût mesurable."
            ),
            (
                "Éviter les recalculs identiques lorsque la logique "
                "et la sécurité le permettent."
            ),
            match,
        )

    if metrics["lines_non_empty"] > 500:
        add(
            "PERF-SIZE-001",
            "code_size",
            "medium",
            "Contrat volumineux",
            (
                "Un contrat très volumineux peut être plus difficile "
                "à optimiser et maintenir."
            ),
            (
                "Séparer les responsabilités et vérifier la taille "
                "du bytecode après compilation."
            ),
            evidence=(
                f"{metrics['lines_non_empty']} lignes non vides"
            ),
        )

    if not findings:
        add(
            "PERF-INFO-001",
            "general",
            "info",
            "Aucun indicateur majeur de surcoût détecté",
            (
                "L'analyse heuristique n'a pas identifié de construction "
                "particulièrement coûteuse selon les règles actives."
            ),
            (
                "Confirmer avec une compilation Solidity et un outil "
                "de mesure du gas avant de conclure sur le coût réel."
            ),
        )

    return findings


def calculate_efficiency_score(
    metrics: dict[str, Any],
    findings: list[PerformanceFinding],
    complexity: dict[str, Any],
) -> dict[str, Any]:

    penalty = 0.0

    for finding in findings:
        penalty += SEVERITY_WEIGHTS.get(
            finding.severity,
            0,
        )

    penalty += min(metrics["loops"] * 2.0, 10)
    penalty += min(metrics["estimated_state_writes"] * 0.6, 12)
    penalty += min(metrics["external_calls"] * 0.8, 8)
    penalty += complexity["score"] * 0.12

    score = round(
        clamp(100 - penalty),
        2,
    )

    return {
        "score": score,
        "level": efficiency_level(score),
        "penalty": round(100 - score, 2),
        "method": (
            "score heuristique basé sur findings, boucles, stockage, "
            "appels externes et complexité"
        ),
    }


def build_cost_profile(
    metrics: dict[str, Any],
) -> dict[str, Any]:

    storage_pressure = (
        metrics["estimated_state_writes"] * 3
        + metrics["dynamic_array_operations"] * 2
        + metrics["mappings"]
    )

    execution_pressure = (
        metrics["loops"] * 5
        + metrics["conditionals"]
        + metrics["external_calls"] * 3
        + metrics["keccak_calls"]
        + metrics["abi_encode_calls"]
    )

    code_pressure = (
        metrics["lines_non_empty"] / 8
        + metrics["functions"] * 2
        + metrics["modifiers"]
        + metrics["structs"]
    )

    storage_score = round(clamp(storage_pressure), 2)
    execution_score = round(clamp(execution_pressure), 2)
    code_score = round(clamp(code_pressure), 2)

    overall_score = round(
        storage_score * 0.40
        + execution_score * 0.40
        + code_score * 0.20,
        2,
    )

    return {
        "storage_pressure": {
            "score": storage_score,
            "level": pressure_level(storage_score),
        },
        "execution_pressure": {
            "score": execution_score,
            "level": pressure_level(execution_score),
        },
        "code_size_pressure": {
            "score": code_score,
            "level": pressure_level(code_score),
        },
        "overall_pressure": {
            "score": overall_score,
            "level": pressure_level(overall_score),
        },
        "warning": (
            "Ces indicateurs ne représentent pas une estimation exacte du gas."
        ),
    }


def build_summary(
    metrics: dict[str, Any],
    findings: list[PerformanceFinding],
    efficiency: dict[str, Any],
    complexity: dict[str, Any],
    cost_profile: dict[str, Any],
) -> dict[str, Any]:

    severity_counts = {
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": 0,
        "info": 0,
    }

    recommendations: list[str] = []

    for finding in findings:
        severity_counts[finding.severity] = (
            severity_counts.get(
                finding.severity,
                0,
            )
            + 1
        )

        if finding.recommendation not in recommendations:
            recommendations.append(
                finding.recommendation
            )

    return {
        "efficiency_score": efficiency["score"],
        "efficiency_level": efficiency["level"],
        "complexity_score": complexity["score"],
        "complexity_level": complexity["level"],
        "cost_pressure_score": (
            cost_profile["overall_pressure"]["score"]
        ),
        "cost_pressure_level": (
            cost_profile["overall_pressure"]["level"]
        ),
        "finding_count": len(findings),
        "severity_counts": severity_counts,
        "main_metrics": {
            "functions": metrics["functions"],
            "loops": metrics["loops"],
            "external_calls": metrics["external_calls"],
            "estimated_state_writes": metrics["estimated_state_writes"],
            "assembly_blocks": metrics["assembly_blocks"],
            "lines_non_empty": metrics["lines_non_empty"],
        },
        "recommendations": recommendations[:8],
    }


def analyze_contract_performance(
    code: str,
) -> dict[str, Any]:

    if not isinstance(code, str):
        raise TypeError(
            "Le code Solidity doit être une chaîne de caractères."
        )

    if not code.strip():
        raise ValueError(
            "Le code Solidity est vide."
        )

    metrics = extract_metrics(code)
    complexity = calculate_complexity(metrics)
    findings = build_findings(code, metrics)

    efficiency = calculate_efficiency_score(
        metrics,
        findings,
        complexity,
    )

    cost_profile = build_cost_profile(
        metrics
    )

    summary = build_summary(
        metrics,
        findings,
        efficiency,
        complexity,
        cost_profile,
    )

    return {
        "success": True,
        "engine": ENGINE_NAME,
        "analysis_type": (
            "heuristic_static_performance_analysis"
        ),
        "metrics": metrics,
        "complexity": complexity,
        "efficiency": efficiency,
        "cost_profile": cost_profile,
        "summary": summary,
        "findings": [
            finding.to_dict()
            for finding in findings
        ],
        "limitations": {
            "exact_gas": False,
            "compiler_used": False,
            "bytecode_analyzed": False,
            "statement": (
                "Cette analyse est heuristique. Elle ne compile pas le contrat "
                "et ne mesure pas le gas réel. Les résultats doivent être "
                "confirmés avec solc, Foundry, Hardhat ou un outil équivalent "
                "pour une évaluation exacte du coût d'exécution."
            ),
        },
    }


analyze_performance = analyze_contract_performance


if __name__ == "__main__":
    sample = """
    pragma solidity ^0.8.20;

    contract Example {

        mapping(address => uint256) public balances;

        function batch(address[] calldata users) external {
            for (uint256 i = 0; i < users.length; i++) {
                balances[users[i]] = i;
            }
        }

        function ping(address target) external {
            (bool ok, ) = target.call("");
            require(ok);
        }
    }
    """

    result = analyze_contract_performance(sample)

    print("SMART BUG - PERFORMANCE ANALYZER")
    print(
        "Score d'efficacité :",
        result["efficiency"]["score"],
    )
    print(
        "Niveau :",
        result["efficiency"]["level"],
    )
    print(
        "Complexité :",
        result["complexity"]["score"],
    )
    print(
        "Findings :",
        len(result["findings"]),
    )
