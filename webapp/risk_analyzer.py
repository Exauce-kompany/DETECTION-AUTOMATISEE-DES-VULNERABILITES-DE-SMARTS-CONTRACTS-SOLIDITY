"""
SMART BUG - Static Risk Analyzer v1
-----------------------------------
Analyse heuristique du code Solidity.

IMPORTANT:
- Le modèle CNN + BiLSTM V3 reste un classifieur BINAIRE:
  vulnerable / non_vulnerable.
- Les catégories affichées ici proviennent d'une analyse
  statique heuristique distincte du modèle IA.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


SEVERITY_POINTS = {
    "critical": 30,
    "high": 20,
    "medium": 10,
    "low": 5,
    "info": 0,
}

SEVERITY_ORDER = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
    "info": 4,
}


def clamp(value: float, minimum: float = 0, maximum: float = 100) -> float:
    """Limite une valeur dans un intervalle."""
    return max(minimum, min(maximum, value))


def get_risk_level(score: float) -> str:
    """Convertit un score 0-100 en niveau de risque."""
    if score >= 85:
        return "critique"
    if score >= 65:
        return "élevé"
    if score >= 40:
        return "modéré"
    return "faible"


def normalize_probability(value: Any) -> float | None:
    """
    Normalise une probabilité en pourcentage.

    Accepte:
    - 0.95 -> 95.0
    - 95 -> 95.0
    - None -> None
    """
    if value is None:
        return None

    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None

    if 0 <= probability <= 1:
        probability *= 100

    return float(clamp(probability))


def strip_comments_preserve_lines(code: str) -> list[str]:
    """
    Supprime les commentaires Solidity en conservant le nombre
    de lignes afin que les numéros de ligne restent exploitables.
    """
    output: list[str] = []
    in_block_comment = False

    for raw_line in code.splitlines():
        result_chars: list[str] = []
        i = 0

        while i < len(raw_line):
            if in_block_comment:
                end = raw_line.find("*/", i)

                if end == -1:
                    i = len(raw_line)
                    continue

                in_block_comment = False
                i = end + 2
                continue

            if raw_line.startswith("//", i):
                break

            if raw_line.startswith("/*", i):
                in_block_comment = True
                i += 2
                continue

            result_chars.append(raw_line[i])
            i += 1

        output.append("".join(result_chars))

    return output


def compact_evidence(text: str, maximum: int = 220) -> str:
    """Réduit une ligne de code trop longue pour l'interface."""
    value = " ".join(text.strip().split())

    if len(value) <= maximum:
        return value

    return value[: maximum - 3] + "..."


def make_finding(
    finding_id: str,
    category: str,
    severity: str,
    title: str,
    description: str,
    recommendation: str,
    line: int | None = None,
    evidence: str | None = None,
    confidence: str = "heuristique",
) -> dict[str, Any]:
    """Construit une alerte standardisée."""
    return {
        "id": finding_id,
        "category": category,
        "severity": severity,
        "title": title,
        "description": description,
        "recommendation": recommendation,
        "line": line,
        "evidence": compact_evidence(evidence) if evidence else None,
        "confidence": confidence,
        "source": "analyse_statique",
    }


def looks_like_state_mutation(line: str) -> bool:
    """
    Heuristique simple pour repérer une modification d'état
    après un appel externe.
    """
    stripped = line.strip()

    if not stripped:
        return False

    if stripped.startswith(("require(", "assert(", "revert(", "emit ")):
        return False

    patterns = (
        r"\b[A-Za-z_]\w*(?:\[[^\]]+\])?\s*=(?!=)",
        r"\b[A-Za-z_]\w*(?:\[[^\]]+\])?\s*\+=",
        r"\b[A-Za-z_]\w*(?:\[[^\]]+\])?\s*-=",
        r"\b[A-Za-z_]\w*(?:\[[^\]]+\])?\s*\+\+",
        r"\b[A-Za-z_]\w*(?:\[[^\]]+\])?\s*--",
        r"\bdelete\s+[A-Za-z_]\w*",
    )

    return any(re.search(pattern, stripped) for pattern in patterns)


def find_function_context(lines: list[str], index: int, search_back: int = 60) -> str:
    """
    Retourne approximativement la signature de la fonction
    contenant une ligne donnée.
    """
    start = max(0, index - search_back)

    for i in range(index, start - 1, -1):
        candidate = lines[i].strip()

        if re.search(r"\bfunction\b", candidate):
            return candidate

    return ""


def is_return_value_checked(line: str, call_kind: str) -> bool:
    """
    Vérification heuristique du contrôle de la valeur de retour
    pour .call() ou .send().
    """
    stripped = line.strip()

    if re.search(r"\brequire\s*\(", stripped):
        return True

    if re.search(r"\bassert\s*\(", stripped):
        return True

    if re.search(r"\bif\s*\(", stripped):
        return True

    if call_kind == "call":
        # Ex: (bool success, ) = target.call(...)
        if re.search(
            r"\(\s*(?:bool\s+)?[A-Za-z_]\w*\s*,[^)]*\)\s*=",
            stripped,
        ):
            return True

        # Ex: bool success = target.call(...)
        if re.search(
            r"\bbool\s+[A-Za-z_]\w*\s*=",
            stripped,
        ):
            return True

    if call_kind == "send":
        # Ex: bool sent = payable(...).send(...)
        if re.search(
            r"\bbool\s+[A-Za-z_]\w*\s*=",
            stripped,
        ):
            return True

        # Ex: sent = payable(...).send(...)
        if re.search(
            r"\b[A-Za-z_]\w*\s*=\s*[^;]*\.send\s*\(",
            stripped,
        ):
            return True

    return False


def analyze_contract_risks(
    code: str,
    probability_vulnerable: float | None = None,
) -> dict[str, Any]:
    """
    Analyse statique heuristique d'un Smart Contract Solidity.

    Parameters
    ----------
    code:
        Code source Solidity.
    probability_vulnerable:
        Probabilité fournie par le CNN + BiLSTM V3.
        Peut être au format 0..1 ou 0..100.

    Returns
    -------
    dict
        Résultat complet exploitable par FastAPI et le frontend.
    """
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Le code Solidity est vide.")

    original_lines = code.splitlines()
    lines = strip_comments_preserve_lines(code)

    findings: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    external_call_lines: list[int] = []
    low_level_call_lines: list[int] = []
    loop_lines: list[int] = []

    function_count = 0
    state_mutations = 0

    def add_finding(finding: dict[str, Any]) -> None:
        key = (
            finding.get("id"),
            finding.get("line"),
            finding.get("evidence"),
        )

        if key in seen:
            return

        seen.add(key)
        findings.append(finding)

    for index, line in enumerate(lines):
        line_number = index + 1
        stripped = line.strip()

        if not stripped:
            continue

        if re.search(r"\bfunction\b", stripped):
            function_count += 1

        if looks_like_state_mutation(stripped):
            state_mutations += 1

        # ----------------------------------------------------
        # Boucles
        # ----------------------------------------------------
        if re.search(r"\b(for|while)\s*\(", stripped):
            loop_lines.append(index)

            add_finding(
                make_finding(
                    "LOOP",
                    "denial_of_service",
                    "low",
                    "Boucle détectée",
                    (
                        "Une boucle dépendant de données non bornées peut "
                        "provoquer une consommation importante de gas ou "
                        "rendre une fonction difficile à exécuter."
                    ),
                    (
                        "S'assurer que le nombre d'itérations est borné ou "
                        "utiliser un traitement par lots."
                    ),
                    line_number,
                    stripped,
                    "faible",
                )
            )

        # ----------------------------------------------------
        # tx.origin
        # ----------------------------------------------------
        if re.search(r"\btx\.origin\b", stripped):
            add_finding(
                make_finding(
                    "TX_ORIGIN",
                    "access_control",
                    "high",
                    "Utilisation de tx.origin",
                    (
                        "tx.origin peut rendre un mécanisme d'autorisation "
                        "vulnérable à des appels effectués via un contrat "
                        "intermédiaire."
                    ),
                    (
                        "Utiliser msg.sender pour les contrôles d'accès et "
                        "appliquer un mécanisme d'autorisation explicite."
                    ),
                    line_number,
                    stripped,
                    "élevée",
                )
            )

        # ----------------------------------------------------
        # delegatecall
        # ----------------------------------------------------
        if re.search(r"\.delegatecall\s*(?:\{|\()", stripped):
            external_call_lines.append(index)
            low_level_call_lines.append(index)

            add_finding(
                make_finding(
                    "DELEGATECALL",
                    "external_call",
                    "high",
                    "Utilisation de delegatecall",
                    (
                        "delegatecall exécute le code d'un autre contrat "
                        "dans le contexte de stockage du contrat appelant."
                    ),
                    (
                        "Limiter strictement les adresses cibles et auditer "
                        "les hypothèses de stockage ainsi que les droits "
                        "d'administration."
                    ),
                    line_number,
                    stripped,
                    "élevée",
                )
            )

        # ----------------------------------------------------
        # selfdestruct / suicide
        # ----------------------------------------------------
        if re.search(r"\b(selfdestruct|suicide)\s*\(", stripped):
            add_finding(
                make_finding(
                    "SELFDESTRUCT",
                    "destructive_operation",
                    "high",
                    "Opération destructive détectée",
                    (
                        "Une instruction de destruction du contrat apparaît "
                        "dans le code et mérite une vérification spécifique."
                    ),
                    (
                        "Supprimer cette logique lorsqu'elle n'est pas "
                        "indispensable ou appliquer un contrôle d'accès "
                        "strict."
                    ),
                    line_number,
                    stripped,
                    "élevée",
                )
            )

        # ----------------------------------------------------
        # .call
        # ----------------------------------------------------
        if re.search(r"\.call\s*(?:\{|\()", stripped):
            external_call_lines.append(index)
            low_level_call_lines.append(index)

            if not is_return_value_checked(stripped, "call"):
                add_finding(
                    make_finding(
                        "UNCHECKED_CALL",
                        "unchecked_low_level_call",
                        "high",
                        "Appel bas niveau potentiellement non vérifié",
                        (
                            "Un appel .call(...) peut échouer sans annuler "
                            "automatiquement la transaction si sa valeur de "
                            "retour n'est pas contrôlée."
                        ),
                        (
                            "Capturer la valeur de retour et vérifier "
                            "explicitement le booléen success."
                        ),
                        line_number,
                        stripped,
                        "moyenne",
                    )
                )

        # ----------------------------------------------------
        # .send
        # ----------------------------------------------------
        if re.search(r"\.send\s*\(", stripped):
            external_call_lines.append(index)

            if not is_return_value_checked(stripped, "send"):
                add_finding(
                    make_finding(
                        "UNCHECKED_SEND",
                        "unchecked_low_level_call",
                        "medium",
                        "Valeur de retour de send potentiellement ignorée",
                        (
                            "send renvoie false en cas d'échec et cette "
                            "valeur doit être contrôlée."
                        ),
                        (
                            "Vérifier explicitement la valeur de retour avant "
                            "de poursuivre l'exécution."
                        ),
                        line_number,
                        stripped,
                        "moyenne",
                    )
                )

        # ----------------------------------------------------
        # .transfer
        # ----------------------------------------------------
        if re.search(r"\.transfer\s*\(", stripped):
            external_call_lines.append(index)

        # ----------------------------------------------------
        # timestamp
        # ----------------------------------------------------
        if re.search(r"\b(block\.timestamp|now)\b", stripped):
            add_finding(
                make_finding(
                    "TIMESTAMP_DEPENDENCY",
                    "time_manipulation",
                    "low",
                    "Dépendance au timestamp du bloc",
                    (
                        "Le timestamp du bloc ne doit pas être considéré "
                        "comme une source forte d'aléa ou une horloge "
                        "parfaitement précise."
                    ),
                    (
                        "Éviter son utilisation pour une logique de sécurité "
                        "critique ou comme unique source de hasard."
                    ),
                    line_number,
                    stripped,
                    "moyenne",
                )
            )

        # ----------------------------------------------------
        # block.number
        # ----------------------------------------------------
        if re.search(r"\bblock\.number\b", stripped):
            add_finding(
                make_finding(
                    "BLOCK_NUMBER_DEPENDENCY",
                    "block_dependency",
                    "low",
                    "Dépendance au numéro de bloc",
                    (
                        "block.number ne constitue ni une mesure exacte du "
                        "temps ni une source d'aléa forte."
                    ),
                    (
                        "Utiliser un mécanisme mieux adapté au besoin "
                        "fonctionnel lorsque le calcul exige le temps ou "
                        "l'aléa."
                    ),
                    line_number,
                    stripped,
                    "moyenne",
                )
            )

        # ----------------------------------------------------
        # pseudo-aléa faible
        # ----------------------------------------------------
        if (
            re.search(r"\bkeccak256\s*\(", stripped)
            and re.search(
                (
                    r"(block\.timestamp|"
                    r"blockhash\s*\(|"
                    r"block\.prevrandao|"
                    r"msg\.sender|"
                    r"block\.number)"
                ),
                stripped,
            )
        ):
            add_finding(
                make_finding(
                    "WEAK_RANDOMNESS",
                    "bad_randomness",
                    "high",
                    "Source de pseudo-aléa potentiellement prévisible",
                    (
                        "Le contrat semble générer une valeur pseudo-aléatoire "
                        "à partir de données de chaîne qui peuvent être "
                        "prévisibles ou influençables."
                    ),
                    (
                        "Utiliser une source de hasard vérifiable lorsque le "
                        "cas d'usage nécessite un véritable aléa."
                    ),
                    line_number,
                    stripped,
                    "élevée",
                )
            )

        # ----------------------------------------------------
        # assembly
        # ----------------------------------------------------
        if re.search(r"\bassembly\s*\{", stripped):
            add_finding(
                make_finding(
                    "INLINE_ASSEMBLY",
                    "audit_attention",
                    "medium",
                    "Bloc assembly détecté",
                    (
                        "Le code assembly contourne certaines garanties "
                        "offertes par Solidity et augmente la surface "
                        "d'audit."
                    ),
                    (
                        "Limiter l'assembly au strict nécessaire et documenter "
                        "précisément ses invariants."
                    ),
                    line_number,
                    stripped,
                    "élevée",
                )
            )

    # ========================================================
    # Réentrance potentielle
    # ========================================================
    for call_index in sorted(set(external_call_lines)):
        call_line = lines[call_index].strip()

        if not re.search(
            r"(\.call\s*(?:\{|\()|\.send\s*\(|\.transfer\s*\()",
            call_line,
        ):
            continue

        function_context = find_function_context(lines, call_index)
        protected = "nonReentrant" in function_context

        mutation_line = None
        mutation_text = None

        for next_index in range(
            call_index + 1,
            min(call_index + 12, len(lines)),
        ):
            next_line = lines[next_index].strip()

            if re.search(r"\bfunction\b", next_line):
                break

            if looks_like_state_mutation(next_line):
                mutation_line = next_index + 1
                mutation_text = next_line
                break

        if mutation_line is not None:
            severity = "medium" if protected else "high"

            add_finding(
                make_finding(
                    "POTENTIAL_REENTRANCY",
                    "reentrancy",
                    severity,
                    "Ordre appel externe / modification d'état à examiner",
                    (
                        "Un appel externe semble précéder une modification "
                        "d'état proche. Ce motif est compatible avec un "
                        "risque potentiel de réentrance."
                    ),
                    (
                        "Appliquer le modèle Checks-Effects-Interactions et "
                        "utiliser un garde nonReentrant lorsque cela est "
                        "pertinent."
                    ),
                    call_index + 1,
                    (
                        f"{call_line} -> ligne {mutation_line}: "
                        f"{mutation_text}"
                    ),
                    "moyenne",
                )
            )

    # ========================================================
    # Appel externe proche d'une boucle
    # ========================================================
    for loop_index in loop_lines:
        for index in range(
            loop_index,
            min(loop_index + 14, len(lines)),
        ):
            current_line = lines[index].strip()

            if (
                index != loop_index
                and re.search(r"\bfunction\b", current_line)
            ):
                break

            if re.search(
                (
                    r"(\.call\s*(?:\{|\()|"
                    r"\.send\s*\(|"
                    r"\.transfer\s*\(|"
                    r"\.delegatecall\s*(?:\{|\())"
                ),
                current_line,
            ):
                add_finding(
                    make_finding(
                        "EXTERNAL_CALL_IN_LOOP",
                        "denial_of_service",
                        "medium",
                        "Appel externe proche d'une boucle",
                        (
                            "Des appels externes répétés dans une boucle "
                            "peuvent augmenter fortement le coût en gas et "
                            "le risque d'échec."
                        ),
                        (
                            "Éviter les appels externes non bornés et préférer "
                            "un traitement limité, paginé ou de type pull."
                        ),
                        index + 1,
                        current_line,
                        "moyenne",
                    )
                )
                break

    # ========================================================
    # Résumé des sévérités
    # ========================================================
    severity_counts = Counter(
        finding["severity"]
        for finding in findings
    )

    static_score = sum(
        SEVERITY_POINTS.get(
            finding["severity"],
            0,
        )
        for finding in findings
    )

    static_score = int(clamp(static_score))

    # ========================================================
    # Score combiné IA + heuristiques
    # ========================================================
    ml_probability = normalize_probability(
        probability_vulnerable
    )

    combined_score = None
    combined_level = None

    # Aucun mélange de deux échelles non validées : les scores restent séparés.

    # ========================================================
    # Tri
    # ========================================================
    findings.sort(
        key=lambda finding: (
            SEVERITY_ORDER.get(
                finding["severity"],
                99,
            ),
            finding["line"]
            if finding["line"] is not None
            else 999999,
        )
    )

    return {
        "success": True,
        "engine": "SMART BUG Static Risk Analyzer v1",
        "static_analysis": {
            "score": static_score,
            "level": get_risk_level(static_score),
            "findings_count": len(findings),
            "severity_counts": {
                "critical": severity_counts.get("critical", 0),
                "high": severity_counts.get("high", 0),
                "medium": severity_counts.get("medium", 0),
                "low": severity_counts.get("low", 0),
                "info": severity_counts.get("info", 0),
            },
            "findings": findings,
        },
        "combined_analysis": {
            "available": combined_score is not None,
            "model_probability_vulnerable": ml_probability,
            "score": combined_score,
            "level": combined_level,
            "formula": (
                "Aucune combinaison : score IA calibré et indice heuristique séparés"
            ),
        },
        "metrics": {
            "lines": len(original_lines),
            "functions": function_count,
            "external_calls": len(set(external_call_lines)),
            "low_level_calls": len(set(low_level_call_lines)),
            "loops": len(loop_lines),
            "state_mutations_detected": state_mutations,
        },
        "disclaimer": (
            "Les catégories de vulnérabilité affichées dans cette section "
            "proviennent d'une analyse statique heuristique distincte du "
            "modèle BiLSTM. Le modèle IA V2 réalise uniquement une "
            "classification binaire : vulnérable ou non vulnérable."
        ),
    }


if __name__ == "__main__":
    sample_contract = """
    pragma solidity ^0.8.20;

    contract Demo {
        mapping(address => uint256) public balances;

        function withdraw() external {
            uint256 amount = balances[msg.sender];

            (bool success, ) = msg.sender.call{value: amount}("");
            require(success, "Transfer failed");

            balances[msg.sender] = 0;
        }
    }
    """

    result = analyze_contract_risks(
        sample_contract,
        probability_vulnerable=92.5,
    )

    print("=" * 60)
    print("SMART BUG - TEST DU RISK ANALYZER")
    print("=" * 60)
    print("Score statique :", result["static_analysis"]["score"])
    print("Niveau statique :", result["static_analysis"]["level"])
    print("Findings :", result["static_analysis"]["findings_count"])
    print("Score combiné :", result["combined_analysis"]["score"])
    print("Niveau combiné :", result["combined_analysis"]["level"])

    for finding in result["static_analysis"]["findings"]:
        print(
            f"- [{finding['severity'].upper()}] "
            f"L{finding['line']} "
            f"{finding['title']}"
        )
