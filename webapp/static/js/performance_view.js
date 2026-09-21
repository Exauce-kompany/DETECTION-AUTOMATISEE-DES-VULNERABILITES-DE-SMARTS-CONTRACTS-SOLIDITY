
"use strict";

/* =========================================================
   SMART BUG - VUE PERFORMANCES DU CONTRAT
   ========================================================= */

const smartBugPerformanceNav =
    Array.from(
        document.querySelectorAll(".nav-item")
    ).find(
        item =>
            item.textContent
                .trim()
                .toLowerCase()
                .includes("performances")
    );


function perfSafeNumber(value) {
    const number = Number(value);

    return Number.isFinite(number)
        ? number
        : 0;
}


function perfClamp(value, min = 0, max = 100) {
    return Math.min(
        max,
        Math.max(
            min,
            perfSafeNumber(value)
        )
    );
}


function perfEscapeHtml(value) {
    const div =
        document.createElement("div");

    div.textContent =
        value ?? "";

    return div.innerHTML;
}


function perfLevelLabel(level) {
    if (!level) {
        return "-";
    }

    return String(level)
        .toUpperCase();
}


function perfSeverityLabel(severity) {
    const labels = {
        critical: "CRITIQUE",
        high: "ÉLEVÉE",
        medium: "MOYENNE",
        low: "FAIBLE",
        info: "INFO"
    };

    return (
        labels[severity]
        ||
        String(
            severity || "INFO"
        ).toUpperCase()
    );
}


function perfCategoryLabel(category) {
    const labels = {
        loops: "Boucles",
        external_calls: "Appels externes",
        storage: "Stockage",
        dynamic_arrays: "Tableaux dynamiques",
        assembly: "Assembly",
        encoding_hashing: "Encodage / hachage",
        code_size: "Taille du code",
        data_location: "Emplacement des données",
        general: "Général"
    };

    return (
        labels[category]
        ||
        category
        ||
        "Autre"
    );
}


function perfGetStoredAnalysis() {
    try {

        const stored =
            sessionStorage.getItem(
                "smartBugLastAnalysis"
            );

        if (!stored) {
            return null;
        }

        return JSON.parse(stored);

    } catch (error) {

        console.warn(
            "SMART BUG : impossible de lire la dernière analyse.",
            error
        );

        return null;
    }
}


function perfSetActiveNavigation() {

    document
        .querySelectorAll(".nav-item")
        .forEach(
            item =>
                item.classList.remove(
                    "active"
                )
        );

    if (smartBugPerformanceNav) {
        smartBugPerformanceNav
            .classList
            .add("active");
    }
}


function perfGoToAnalysis() {

    if (
        typeof window.showAnalysisView
        ===
        "function"
    ) {

        window.showAnalysisView();

    } else {

        window.location.reload();
    }
}


function renderPerformanceEmptyState(
    container
) {

    container.innerHTML = `
        <section class="performance-page">

            <div class="performance-page-header">

                <div>
                    <span class="performance-eyebrow">
                        SMART BUG CONTRACT OPTIMIZER
                    </span>

                    <h1>
                        Performances du contrat
                    </h1>

                    <p>
                        Analyse heuristique de la complexité,
                        de l'efficacité et des indicateurs
                        potentiels de coût d'exécution.
                    </p>
                </div>

                <button
                    id="performanceRunAnalysis"
                    class="performance-primary-button"
                    type="button"
                >
                    Lancer une analyse
                </button>

            </div>

            <div class="performance-empty-state">

                <div class="performance-empty-icon">
                    ⚡
                </div>

                <h2>
                    Aucune analyse de performance disponible
                </h2>

                <p>
                    Analyse d'abord un contrat Solidity.
                    SMART BUG affichera ensuite ici ses indicateurs
                    d'efficacité, de complexité et d'optimisation.
                </p>

            </div>

        </section>
    `;


    const button =
        document.getElementById(
            "performanceRunAnalysis"
        );

    if (button) {
        button.addEventListener(
            "click",
            perfGoToAnalysis
        );
    }
}


function performanceScoreCard(
    title,
    score,
    level,
    subtitle,
    cssClass
) {

    const safeScore =
        perfClamp(score);

    return `
        <article
            class="performance-score-card ${cssClass}"
        >

            <div class="performance-score-top">
                <span>${perfEscapeHtml(title)}</span>
                <strong>${safeScore.toFixed(0)}/100</strong>
            </div>

            <div class="performance-score-track">
                <span
                    style="width:${safeScore}%"
                ></span>
            </div>

            <div class="performance-score-bottom">
                <strong>
                    ${perfEscapeHtml(
                        perfLevelLabel(level)
                    )}
                </strong>

                <small>
                    ${perfEscapeHtml(subtitle)}
                </small>
            </div>

        </article>
    `;
}


function performanceMetricCard(
    label,
    value,
    helper = ""
) {

    return `
        <article class="performance-metric-card">

            <span>
                ${perfEscapeHtml(label)}
            </span>

            <strong>
                ${perfEscapeHtml(value)}
            </strong>

            <small>
                ${perfEscapeHtml(helper)}
            </small>

        </article>
    `;
}


function performanceFindingCard(
    finding,
    index
) {

    const line =
        finding.line
            ? `Ligne ${finding.line}`
            : "Ligne non précisée";

    return `
        <article
            class="performance-finding-card ${perfEscapeHtml(
                finding.severity || "info"
            )}"
        >

            <div class="performance-finding-head">

                <div>
                    <span class="performance-finding-number">
                        #${index + 1}
                    </span>

                    <span
                        class="performance-severity-badge ${perfEscapeHtml(
                            finding.severity || "info"
                        )}"
                    >
                        ${perfEscapeHtml(
                            perfSeverityLabel(
                                finding.severity
                            )
                        )}
                    </span>

                    <span class="performance-category">
                        ${perfEscapeHtml(
                            perfCategoryLabel(
                                finding.category
                            )
                        )}
                    </span>
                </div>

                <small>
                    ${perfEscapeHtml(line)}
                </small>

            </div>

            <h3>
                ${perfEscapeHtml(
                    finding.title
                    ||
                    "Optimisation potentielle"
                )}
            </h3>

            <p>
                ${perfEscapeHtml(
                    finding.description
                    ||
                    "Aucune description disponible."
                )}
            </p>

            ${
                finding.evidence
                    ? `
                        <div class="performance-evidence">
                            <span>INDICE</span>

                            <code>
                                ${perfEscapeHtml(
                                    finding.evidence
                                )}
                            </code>
                        </div>
                    `
                    : ""
            }

            <div class="performance-recommendation-box">
                <span>
                    RECOMMANDATION
                </span>

                <p>
                    ${perfEscapeHtml(
                        finding.recommendation
                        ||
                        "Vérifier manuellement cette zone."
                    )}
                </p>
            </div>

        </article>
    `;
}


function showPerformanceView() {

    const container =
        document.querySelector(
            ".content"
        );

    if (!container) {
        return;
    }

    perfSetActiveNavigation();


    const data =
        perfGetStoredAnalysis();

    if (!data) {
        renderPerformanceEmptyState(
            container
        );
        return;
    }


    const performance =
        data.performance_analysis
        ||
        {};

    if (
        performance.success
        ===
        false
    ) {

        container.innerHTML = `
            <section class="performance-page">

                <div class="performance-page-header">

                    <div>
                        <span class="performance-eyebrow">
                            SMART BUG CONTRACT OPTIMIZER
                        </span>

                        <h1>
                            Performances du contrat
                        </h1>

                        <p>
                            Le moteur de performances n'a pas pu
                            analyser le dernier contrat.
                        </p>
                    </div>

                    <button
                        id="performanceBackAnalysis"
                        class="performance-primary-button"
                        type="button"
                    >
                        Nouvelle analyse
                    </button>

                </div>

                <div class="performance-error-state">
                    <strong>
                        Analyse de performance indisponible
                    </strong>

                    <p>
                        ${perfEscapeHtml(
                            performance.error
                            ||
                            "Erreur inconnue."
                        )}
                    </p>
                </div>

            </section>
        `;

        const button =
            document.getElementById(
                "performanceBackAnalysis"
            );

        if (button) {
            button.addEventListener(
                "click",
                perfGoToAnalysis
            );
        }

        return;
    }


    const summary =
        performance.summary
        ||
        {};

    const metrics =
        performance.metrics
        ||
        {};

    const efficiency =
        performance.efficiency
        ||
        {};

    const complexity =
        performance.complexity
        ||
        {};

    const costProfile =
        performance.cost_profile
        ||
        {};

    const findings =
        Array.isArray(
            performance.findings
        )
            ? performance.findings
            : [];

    const limitations =
        performance.limitations
        ||
        {};


    const filename =
        data.file?.filename
        ||
        "contract.sol";


    const efficiencyScore =
        perfSafeNumber(
            summary.efficiency_score
            ??
            efficiency.score
            ??
            data.performance_efficiency_score
        );

    const efficiencyLevel =
        summary.efficiency_level
        ||
        efficiency.level
        ||
        data.performance_efficiency_level
        ||
        "-";


    const complexityScore =
        perfSafeNumber(
            summary.complexity_score
            ??
            complexity.score
            ??
            data.performance_complexity_score
        );

    const complexityLevel =
        summary.complexity_level
        ||
        complexity.level
        ||
        data.performance_complexity_level
        ||
        "-";


    const overallPressure =
        costProfile.overall_pressure
        ||
        {};

    const storagePressure =
        costProfile.storage_pressure
        ||
        {};

    const executionPressure =
        costProfile.execution_pressure
        ||
        {};

    const codePressure =
        costProfile.code_size_pressure
        ||
        {};


    const costPressureScore =
        perfSafeNumber(
            summary.cost_pressure_score
            ??
            overallPressure.score
            ??
            data.performance_cost_pressure_score
        );

    const costPressureLevel =
        summary.cost_pressure_level
        ||
        overallPressure.level
        ||
        data.performance_cost_pressure_level
        ||
        "-";


    const recommendations =
        Array.isArray(
            summary.recommendations
        )
            ? summary.recommendations
            : [];


    container.innerHTML = `
        <section class="performance-page">

            <div class="performance-page-header">

                <div>
                    <span class="performance-eyebrow">
                        SMART BUG CONTRACT OPTIMIZER
                    </span>

                    <h1>
                        Performances du contrat
                    </h1>

                    <p>
                        Analyse heuristique du contrat
                        <strong>
                            ${perfEscapeHtml(filename)}
                        </strong>.
                    </p>
                </div>

                <button
                    id="performanceNewAnalysis"
                    class="performance-secondary-button"
                    type="button"
                >
                    Nouvelle analyse
                </button>

            </div>


            <section class="performance-hero-grid">

                <article class="performance-main-score">

                    <span class="performance-card-label">
                        EFFICACITÉ GLOBALE
                    </span>

                    <div
                        class="performance-score-ring"
                        style="
                            --performance-value:
                            ${perfClamp(
                                efficiencyScore
                            ) * 3.6}deg;
                        "
                    >
                        <div>
                            <strong>
                                ${Math.round(
                                    efficiencyScore
                                )}
                            </strong>
                            <small>/100</small>
                        </div>
                    </div>

                    <h2>
                        ${perfEscapeHtml(
                            perfLevelLabel(
                                efficiencyLevel
                            )
                        )}
                    </h2>

                    <p>
                        Score heuristique d'efficacité du contrat
                    </p>

                </article>


                <div class="performance-score-stack">

                    ${performanceScoreCard(
                        "Complexité",
                        complexityScore,
                        complexityLevel,
                        "Complexité structurelle approximative",
                        "complexity"
                    )}

                    ${performanceScoreCard(
                        "Pression de coût",
                        costPressureScore,
                        costPressureLevel,
                        "Indicateur relatif, pas un gas exact",
                        "cost"
                    )}

                </div>


                <article class="performance-engine-card">

                    <span class="performance-card-label">
                        MOTEUR
                    </span>

                    <strong>
                        ${perfEscapeHtml(
                            performance.engine
                            ||
                            "SMART BUG Performance Analyzer"
                        )}
                    </strong>

                    <p>
                        Analyse statique heuristique
                    </p>

                    <div class="performance-engine-flags">

                        <span>
                            Compilation
                            <b>
                                ${
                                    limitations.compiler_used
                                        ? "OUI"
                                        : "NON"
                                }
                            </b>
                        </span>

                        <span>
                            Gas exact
                            <b>
                                ${
                                    limitations.exact_gas
                                        ? "OUI"
                                        : "NON"
                                }
                            </b>
                        </span>

                        <span>
                            Bytecode
                            <b>
                                ${
                                    limitations.bytecode_analyzed
                                        ? "OUI"
                                        : "NON"
                                }
                            </b>
                        </span>

                    </div>

                </article>

            </section>


            <section class="performance-section">

                <div class="performance-section-heading">

                    <div>
                        <span>
                            MÉTRIQUES DU CONTRAT
                        </span>

                        <h2>
                            Profil structurel
                        </h2>
                    </div>

                </div>

                <div class="performance-metrics-grid">

                    ${performanceMetricCard(
                        "Fonctions",
                        metrics.functions ?? 0,
                        "Fonctions Solidity détectées"
                    )}

                    ${performanceMetricCard(
                        "Boucles",
                        metrics.loops ?? 0,
                        "for / while / do"
                    )}

                    ${performanceMetricCard(
                        "Appels externes",
                        metrics.external_calls ?? 0,
                        "call / transfer / send"
                    )}

                    ${performanceMetricCard(
                        "Écritures stockage",
                        metrics.estimated_state_writes ?? 0,
                        "Estimation heuristique"
                    )}

                    ${performanceMetricCard(
                        "Variables d'état",
                        metrics.estimated_state_variables ?? 0,
                        "Estimation"
                    )}

                    ${performanceMetricCard(
                        "Mappings",
                        metrics.mappings ?? 0,
                        "Structures mapping"
                    )}

                    ${performanceMetricCard(
                        "Assembly",
                        metrics.assembly_blocks ?? 0,
                        "Blocs assembly"
                    )}

                    ${performanceMetricCard(
                        "Lignes utiles",
                        metrics.lines_non_empty ?? 0,
                        "Lignes non vides"
                    )}

                </div>

            </section>


            <section class="performance-section">

                <div class="performance-section-heading">

                    <div>
                        <span>
                            PRESSION DE COÛT
                        </span>

                        <h2>
                            Indicateurs relatifs
                        </h2>
                    </div>

                    <small>
                        Ces scores ne représentent pas une mesure
                        exacte du gas.
                    </small>

                </div>


                <div class="performance-pressure-grid">

                    ${performanceScoreCard(
                        "Stockage",
                        storagePressure.score ?? 0,
                        storagePressure.level ?? "-",
                        "Écritures et structures persistantes",
                        "storage"
                    )}

                    ${performanceScoreCard(
                        "Exécution",
                        executionPressure.score ?? 0,
                        executionPressure.level ?? "-",
                        "Boucles, conditions et appels",
                        "execution"
                    )}

                    ${performanceScoreCard(
                        "Taille du code",
                        codePressure.score ?? 0,
                        codePressure.level ?? "-",
                        "Volume et structure du contrat",
                        "code-size"
                    )}

                </div>

            </section>


            <section class="performance-section">

                <div class="performance-section-heading">

                    <div>
                        <span>
                            OPTIMISATIONS
                        </span>

                        <h2>
                            Points à examiner
                        </h2>
                    </div>

                    <strong class="performance-findings-count">
                        ${findings.length}
                        ${
                            findings.length > 1
                                ? "findings"
                                : "finding"
                        }
                    </strong>

                </div>


                <div class="performance-findings-list">

                    ${
                        findings.length
                            ? findings
                                .map(
                                    (
                                        finding,
                                        index
                                    ) =>
                                        performanceFindingCard(
                                            finding,
                                            index
                                        )
                                )
                                .join("")
                            : `
                                <div class="performance-no-findings">
                                    Aucun point d'optimisation majeur
                                    détecté par les règles actives.
                                </div>
                            `
                    }

                </div>

            </section>


            <section class="performance-section">

                <div class="performance-section-heading">

                    <div>
                        <span>
                            RECOMMANDATIONS
                        </span>

                        <h2>
                            Actions suggérées
                        </h2>
                    </div>

                </div>


                <div class="performance-recommendations">

                    ${
                        recommendations.length
                            ? recommendations
                                .map(
                                    (
                                        recommendation,
                                        index
                                    ) => `
                                        <div class="performance-recommendation-item">

                                            <span>
                                                ${index + 1}
                                            </span>

                                            <p>
                                                ${perfEscapeHtml(
                                                    recommendation
                                                )}
                                            </p>

                                        </div>
                                    `
                                )
                                .join("")
                            : `
                                <div class="performance-no-findings">
                                    Aucune recommandation supplémentaire.
                                </div>
                            `
                    }

                </div>

            </section>


            <div class="performance-disclaimer">

                <strong>
                    Limite méthodologique
                </strong>

                <p>
                    ${perfEscapeHtml(
                        limitations.statement
                        ||
                        "Cette analyse est heuristique et ne mesure pas le gas réel."
                    )}
                </p>

            </div>

        </section>
    `;


    const newAnalysisButton =
        document.getElementById(
            "performanceNewAnalysis"
        );

    if (newAnalysisButton) {
        newAnalysisButton.addEventListener(
            "click",
            perfGoToAnalysis
        );
    }
}


if (smartBugPerformanceNav) {

    smartBugPerformanceNav
        .addEventListener(
            "click",
            showPerformanceView
        );
}


window.showPerformanceView =
    showPerformanceView;
