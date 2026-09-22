"use strict";

/* =========================================================
   SMART BUG - SYNTHÈSE DES RISQUES
   ========================================================= */

let smartBugLastAnalysis = null;


/* =========================================================
   CAPTURE DU DERNIER RÉSULTAT D'ANALYSE
   ========================================================= */

if (typeof window.displayResult === "function") {

    const originalDisplayResult =
        window.displayResult;

    window.displayResult =
        function (data) {

            smartBugLastAnalysis = data;

            try {
                sessionStorage.setItem(
                    "smartBugLastAnalysis",
                    JSON.stringify(data)
                );
            } catch (error) {
                console.warn(
                    "SMART BUG : impossible de sauvegarder le dernier résultat.",
                    error
                );
            }

            return originalDisplayResult(data);
        };
}


/* =========================================================
   RESTAURATION APRÈS RECHARGEMENT
   ========================================================= */

try {

    const stored =
        sessionStorage.getItem(
            "smartBugLastAnalysis"
        );

    if (stored) {
        smartBugLastAnalysis =
            JSON.parse(stored);
    }

} catch (error) {

    console.warn(
        "SMART BUG : impossible de restaurer le dernier résultat.",
        error
    );
}


/* =========================================================
   RECHERCHE DU BOUTON SIDEBAR
   ========================================================= */

const smartBugRiskNav =
    Array.from(
        document.querySelectorAll(
            ".nav-item"
        )
    ).find(
        item =>
            item.textContent
                .trim()
                .toLowerCase()
                .includes(
                    "synthèse des risques"
                )
    );


/* =========================================================
   UTILITAIRES
   ========================================================= */

function riskSafeNumber(
    value
) {

    const number =
        Number(value);

    return Number.isFinite(number)
        ? number
        : 0;
}


function riskClamp(
    value,
    min = 0,
    max = 100
) {

    return Math.min(
        max,
        Math.max(
            min,
            riskSafeNumber(value)
        )
    );
}


function riskEscapeHtml(
    value
) {

    const div =
        document.createElement(
            "div"
        );

    div.textContent =
        value ?? "";

    return div.innerHTML;
}


function riskSeverityLabel(
    severity
) {

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


function riskLevelLabel(
    level
) {

    if (!level) {
        return "-";
    }

    return String(level)
        .toUpperCase();
}


function riskCategoryLabel(
    category
) {

    const labels = {
        access_control:
            "Contrôle d'accès",

        external_call:
            "Appel externe",

        destructive_operation:
            "Opération destructive",

        unchecked_low_level_call:
            "Appel bas niveau",

        time_manipulation:
            "Manipulation temporelle",

        block_dependency:
            "Dépendance au bloc",

        bad_randomness:
            "Pseudo-aléa",

        audit_attention:
            "Attention d'audit",

        reentrancy:
            "Réentrance potentielle",

        denial_of_service:
            "Déni de service"
    };

    return (
        labels[category]
        ||
        category
        ||
        "Autre"
    );
}


/* =========================================================
   NAVIGATION ACTIVE
   ========================================================= */

function riskSetActiveNavigation() {

    document
        .querySelectorAll(
            ".nav-item"
        )
        .forEach(
            item =>
                item.classList.remove(
                    "active"
                )
        );

    if (smartBugRiskNav) {
        smartBugRiskNav.classList.add(
            "active"
        );
    }
}


/* =========================================================
   PAGE VIDE
   ========================================================= */

function renderRiskEmptyState(
    container
) {

    container.innerHTML = `
        <section class="risk-summary-page">

            <div class="risk-summary-header">

                <div>

                    <span class="risk-summary-eyebrow">
                        SMART BUG SECURITY CENTER
                    </span>

                    <h1>
                        Synthèse des risques
                    </h1>

                    <p>
                        Cette section présente séparément le score du modèle
                        CNN + BiLSTM V3 et l'analyse statique heuristique.
                    </p>

                </div>

                <button
                    id="riskGoToAnalysis"
                    class="risk-primary-button"
                    type="button"
                >
                    Lancer une analyse
                </button>

            </div>


            <div class="risk-empty-state">

                <div class="risk-empty-icon">
                    ◇
                </div>

                <h2>
                    Aucune analyse disponible
                </h2>

                <p>
                    Analyse d'abord un fichier Solidity .sol.
                    SMART BUG pourra ensuite afficher ici
                    la synthèse détaillée des risques.
                </p>

            </div>

        </section>
    `;


    const button =
        document.getElementById(
            "riskGoToAnalysis"
        );

    if (button) {

        button.addEventListener(
            "click",
            () => {

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
        );
    }
}


/* =========================================================
   CARTES DE SCORE
   ========================================================= */

function riskScoreCard(
    title,
    score,
    level,
    className,
    subtitle
) {

    const value =
        riskClamp(score);

    return `
        <article class="risk-score-card ${className}">

            <div class="risk-score-card-top">

                <span>
                    ${riskEscapeHtml(title)}
                </span>

                <strong>
                    ${Math.round(value)}/100
                </strong>

            </div>

            <div class="risk-score-track">

                <span
                    style="width:${value}%"
                ></span>

            </div>

            <div class="risk-score-card-bottom">

                <strong>
                    ${riskEscapeHtml(
                        riskLevelLabel(level)
                    )}
                </strong>

                <small>
                    ${riskEscapeHtml(subtitle)}
                </small>

            </div>

        </article>
    `;
}


/* =========================================================
   FINDING
   ========================================================= */

function riskFindingCard(
    finding,
    index
) {

    const line =
        finding.line
            ? `Ligne ${finding.line}`
            : "Ligne non précisée";

    return `
        <article class="risk-finding-card ${riskEscapeHtml(
            finding.severity || "info"
        )}">

            <div class="risk-finding-head">

                <div>

                    <span class="risk-finding-number">
                        #${index + 1}
                    </span>

                    <span class="risk-severity-badge ${riskEscapeHtml(
                        finding.severity || "info"
                    )}">
                        ${riskEscapeHtml(
                            riskSeverityLabel(
                                finding.severity
                            )
                        )}
                    </span>

                    <span class="risk-category-badge">
                        ${riskEscapeHtml(
                            riskCategoryLabel(
                                finding.category
                            )
                        )}
                    </span>

                </div>

                <small>
                    ${riskEscapeHtml(line)}
                </small>

            </div>


            <h3>
                ${riskEscapeHtml(
                    finding.title
                    ||
                    "Risque détecté"
                )}
            </h3>


            <p>
                ${riskEscapeHtml(
                    finding.description
                    ||
                    "Aucune description disponible."
                )}
            </p>


            ${
                finding.evidence
                    ? `
                        <div class="risk-evidence">

                            <span>
                                INDICE DANS LE CODE
                            </span>

                            <code>
                                ${riskEscapeHtml(
                                    finding.evidence
                                )}
                            </code>

                        </div>
                    `
                    : ""
            }


            <div class="risk-recommendation">

                <span>
                    RECOMMANDATION
                </span>

                <p>
                    ${riskEscapeHtml(
                        finding.recommendation
                        ||
                        "Vérifier manuellement cette zone du contrat."
                    )}
                </p>

            </div>

        </article>
    `;
}


/* =========================================================
   RENDU PRINCIPAL
   ========================================================= */

function showRiskSummaryView() {

    const container =
        document.querySelector(
            ".content"
        );

    if (!container) {
        return;
    }

    riskSetActiveNavigation();


    if (!smartBugLastAnalysis) {

        renderRiskEmptyState(
            container
        );

        return;
    }


    const data =
        smartBugLastAnalysis;


    const riskAnalysis =
        data.risk_analysis
        ||
        {};


    const staticAnalysis =
        riskAnalysis.static_analysis
        ||
        {};


    const combinedAnalysis =
        riskAnalysis.combined_analysis
        ||
        {};


    const metrics =
        riskAnalysis.metrics
        ||
        {};


    const severityCounts =
        staticAnalysis.severity_counts
        ||
        {};


    const findings =
        Array.isArray(
            staticAnalysis.findings
        )
            ? staticAnalysis.findings
            : [];


    const filename =
        data.file?.filename
        ||
        "contract.sol";


    const predictedLabel =
        Number(
            data.predicted_label
        );


    const verdict =
        predictedLabel === 1
            ? "VULNÉRABLE"
            : "NON VULNÉRABLE";


    const probabilityVulnerable =
        riskSafeNumber(
            data.probability_vulnerable_percent
        );


    const mlScore =
        riskSafeNumber(
            data.ml_risk_score
        );


    const staticScore =
        riskSafeNumber(
            data.static_risk_score
        );


    const combinedScore =
        riskSafeNumber(
            data.ml_risk_score ?? data.risk_score
        );


    container.innerHTML = `
        <section class="risk-summary-page">

            <div class="risk-summary-header">

                <div>

                    <span class="risk-summary-eyebrow">
                        SMART BUG SECURITY CENTER
                    </span>

                    <h1>
                        Synthèse des risques
                    </h1>

                    <p>
                        Analyse du contrat
                        <strong>${riskEscapeHtml(filename)}</strong>.
                    </p>

                </div>


                <div class="risk-header-actions">

                    <button
                        id="riskBackToAnalysis"
                        class="risk-secondary-button"
                        type="button"
                    >
                        Nouvelle analyse
                    </button>

                </div>

            </div>


            <section class="risk-overview-grid">

                <article class="risk-main-card">

                    <span class="risk-card-label">
                        SCORE IA
                    </span>

                    <div
                        class="risk-main-ring"
                        style="
                            --risk-value:
                            ${riskClamp(
                                combinedScore
                            ) * 3.6}deg;
                        "
                    >

                        <div>

                            <strong>
                                ${Math.round(
                                    combinedScore
                                )}
                            </strong>

                            <small>
                                /100
                            </small>

                        </div>

                    </div>

                    <h2>
                        ${riskEscapeHtml(
                            riskLevelLabel(
                                data.ml_risk_level || data.risk_level
                            )
                        )}
                    </h2>

                    <p>
                        Score IA ; les alertes statiques restent distinctes
                    </p>

                </article>


                <article class="risk-verdict-card">

                    <span class="risk-card-label">
                        VERDICT DU MODÈLE IA
                    </span>

                    <strong class="${
                        predictedLabel === 1
                            ? "danger"
                            : "success"
                    }">
                        ${verdict}
                    </strong>

                    <div class="risk-verdict-probability">

                        <span>
                            Probabilité vulnérable
                        </span>

                        <b>
                            ${probabilityVulnerable.toFixed(2)}%
                        </b>

                    </div>

                    <div class="risk-verdict-track">

                        <span
                            style="
                                width:
                                ${riskClamp(
                                    probabilityVulnerable
                                )}%;
                            "
                        ></span>

                    </div>

                    <small>
                        Classification binaire CNN + BiLSTM V3
                    </small>

                </article>


                <article class="risk-alert-card">

                    <span class="risk-card-label">
                        FINDINGS STATIQUES
                    </span>

                    <strong>
                        ${findings.length}
                    </strong>

                    <p>
                        Motif(s) de risque détecté(s)
                    </p>

                    <div class="risk-severity-mini-grid">

                        <span class="critical">
                            C
                            <b>
                                ${riskSafeNumber(
                                    severityCounts.critical
                                )}
                            </b>
                        </span>

                        <span class="high">
                            H
                            <b>
                                ${riskSafeNumber(
                                    severityCounts.high
                                )}
                            </b>
                        </span>

                        <span class="medium">
                            M
                            <b>
                                ${riskSafeNumber(
                                    severityCounts.medium
                                )}
                            </b>
                        </span>

                        <span class="low">
                            L
                            <b>
                                ${riskSafeNumber(
                                    severityCounts.low
                                )}
                            </b>
                        </span>

                    </div>

                </article>

            </section>


            <section class="risk-score-grid">

                ${riskScoreCard(
                    "Score IA",
                    mlScore,
                    data.ml_risk_level,
                    "ai",
                    "Score IA V3 calibré sur un jeu dédié"
                )}

                ${riskScoreCard(
                    "Score statique",
                    staticScore,
                    data.static_risk_level,
                    "static",
                    "Analyse heuristique des motifs de code"
                )}

            </section>


            <section class="risk-metrics-panel">

                <div class="risk-section-heading">

                    <div>

                        <span>
                            MÉTRIQUES DU CONTRAT
                        </span>

                        <h2>
                            Surface technique analysée
                        </h2>

                    </div>

                </div>


                <div class="risk-metrics-grid">

                    <div>
                        <span>
                            Lignes
                        </span>

                        <strong>
                            ${riskSafeNumber(
                                metrics.lines
                            )}
                        </strong>
                    </div>

                    <div>
                        <span>
                            Fonctions
                        </span>

                        <strong>
                            ${riskSafeNumber(
                                metrics.functions
                            )}
                        </strong>
                    </div>

                    <div>
                        <span>
                            Appels externes
                        </span>

                        <strong>
                            ${riskSafeNumber(
                                metrics.external_calls
                            )}
                        </strong>
                    </div>

                    <div>
                        <span>
                            Appels bas niveau
                        </span>

                        <strong>
                            ${riskSafeNumber(
                                metrics.low_level_calls
                            )}
                        </strong>
                    </div>

                    <div>
                        <span>
                            Boucles
                        </span>

                        <strong>
                            ${riskSafeNumber(
                                metrics.loops
                            )}
                        </strong>
                    </div>

                    <div>
                        <span>
                            Mutations d'état
                        </span>

                        <strong>
                            ${riskSafeNumber(
                                metrics.state_mutations_detected
                            )}
                        </strong>
                    </div>

                </div>

            </section>


            <section class="risk-findings-panel">

                <div class="risk-section-heading">

                    <div>

                        <span>
                            ANALYSE STATIQUE
                        </span>

                        <h2>
                            Risques et recommandations
                        </h2>

                    </div>

                    <strong class="risk-findings-count">
                        ${findings.length}
                        finding(s)
                    </strong>

                </div>


                ${
                    findings.length > 0

                        ? `
                            <div class="risk-findings-list">

                                ${findings
                                    .map(
                                        (
                                            finding,
                                            index
                                        ) =>
                                            riskFindingCard(
                                                finding,
                                                index
                                            )
                                    )
                                    .join("")}

                            </div>
                        `

                        : `
                            <div class="risk-no-findings">

                                <div>
                                    ✓
                                </div>

                                <h3>
                                    Aucun motif statique important détecté
                                </h3>

                                <p>
                                    Cette absence de findings ne garantit
                                    pas que le contrat est exempt de
                                    vulnérabilités.
                                </p>

                            </div>
                        `
                }

            </section>


            <section class="risk-disclaimer-panel">

                <strong>
                    Distinction importante
                </strong>

                <p>
                    ${
                        riskEscapeHtml(
                            riskAnalysis.disclaimer
                            ||
                            "Le CNN + BiLSTM V3 effectue uniquement une classification binaire. Les catégories de risques proviennent du moteur statique heuristique."
                        )
                    }
                </p>

            </section>

        </section>
    `;


    const backButton =
        document.getElementById(
            "riskBackToAnalysis"
        );


    if (backButton) {

        backButton.addEventListener(
            "click",
            () => {

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
        );
    }
}


/* =========================================================
   ÉVÉNEMENT SIDEBAR
   ========================================================= */

if (smartBugRiskNav) {

    smartBugRiskNav.addEventListener(
        "click",
        event => {

            event.preventDefault();

            showRiskSummaryView();
        }
    );
}
