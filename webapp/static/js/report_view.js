
"use strict";

/* =========================================================
   SMART BUG - RAPPORTS
   ========================================================= */

const smartBugReportsNav =
    Array.from(
        document.querySelectorAll(".nav-item")
    ).find(
        item =>
            item.textContent
                .trim()
                .toLowerCase()
                .includes("rapports")
    );


function reportEscapeHtml(value) {
    const div =
        document.createElement("div");

    div.textContent =
        value ?? "";

    return div.innerHTML;
}


function reportNumber(
    value,
    fallback = 0
) {
    const number =
        Number(value);

    return Number.isFinite(number)
        ? number
        : fallback;
}


function reportPercent(value) {
    return `${reportNumber(value).toFixed(2)}%`;
}


function reportSetActiveNavigation() {

    document
        .querySelectorAll(".nav-item")
        .forEach(
            item =>
                item.classList.remove(
                    "active"
                )
        );

    if (smartBugReportsNav) {
        smartBugReportsNav
            .classList
            .add("active");
    }
}


function reportGetLastAnalysis() {

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
            "SMART BUG : lecture du dernier rapport impossible.",
            error
        );

        return null;
    }
}


function reportGoToAnalysis() {

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


function reportRiskLabel(level) {

    if (!level) {
        return "-";
    }

    return String(level)
        .toUpperCase();
}


function reportFindingRows(
    findings
) {

    if (
        !Array.isArray(findings)
        ||
        findings.length === 0
    ) {

        return `
            <tr>
                <td colspan="5">
                    Aucun finding heuristique disponible.
                </td>
            </tr>
        `;
    }

    return findings
        .slice(0, 25)
        .map(
            finding => `
                <tr>
                    <td>
                        ${reportEscapeHtml(
                            String(
                                finding.severity
                                ||
                                "info"
                            ).toUpperCase()
                        )}
                    </td>

                    <td>
                        ${reportEscapeHtml(
                            finding.category
                            ||
                            "-"
                        )}
                    </td>

                    <td>
                        ${reportEscapeHtml(
                            finding.title
                            ||
                            "Finding"
                        )}
                    </td>

                    <td>
                        ${
                            finding.line
                                ? reportEscapeHtml(
                                    `L${finding.line}`
                                )
                                : "-"
                        }
                    </td>

                    <td>
                        ${reportEscapeHtml(
                            finding.recommendation
                            ||
                            "-"
                        )}
                    </td>
                </tr>
            `
        )
        .join("");
}


function reportRecommendationList(
    recommendations
) {

    if (
        !Array.isArray(recommendations)
        ||
        recommendations.length === 0
    ) {
        return `
            <li>
                Aucune recommandation supplémentaire.
            </li>
        `;
    }

    return recommendations
        .slice(0, 10)
        .map(
            item => `
                <li>
                    ${reportEscapeHtml(item)}
                </li>
            `
        )
        .join("");
}


function renderReportsEmptyState(
    container
) {

    container.innerHTML = `
        <section class="reports-page">

            <div class="reports-page-header">

                <div>
                    <span class="reports-eyebrow">
                        SMART BUG SECURITY REPORTING
                    </span>

                    <h1>
                        Rapports
                    </h1>

                    <p>
                        Générez un rapport à partir du dernier
                        contrat analysé.
                    </p>
                </div>

            </div>


            <div class="reports-empty-state">

                <div class="reports-empty-icon">
                    ▣
                </div>

                <h2>
                    Aucun rapport disponible
                </h2>

                <p>
                    Analysez d'abord un contrat Solidity.
                    Le rapport de sécurité pourra ensuite
                    être consulté, imprimé ou enregistré en PDF.
                </p>

                <button
                    id="reportsStartAnalysis"
                    class="reports-primary-button"
                    type="button"
                >
                    Lancer une analyse
                </button>

            </div>

        </section>
    `;


    const button =
        document.getElementById(
            "reportsStartAnalysis"
        );

    if (button) {
        button.addEventListener(
            "click",
            reportGoToAnalysis
        );
    }
}


function showReportsView() {

    const container =
        document.querySelector(
            ".content"
        );

    if (!container) {
        return;
    }

    reportSetActiveNavigation();


    const data =
        reportGetLastAnalysis();

    if (!data) {
        renderReportsEmptyState(
            container
        );
        return;
    }


    const file =
        data.file
        ||
        {};

    const preprocessing =
        data.preprocessing
        ||
        {};

    const risk =
        data.risk_analysis
        ||
        {};

    const riskSummary =
        risk.summary
        ||
        {};

    const riskFindings =
        Array.isArray(
            risk.findings
        )
            ? risk.findings
            : [];


    const performance =
        data.performance_analysis
        ||
        {};

    const performanceSummary =
        performance.summary
        ||
        {};

    const performanceFindings =
        Array.isArray(
            performance.findings
        )
            ? performance.findings
            : [];


    const recommendations =
        Array.isArray(
            performanceSummary.recommendations
        )
            ? performanceSummary.recommendations
            : [];


    const filename =
        file.filename
        ||
        data.filename
        ||
        "contract.sol";


    const vulnerable =
        Number(
            data.predicted_label
        ) === 1;


    const verdict =
        vulnerable
            ? "VULNÉRABLE"
            : "NON VULNÉRABLE";


    const confidence =
        reportNumber(
            data.confidence
        );


    const vulnerableProbability =
        reportNumber(
            data.probability_vulnerable_percent
            ??
            data.probability_vulnerable
        );


    const safeProbability =
        reportNumber(
            data.probability_non_vulnerable_percent
            ??
            data.probability_non_vulnerable
        );


    const combinedRisk =
        reportNumber(
            data.ml_risk_score ?? data.risk_score
        );


    const combinedRiskLevel =
        data.ml_risk_level
        ??
        data.risk_level
        ??
        "-";


    const staticRisk =
        reportNumber(
            data.static_risk_score
        );


    const staticRiskLevel =
        data.static_risk_level
        ??
        "-";


    const efficiency =
        reportNumber(
            performanceSummary.efficiency_score
            ??
            data.performance_efficiency_score
        );


    const efficiencyLevel =
        performanceSummary.efficiency_level
        ??
        data.performance_efficiency_level
        ??
        "-";


    const complexity =
        reportNumber(
            performanceSummary.complexity_score
            ??
            data.performance_complexity_score
        );


    const complexityLevel =
        performanceSummary.complexity_level
        ??
        data.performance_complexity_level
        ??
        "-";


    const costPressure =
        reportNumber(
            performanceSummary.cost_pressure_score
            ??
            data.performance_cost_pressure_score
        );


    const costPressureLevel =
        performanceSummary.cost_pressure_level
        ??
        data.performance_cost_pressure_level
        ??
        "-";


    const reportDate =
        new Date()
            .toLocaleString(
                "fr-FR"
            );


    const reportId =
        `SB-${Date.now()
            .toString()
            .slice(-8)}`;


    container.innerHTML = `
        <section class="reports-page">

            <div class="reports-page-header no-print">

                <div>
                    <span class="reports-eyebrow">
                        SMART BUG SECURITY REPORTING
                    </span>

                    <h1>
                        Rapports
                    </h1>

                    <p>
                        Rapport généré à partir du dernier
                        audit disponible dans cette session.
                    </p>
                </div>


                <div class="reports-actions">

                    <button
                        id="reportsNewAnalysis"
                        class="reports-secondary-button"
                        type="button"
                    >
                        Nouvelle analyse
                    </button>

                    <button
                        id="reportsPrint"
                        class="reports-primary-button"
                        type="button"
                    >
                        Imprimer / Enregistrer PDF
                    </button>

                </div>

            </div>


            <article
                id="smartBugPrintableReport"
                class="security-report"
            >

                <header class="security-report-header">

                    <div>

                        <span class="security-report-brand">
                            SMART BUG
                        </span>

                        <h2>
                            Rapport d'analyse de sécurité
                        </h2>

                        <p>
                            AI-Powered Smart Contract Security
                        </p>

                    </div>


                    <div class="security-report-meta">

                        <span>
                            RAPPORT
                            <strong>
                                ${reportEscapeHtml(reportId)}
                            </strong>
                        </span>

                        <span>
                            GÉNÉRÉ LE
                            <strong>
                                ${reportEscapeHtml(reportDate)}
                            </strong>
                        </span>

                    </div>

                </header>


                <section class="report-summary-grid">

                    <div>
                        <span>
                            CONTRAT
                        </span>
                        <strong>
                            ${reportEscapeHtml(filename)}
                        </strong>
                    </div>

                    <div>
                        <span>
                            VERDICT IA
                        </span>
                        <strong
                            class="${
                                vulnerable
                                    ? "danger"
                                    : "success"
                            }"
                        >
                            ${verdict}
                        </strong>
                    </div>

                    <div>
                        <span>
                            CONFIANCE
                        </span>
                        <strong>
                            ${confidence.toFixed(2)}%
                        </strong>
                    </div>

                    <div>
                        <span>
                            RISQUE COMBINÉ
                        </span>
                        <strong>
                            ${combinedRisk.toFixed(0)}/100
                        </strong>
                        <small>
                            ${reportEscapeHtml(
                                reportRiskLabel(
                                    combinedRiskLevel
                                )
                            )}
                        </small>
                    </div>

                </section>


                <section class="security-report-section">

                    <h3>
                        1. Résultat du modèle IA
                    </h3>

                    <div class="report-kpis">

                        <div>
                            <span>
                                Probabilité vulnérable
                            </span>
                            <strong>
                                ${reportPercent(
                                    vulnerableProbability
                                )}
                            </strong>
                        </div>

                        <div>
                            <span>
                                Probabilité non vulnérable
                            </span>
                            <strong>
                                ${reportPercent(
                                    safeProbability
                                )}
                            </strong>
                        </div>

                        <div>
                            <span>
                                Tokens utilisés
                            </span>
                            <strong>
                                ${reportEscapeHtml(
                                    preprocessing.tokens_used
                                    ??
                                    "-"
                                )}
                            </strong>
                        </div>

                        <div>
                            <span>
                                Troncature
                            </span>
                            <strong>
                                ${
                                    preprocessing.truncated
                                        ? "OUI"
                                        : "NON"
                                }
                            </strong>
                        </div>

                    </div>


                    <div class="report-method-note">

                        <strong>
                            Portée du modèle
                        </strong>

                        <p>
                            Le modèle CNN + BiLSTM V3 réalise uniquement
                            une classification binaire :
                            <b>vulnerable</b> ou
                            <b>non_vulnerable</b>.
                            Il ne prédit pas directement un type précis
                            de vulnérabilité.
                        </p>

                    </div>

                </section>


                <section class="security-report-section">

                    <h3>
                        2. Analyse des risques
                    </h3>


                    <div class="report-kpis">

                        <div>
                            <span>
                                Risque IA
                            </span>
                            <strong>
                                ${reportNumber(
                                    data.ml_risk_score
                                ).toFixed(0)}/100
                            </strong>
                        </div>

                        <div>
                            <span>
                                Risque statique
                            </span>
                            <strong>
                                ${staticRisk.toFixed(0)}/100
                            </strong>
                            <small>
                                ${reportEscapeHtml(
                                    reportRiskLabel(
                                        staticRiskLevel
                                    )
                                )}
                            </small>
                        </div>

                        <div>
                            <span>
                                Score IA
                            </span>
                            <strong>
                                ${combinedRisk.toFixed(0)}/100
                            </strong>
                        </div>

                        <div>
                            <span>
                                Findings
                            </span>
                            <strong>
                                ${riskFindings.length}
                            </strong>
                        </div>

                    </div>


                    <div class="report-table-wrapper">

                        <table class="report-table">

                            <thead>
                                <tr>
                                    <th>Sévérité</th>
                                    <th>Catégorie</th>
                                    <th>Finding</th>
                                    <th>Ligne</th>
                                    <th>Recommandation</th>
                                </tr>
                            </thead>

                            <tbody>
                                ${reportFindingRows(
                                    riskFindings
                                )}
                            </tbody>

                        </table>

                    </div>

                    <p class="report-footnote">
                        Les catégories détaillées ci-dessus
                        proviennent du moteur d'analyse statique
                        heuristique et non de la sortie directe
                        du BiLSTM.
                    </p>

                </section>


                <section class="security-report-section">

                    <h3>
                        3. Performances du contrat
                    </h3>


                    <div class="report-kpis">

                        <div>
                            <span>
                                Efficacité
                            </span>
                            <strong>
                                ${efficiency.toFixed(0)}/100
                            </strong>
                            <small>
                                ${reportEscapeHtml(
                                    reportRiskLabel(
                                        efficiencyLevel
                                    )
                                )}
                            </small>
                        </div>

                        <div>
                            <span>
                                Complexité
                            </span>
                            <strong>
                                ${complexity.toFixed(0)}/100
                            </strong>
                            <small>
                                ${reportEscapeHtml(
                                    reportRiskLabel(
                                        complexityLevel
                                    )
                                )}
                            </small>
                        </div>

                        <div>
                            <span>
                                Pression de coût
                            </span>
                            <strong>
                                ${costPressure.toFixed(0)}/100
                            </strong>
                            <small>
                                ${reportEscapeHtml(
                                    reportRiskLabel(
                                        costPressureLevel
                                    )
                                )}
                            </small>
                        </div>

                        <div>
                            <span>
                                Findings performance
                            </span>
                            <strong>
                                ${performanceFindings.length}
                            </strong>
                        </div>

                    </div>


                    <div class="report-table-wrapper">

                        <table class="report-table">

                            <thead>
                                <tr>
                                    <th>Sévérité</th>
                                    <th>Catégorie</th>
                                    <th>Finding</th>
                                    <th>Ligne</th>
                                    <th>Recommandation</th>
                                </tr>
                            </thead>

                            <tbody>
                                ${reportFindingRows(
                                    performanceFindings
                                )}
                            </tbody>

                        </table>

                    </div>


                    <p class="report-footnote">
                        Les indicateurs de performance sont
                        heuristiques. SMART BUG ne compile pas
                        le contrat dans cette étape et ne mesure
                        donc pas le gas exact.
                    </p>

                </section>


                <section class="security-report-section">

                    <h3>
                        4. Recommandations prioritaires
                    </h3>

                    <ol class="report-recommendation-list">
                        ${reportRecommendationList(
                            recommendations
                        )}
                    </ol>

                </section>


                <section class="security-report-section">

                    <h3>
                        5. Conclusion
                    </h3>

                    <p class="report-conclusion">
                        SMART BUG fournit une aide automatisée
                        à l'analyse des Smart Contracts Solidity.
                        Le verdict IA, les findings statiques et
                        les indicateurs de performance doivent
                        être interprétés conjointement.
                        Une validation manuelle par un spécialiste
                        reste recommandée avant tout déploiement
                        en environnement de production.
                    </p>

                </section>


                <footer class="security-report-footer">

                    <span>
                        SMART BUG
                    </span>

                    <span>
                        Rapport généré localement
                    </span>

                    <span>
                        CNN + BiLSTM V3 · Classification binaire
                    </span>

                </footer>

            </article>

        </section>
    `;


    const printButton =
        document.getElementById(
            "reportsPrint"
        );

    if (printButton) {
        printButton.addEventListener(
            "click",
            () => window.print()
        );
    }


    const newAnalysisButton =
        document.getElementById(
            "reportsNewAnalysis"
        );

    if (newAnalysisButton) {
        newAnalysisButton.addEventListener(
            "click",
            reportGoToAnalysis
        );
    }
}


if (smartBugReportsNav) {

    smartBugReportsNav.addEventListener(
        "click",
        showReportsView
    );
}


window.showReportsView =
    showReportsView;
