/* Run with: node tests/frontend/report_view.test.js
 * The runner also works in a JavaScript engine with the source supplied as text.
 */
function runReportViewTests(source) {
    let checks = 0;
    function assert(condition, message) {
        checks += 1;
        if (!condition) throw new Error(message);
    }

    function render(data) {
        const container = { innerHTML: "" };
        const document = {
            querySelectorAll: () => [],
            querySelector: () => container,
            getElementById: () => null,
            createElement: () => ({
                textContent: "",
                get innerHTML() {
                    return String(this.textContent).replaceAll("&", "&amp;")
                        .replaceAll("<", "&lt;").replaceAll(">", "&gt;");
                },
            }),
        };
        const window = {};
        const sessionStorage = { getItem: () => JSON.stringify(data) };
        new Function("document", "window", "sessionStorage", source + "\nshowReportsView();")(
            document, window, sessionStorage,
        );
        return container.innerHTML.replace(/\s+/g, " ");
    }

    function fixture() {
        return {
            predicted_label: 0, confidence: 0.995, confidence_percent: 99.5,
            probability_vulnerable_percent: 0.5, probability_non_vulnerable_percent: 99.5,
            ml_risk_score: 0, ml_risk_level: "faible",
            static_risk_score: 0, static_risk_level: "faible",
            file: { filename: "contract.sol" }, preprocessing: {},
            risk_analysis: { success: true, static_analysis: { score: 0, findings: [] } },
            performance_analysis: { success: true, findings: [], summary: {
                efficiency_score: 0, complexity_score: 0, cost_pressure_score: 0,
                recommendations: [],
            } },
        };
    }

    function kpi(html, label) {
        const match = html.match(new RegExp(label + " </span> <strong[^>]*> ([^<]+)"));
        assert(Boolean(match), "Missing KPI: " + label);
        return match[1].trim();
    }

    let data = fixture();
    data.risk_analysis.static_analysis.findings = [{
        severity: "high", category: "reentrancy", line: 8,
        title: "ALERTE_REENTRANCE", recommendation: "Vérifier <owner>",
    }];
    let html = render(data);
    assert(html.includes("ALERTE_REENTRANCE"), "API static findings must be in the printable report");
    assert(kpi(html, "Findings") === "1", "Static finding count must match API findings");
    assert(html.includes("Vérifier &lt;owner&gt;"), "Finding text must remain escaped");
    assert(kpi(html, "CONFIANCE") === "99.50%", "Confidence must use percentage units");
    assert(!html.includes("RISQUE COMBINÉ"), "ML probability is not a combined risk score");

    html = render(fixture());
    for (const label of ["Risque statique", "Efficacité", "Complexité", "Pression de coût"]) {
        assert(kpi(html, label) === "0/100", "A real zero must stay zero: " + label);
    }
    assert(kpi(html, "Findings") === "0", "A successful empty analysis has zero findings");

    data = fixture();
    data.risk_analysis.success = false;
    data.performance_analysis.success = false;
    // Even stale scores/counts must not override a failed engine's status.
    data.risk_analysis.static_analysis.findings = [{ title: "STALE_FINDING" }];
    html = render(data);
    for (const label of ["Risque statique", "Efficacité", "Complexité", "Pression de coût"]) {
        assert(kpi(html, label) === "Indisponible", "Failed engine must not show a score: " + label);
    }
    assert(kpi(html, "Findings") === "—", "Failed static engine must not report zero findings");
    assert(kpi(html, "Findings performance") === "—", "Failed performance engine must not report zero findings");
    assert(!html.includes("STALE_FINDING"), "Failed engine must not show stale findings");
    assert(html.includes("Analyse statique indisponible"), "Missing static engine explanation");
    assert(html.includes("Analyse de performances indisponible"), "Missing performance engine explanation");

    data = fixture();
    data.static_risk_score = null;
    data.risk_analysis.static_analysis.score = null;
    data.performance_analysis.summary.efficiency_score = null;
    delete data.performance_analysis.summary.complexity_score;
    data.performance_analysis.summary.cost_pressure_score = "invalid";
    html = render(data);
    for (const label of ["Risque statique", "Efficacité", "Complexité", "Pression de coût"]) {
        assert(kpi(html, label) === "Indisponible", "Missing/invalid score must not become zero: " + label);
    }

    data = fixture();
    delete data.risk_analysis;
    delete data.performance_analysis;
    html = render(data);
    assert(kpi(html, "Findings") === "—", "Missing static engine is unavailable");
    assert(kpi(html, "Efficacité") === "Indisponible", "Missing performance engine is unavailable");
    assert(!/NaN|undefined|null/.test(html), "Missing data must not leak JavaScript values");
    return checks;
}

if (typeof module !== "undefined" && require.main === module) {
    const fs = require("node:fs");
    const path = require("node:path");
    const source = fs.readFileSync(path.join(__dirname, "../../webapp/static/js/report_view.js"), "utf8");
    console.log(`${runReportViewTests(source)} report rendering checks passed`);
}
