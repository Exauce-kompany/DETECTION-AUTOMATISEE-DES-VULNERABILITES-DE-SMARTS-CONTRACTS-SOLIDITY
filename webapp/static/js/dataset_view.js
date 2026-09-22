"use strict";

async function showDatasetView() {
    const page = v3Navigate("jeux de données", "dataset-page", "Jeux de données V3");
    try {
        const report = await v3Fetch("/api/dataset-info");
        if (!page.isConnected) return;
        const names = {train: "Entraînement", validation: "Validation", calibration: "Calibration", test: "Test final", source_holdout: "Source réservée"};
        page.innerHTML = `<div class="dataset-page-header"><div><span class="dataset-eyebrow">DATASET V3</span><h1>Données et contrôles d'indépendance</h1><p>Les familles de contrats restent dans une seule partition.</p></div></div>
            <section class="dataset-splits-grid">${Object.entries(report.splits).map(([key, value]) => `<article class="dataset-split-card"><h2>${names[key] || modelEscapeHtml(key)}</h2><strong>${value.samples.toLocaleString("fr-FR")} contrats</strong><p>${value.groups} groupes · label 0 : ${value.labels["0"] || 0} · label 1 : ${value.labels["1"] || 0}</p><p>${Object.entries(value.sources).map(([source, count]) => `${modelEscapeHtml(source)} : ${count}`).join(" · ")}</p></article>`).join("")}</section>
            <section class="dataset-section"><h2>Nettoyage traçable</h2><p>${report.removed_duplicate_samples} doublons retirés ; leur provenance est conservée.</p><ul>${Object.entries(report.quarantine).map(([reason, count]) => `<li>${modelEscapeHtml(reason)} : ${count}</li>`).join("")}</ul><p>Les annotations contradictoires, les fonctions sans contexte et les négatifs CGT insuffisamment évalués sont mis en quarantaine. Ils ne sont pas réétiquetés automatiquement.</p></section>
            <section class="dataset-section"><h2>Contrôles</h2><p>${report.passed ? "Contrôles réussis" : "Contrôles échoués"} : recouvrements vérifiés sur les identifiants, groupes, code canonique, tokens normalisés, structure et entrées encodées complètes.</p><p>Les commentaires sont retirés ; les identifiants sont normalisés. Le vocabulaire est appris uniquement sur l'entraînement. Les tokens inconnus sont encodés en octets sans perdre leur contenu.</p><p>Ces contrôles ne prouvent pas l'absence de tous les clones approximatifs. Les labels négatifs ne constituent pas une certification de sécurité.</p></section>
            <div class="dataset-footer-note" style="overflow-wrap:anywhere">Empreinte du dataset : ${modelEscapeHtml(report.dataset_id)}</div>`;
    } catch (error) {
        if (page.isConnected) page.innerHTML = `<h1>Jeux de données</h1><p>${modelEscapeHtml(error.message)}</p>`;
    }
}
document.querySelectorAll(".nav-item").forEach(item => {
    if (item.textContent.toLowerCase().includes("jeux de données")) item.addEventListener("click", showDatasetView);
});
window.showDatasetView = showDatasetView;
