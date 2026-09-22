"use strict";

function modelEscapeHtml(value) {
    const node = document.createElement("div");
    node.textContent = value ?? "";
    return node.innerHTML;
}

function v3Navigate(label, pageClass, title) {
    document.querySelectorAll(".nav-item").forEach(item => {
        item.classList.toggle("active", item.textContent.toLowerCase().includes(label));
    });
    const container = document.querySelector(".content");
    container.innerHTML = `<section class="${pageClass}"><h1>${modelEscapeHtml(title)}</h1><p>Chargement des résultats du modèle actif…</p></section>`;
    return container.firstElementChild;
}

async function v3Fetch(path) {
    const response = await fetch(path);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Résultats indisponibles");
    return data;
}

function v3Percent(value) {
    return value == null ? "Indisponible" : `${(Number(value) * 100).toFixed(2)} %`;
}

function v3Card(label, value, note = "") {
    return `<article class="model-metric-card"><span>${modelEscapeHtml(label)}</span><strong>${modelEscapeHtml(value)}</strong><small>${modelEscapeHtml(note)}</small></article>`;
}

async function showModelView() {
    const page = v3Navigate("modèle ia", "model-page", "Modèle IA V3");
    try {
        const report = await v3Fetch("/api/model-info");
        if (!page.isConnected) return;
        const c = report.config;
        const metrics = report.evaluation.test.neural_calibrated;
        const baseline = report.evaluation.test.tfidf_baseline;
        const heldout = report.evaluation.source_holdout.neural_calibrated;
        page.innerHTML = `
            <div class="model-page-header"><div><span class="model-eyebrow">MODÈLE ACTIF · V3</span><h1>${modelEscapeHtml(report.architecture)}</h1><p>Entraîné depuis zéro, sans poids pré-entraînés ni fine-tuning.</p></div></div>
            <section class="model-metrics-grid">
                ${v3Card("Exactitude sur le test", v3Percent(metrics.accuracy), "Seuil choisi sur la calibration")}
                ${v3Card("F1 macro", v3Percent(metrics.f1_macro))}
                ${v3Card("Rappel des positifs", v3Percent(metrics.recall_vulnerable))}
                ${v3Card("Paramètres", report.selected.parameters.toLocaleString("fr-FR"))}
            </section>
            <section class="model-section"><h2>Architecture et apprentissage</h2><p>Vocabulaire de ${c.vocabulary_size} entrées, embeddings de dimension ${c.embedding_dim}, convolution de ${c.convolution_filters} filtres (noyau ${c.convolution_kernel}), puis BiLSTM de ${c.lstm_units} unités par direction sur les fenêtres de ${c.window_size} tokens. Toutes les fenêtres du contrat sont traitées ; le code n'est pas tronqué.</p><p>Adam : ${c.learning_rate} · batch de référence : ${c.batch_size} (réduit pour les longs contrats) · maximum ${c.epochs} époques · arrêt anticipé après ${c.early_stopping_patience} époques sans amélioration · dropout : ${c.dropout}.</p><p>Graines : ${c.seeds.join(", ")}. Graine sélectionnée sur la validation : ${report.selected.seed}, époque ${report.selected.best_epoch}. F1 macro de validation, moyenne ± écart-type : ${v3Percent(report.experiment.validation_f1_mean)} ± ${v3Percent(report.experiment.validation_f1_std)}.</p></section>
            <section class="model-section"><h2>Calibration et décision</h2><p>Température : ${report.calibration.temperature.toFixed(4)}. Seuil : ${v3Percent(report.calibration.threshold)}. Objectif de rappel sur la calibration : ${v3Percent(report.calibration.target_recall)} ; cet objectif n'est pas une garantie sur de nouveaux contrats.</p><p>Brier sur le test : ${metrics.brier_score.toFixed(4)} · erreur de calibration ECE : ${v3Percent(metrics.ece_10_bins)}. Les scores IA et heuristiques sont présentés séparément.</p></section>
            <section class="model-section"><h2>Comparaison et limites</h2><p>Référence TF-IDF + régression logistique : exactitude ${v3Percent(baseline.accuracy)}, F1 macro ${v3Percent(baseline.f1_macro)}. Source réservée : rappel positif ${v3Percent(heldout.recall_vulnerable)}. Ce dernier ensemble est presque exclusivement positif : il ne permet pas d'estimer solidement les faux positifs.</p><p>Les labels reflètent les annotations disponibles. Une absence de signal ne prouve pas la sécurité. Les résultats V2 restent des archives et ne sont pas directement comparables à ce nouveau protocole.</p></section>
            <div class="model-footer-note">Expérience : ${modelEscapeHtml(report.run_id)}</div>`;
    } catch (error) {
        if (page.isConnected) page.innerHTML = `<h1>Modèle IA</h1><p>${modelEscapeHtml(error.message)}</p>`;
    }
}

document.querySelectorAll(".nav-item").forEach(item => {
    if (item.textContent.toLowerCase().includes("modèle ia")) item.addEventListener("click", showModelView);
});
window.showModelView = showModelView;

async function refreshModelMetrics() {
    try {
        const report = await v3Fetch("/api/model-info");
        const metrics = report.evaluation.test.neural_calibrated;
        document.querySelectorAll("[data-model-metric]").forEach(node => {
            node.textContent = v3Percent(metrics[node.dataset.modelMetric]);
        });
    } catch (_) {
        document.querySelectorAll("[data-model-metric]").forEach(node => { node.textContent = "Indisponible"; });
    }
}
window.refreshModelMetrics = refreshModelMetrics;
refreshModelMetrics();
