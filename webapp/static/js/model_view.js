"use strict";

function modelEscapeHtml(value) {
    const node = document.createElement("div");
    node.textContent = value ?? "";
    return node.innerHTML;
}

function smartBugNavigate(label, pageClass, title) {
    document.querySelectorAll(".nav-item").forEach(item => {
        item.classList.toggle("active", item.textContent.toLowerCase().includes(label));
    });
    const container = document.querySelector(".content");
    container.innerHTML = `<section class="${pageClass}"><h1>${modelEscapeHtml(title)}</h1><p>Chargement des résultats du modèle actif…</p></section>`;
    return container.firstElementChild;
}

async function smartBugFetch(path) {
    const response = await fetch(path);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Résultats indisponibles");
    return data;
}

function smartBugPercent(value) {
    return value == null ? "Indisponible" : `${(Number(value) * 100).toFixed(2)} %`;
}

function smartBugCard(label, value, note = "") {
    return `<article class="model-metric-card"><span>${modelEscapeHtml(label)}</span><strong>${modelEscapeHtml(value)}</strong><small>${modelEscapeHtml(note)}</small></article>`;
}

async function showModelView() {
    const page = smartBugNavigate("modèle ia", "model-page", "Modèle IA SMART BUG");
    try {
        const report = await smartBugFetch("/api/model-info");
        if (!page.isConnected) return;
        const c = report.config;
        const metrics = report.evaluation.test.neural_calibrated;
        const baseline = report.evaluation.test.tfidf_baseline;
        const heldout = report.evaluation.source_holdout.neural_calibrated;
        page.innerHTML = `
            <div class="model-page-header"><div><span class="model-eyebrow">MODÈLE ACTIF · SMART BUG</span><h1>${modelEscapeHtml(report.architecture)}</h1><p>Entraîné depuis zéro, sans poids pré-entraînés ni fine-tuning.</p></div></div>
            <section class="model-metrics-grid">
                ${smartBugCard("Exactitude sur le test", smartBugPercent(metrics.accuracy), "Seuil choisi sur la calibration")}
                ${smartBugCard("F1 macro", smartBugPercent(metrics.f1_macro))}
                ${smartBugCard("Rappel des positifs", smartBugPercent(metrics.recall_vulnerable))}
                ${smartBugCard("Paramètres", report.selected.parameters.toLocaleString("fr-FR"))}
            </section>
            <section class="model-section"><h2>Architecture et apprentissage</h2><p>Vocabulaire de ${c.vocabulary_size} entrées, embeddings de dimension ${c.embedding_dim}, convolution de ${c.convolution_filters} filtres (noyau ${c.convolution_kernel}), puis BiLSTM de ${c.lstm_units} unités par direction sur les fenêtres de ${c.window_size} tokens. Toutes les fenêtres du contrat sont traitées ; le code n'est pas tronqué.</p><p>Adam : ${c.learning_rate} · batch de référence : ${c.batch_size} (réduit pour les longs contrats) · maximum ${c.epochs} époques · arrêt anticipé après ${c.early_stopping_patience} époques sans amélioration · dropout : ${c.dropout}.</p><p>Graines : ${c.seeds.join(", ")}. Graine sélectionnée sur la validation : ${report.selected.seed}, époque ${report.selected.best_epoch}. F1 macro de validation, moyenne ± écart-type : ${smartBugPercent(report.experiment.validation_f1_mean)} ± ${smartBugPercent(report.experiment.validation_f1_std)}.</p></section>
            <section class="model-section"><h2>Calibration et décision</h2><p>Température : ${report.calibration.temperature.toFixed(4)}. Seuil : ${smartBugPercent(report.calibration.threshold)}. Objectif de rappel sur la calibration : ${smartBugPercent(report.calibration.target_recall)} ; cet objectif n'est pas une garantie sur de nouveaux contrats.</p><p>Brier sur le test : ${metrics.brier_score.toFixed(4)} · erreur de calibration ECE : ${smartBugPercent(metrics.ece_10_bins)}. Les scores IA et heuristiques sont présentés séparément.</p></section>
            <section class="model-section"><h2>Comparaison et limites</h2><p>Référence TF-IDF + régression logistique : exactitude ${smartBugPercent(baseline.accuracy)}, F1 macro ${smartBugPercent(baseline.f1_macro)}. Source réservée : rappel positif ${smartBugPercent(heldout.recall_vulnerable)}. Ce dernier ensemble est presque exclusivement positif : il ne permet pas d'estimer solidement les faux positifs.</p><p>Les labels reflètent les annotations disponibles. Une absence de signal ne prouve pas la sécurité. Ces indicateurs correspondent au modèle SMART BUG actif ; la comparaison expérimentale des architectures est présentée séparément.</p></section>
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
        const report = await smartBugFetch("/api/model-info");
        const metrics = report.evaluation.test.neural_calibrated;
        document.querySelectorAll("[data-model-metric]").forEach(node => {
            node.textContent = smartBugPercent(metrics[node.dataset.modelMetric]);
        });
    } catch (_) {
        document.querySelectorAll("[data-model-metric]").forEach(node => { node.textContent = "Indisponible"; });
    }
}
window.refreshModelMetrics = refreshModelMetrics;
refreshModelMetrics();
