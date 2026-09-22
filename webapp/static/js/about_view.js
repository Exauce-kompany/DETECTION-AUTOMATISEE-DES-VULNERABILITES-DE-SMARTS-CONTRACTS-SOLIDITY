"use strict";

function showAboutView() {
    const page = v3Navigate("à propos", "model-page", "À propos de SMART BUG");
    page.innerHTML = `<div class="model-page-header"><div><span class="model-eyebrow">SMART BUG</span><h1>Aide à l'audit de contrats Solidity</h1></div></div>
        <section class="model-section"><h2>Trois analyses complémentaires</h2><p>Le modèle V3 classe le code complet à partir d'annotations de contrats. L'analyse statique signale des motifs de risque. L'analyse des performances fournit des indicateurs heuristiques. Ces résultats restent distincts.</p></section>
        <section class="model-section"><h2>Un protocole vérifiable</h2><p>Le modèle CNN + BiLSTM hiérarchique est entraîné depuis zéro. Les jeux d'entraînement, de validation, de calibration et de test sont séparés par familles de contrats. Une source est réservée pour mesurer le transfert. Les vues Modèle IA et Jeux de données lisent les résultats de l'expérience active.</p></section>
        <section class="model-section"><h2>Limites</h2><p>Une prédiction n'est ni une preuve de vulnérabilité ni une certification de sécurité. Les annotations peuvent rester imparfaites et le transfert vers de nouveaux projets doit être vérifié. Une revue du code et des tests spécialisés restent nécessaires avant déploiement.</p></section>`;
}
document.querySelectorAll(".nav-item").forEach(item => {
    if (item.textContent.toLowerCase().includes("à propos")) item.addEventListener("click", showAboutView);
});
window.showAboutView = showAboutView;
