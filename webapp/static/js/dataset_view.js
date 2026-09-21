
"use strict";

/* =========================================================
   SMART BUG - VUE JEUX DE DONNÉES
   ========================================================= */

const smartBugDatasetNav =
    Array.from(
        document.querySelectorAll(".nav-item")
    ).find(
        item =>
            item.textContent
                .trim()
                .toLowerCase()
                .includes("jeux de données")
    );


function datasetSetActiveNavigation() {
    document
        .querySelectorAll(".nav-item")
        .forEach(
            item =>
                item.classList.remove("active")
        );

    if (smartBugDatasetNav) {
        smartBugDatasetNav
            .classList
            .add("active");
    }
}


function datasetStatCard(
    label,
    value,
    helper,
    cssClass = ""
) {
    return `
        <article class="dataset-stat-card ${cssClass}">
            <span>${label}</span>
            <strong>${value}</strong>
            <small>${helper}</small>
        </article>
    `;
}


function datasetSplitCard(
    title,
    total,
    safe,
    vulnerable,
    safePercent,
    vulnerablePercent,
    note,
    cssClass
) {
    return `
        <article class="dataset-split-card ${cssClass}">

            <div class="dataset-split-head">
                <div>
                    <span>${title}</span>
                    <strong>${total}</strong>
                </div>

                <small>${note}</small>
            </div>

            <div class="dataset-class-bar">
                <span
                    class="dataset-safe-part"
                    style="width:${safePercent}%"
                ></span>
                <span
                    class="dataset-vulnerable-part"
                    style="width:${vulnerablePercent}%"
                ></span>
            </div>

            <div class="dataset-class-legend">
                <div>
                    <i class="safe"></i>
                    <span>Non vulnérables</span>
                    <strong>${safe}</strong>
                    <small>${safePercent}%</small>
                </div>

                <div>
                    <i class="danger"></i>
                    <span>Vulnérables</span>
                    <strong>${vulnerable}</strong>
                    <small>${vulnerablePercent}%</small>
                </div>
            </div>

        </article>
    `;
}


function showDatasetView() {

    const container =
        document.querySelector(".content");

    if (!container) {
        return;
    }

    datasetSetActiveNavigation();

    container.innerHTML = `
        <section class="dataset-page">

            <div class="dataset-page-header">

                <div>
                    <span class="dataset-eyebrow">
                        SMART BUG DATA LAB
                    </span>

                    <h1>
                        Jeux de données
                    </h1>

                    <p>
                        Vue scientifique du dataset V2 utilisé pour
                        entraîner et évaluer le modèle BiLSTM de SMART BUG.
                    </p>
                </div>

                <div class="dataset-version-chip">
                    V2 + CGT
                </div>

            </div>


            <section class="dataset-overview-grid">

                ${datasetStatCard(
                    "TOTAL DES SPLITS",
                    "26 134",
                    "Train + validation + test",
                    "blue"
                )}

                ${datasetStatCard(
                    "TRAIN",
                    "19 104",
                    "73,10 % du total",
                    "purple"
                )}

                ${datasetStatCard(
                    "VALIDATION",
                    "4 667",
                    "17,86 % du total",
                    "cyan"
                )}

                ${datasetStatCard(
                    "TEST",
                    "2 363",
                    "9,04 % du total",
                    "green"
                )}

                ${datasetStatCard(
                    "VOCABULAIRE",
                    "5 000",
                    "Tokens retenus",
                    "orange"
                )}

                ${datasetStatCard(
                    "SÉQUENCE",
                    "512",
                    "Tokens maximum par contrat",
                    "blue"
                )}

            </section>


            <section class="dataset-section">

                <div class="dataset-section-heading">
                    <div>
                        <span>RÉPARTITION DES CLASSES</span>
                        <h2>Train · Validation · Test</h2>
                    </div>

                    <small>
                        Classe 0 = non_vulnerable · Classe 1 = vulnerable
                    </small>
                </div>

                <div class="dataset-splits-grid">

                    ${datasetSplitCard(
                        "TRAIN",
                        "19 104",
                        "9 182",
                        "9 922",
                        "48.06",
                        "51.94",
                        "Entraînement du modèle",
                        "train"
                    )}

                    ${datasetSplitCard(
                        "VALIDATION",
                        "4 667",
                        "2 298",
                        "2 369",
                        "49.24",
                        "50.76",
                        "Sélection / contrôle",
                        "validation"
                    )}

                    ${datasetSplitCard(
                        "TEST",
                        "2 363",
                        "1 153",
                        "1 210",
                        "48.79",
                        "51.21",
                        "Évaluation finale",
                        "test"
                    )}

                </div>

            </section>


            <section class="dataset-two-column">

                <article class="dataset-section">

                    <div class="dataset-section-heading">
                        <div>
                            <span>CONSTRUCTION V2</span>
                            <h2>Enrichissement du train</h2>
                        </div>
                    </div>

                    <div class="dataset-construction-flow">

                        <div class="dataset-source-node base">
                            <span>V1 TRAIN</span>
                            <strong>17 411</strong>
                            <small>Dataset de base conservé</small>
                        </div>

                        <div class="dataset-flow-arrow">
                            +
                        </div>

                        <div class="dataset-source-node cgt">
                            <span>AJOUTS CGT</span>
                            <strong>1 693</strong>
                            <small>Ajouts uniquement dans train</small>
                        </div>

                        <div class="dataset-flow-arrow">
                            =
                        </div>

                        <div class="dataset-source-node final">
                            <span>V2 TRAIN</span>
                            <strong>19 104</strong>
                            <small>Dataset final d'entraînement</small>
                        </div>

                    </div>

                    <div class="dataset-cgt-details">

                        <div>
                            <span>Positifs ajoutés</span>
                            <strong>1 190</strong>
                            <small>Contrats vulnérables CGT</small>
                        </div>

                        <div>
                            <span>Négatifs ajoutés</span>
                            <strong>503</strong>
                            <small>Candidats négatifs conservateurs</small>
                        </div>

                        <div>
                            <span>Conflits exclus</span>
                            <strong>142</strong>
                            <small>Conflits de labels CGT / V1</small>
                        </div>

                    </div>

                </article>


                <article class="dataset-section">

                    <div class="dataset-section-heading">
                        <div>
                            <span>PRÉTRAITEMENT</span>
                            <h2>Représentation du code</h2>
                        </div>
                    </div>

                    <div class="dataset-pipeline">

                        <div class="dataset-pipeline-step">
                            <b>01</b>
                            <div>
                                <strong>Code Solidity</strong>
                                <span>Contrat source .sol</span>
                            </div>
                        </div>

                        <div class="dataset-pipeline-arrow">↓</div>

                        <div class="dataset-pipeline-step">
                            <b>02</b>
                            <div>
                                <strong>Tokenisation</strong>
                                <span>Tokenizer Solidity du projet</span>
                            </div>
                        </div>

                        <div class="dataset-pipeline-arrow">↓</div>

                        <div class="dataset-pipeline-step">
                            <b>03</b>
                            <div>
                                <strong>Vocabulaire 5 000</strong>
                                <span>Encodage en identifiants entiers</span>
                            </div>
                        </div>

                        <div class="dataset-pipeline-arrow">↓</div>

                        <div class="dataset-pipeline-step">
                            <b>04</b>
                            <div>
                                <strong>Séquence 512</strong>
                                <span>Padding ou troncature</span>
                            </div>
                        </div>

                        <div class="dataset-pipeline-arrow">↓</div>

                        <div class="dataset-pipeline-step">
                            <b>05</b>
                            <div>
                                <strong>BiLSTM V2</strong>
                                <span>Classification binaire</span>
                            </div>
                        </div>

                    </div>

                </article>

            </section>


            <section class="dataset-section">

                <div class="dataset-section-heading">
                    <div>
                        <span>INTÉGRITÉ EXPÉRIMENTALE</span>
                        <h2>Contrôles appliqués</h2>
                    </div>
                </div>

                <div class="dataset-checks-grid">

                    <article class="dataset-check-card success">
                        <div>✓</div>
                        <strong>Validation et test inchangés</strong>
                        <p>
                            L'enrichissement CGT a été ajouté uniquement
                            au split d'entraînement afin de préserver
                            la comparabilité V1 → V2.
                        </p>
                    </article>

                    <article class="dataset-check-card success">
                        <div>✓</div>
                        <strong>Pas de chevauchement inter-splits détecté</strong>
                        <p>
                            L'audit de préparation a contrôlé les empreintes
                            de contenu entre train, validation et test.
                        </p>
                    </article>

                    <article class="dataset-check-card success">
                        <div>✓</div>
                        <strong>Labels binaires vérifiés</strong>
                        <p>
                            Les classes utilisées par le modèle sont
                            strictement 0 = non_vulnerable et
                            1 = vulnerable.
                        </p>
                    </article>

                    <article class="dataset-check-card success">
                        <div>✓</div>
                        <strong>Longueur des séquences vérifiée</strong>
                        <p>
                            Les entrées préparées respectent la longueur
                            fixe de 512 tokens attendue par le BiLSTM.
                        </p>
                    </article>

                </div>

            </section>


            <section class="dataset-two-column">

                <article class="dataset-section">

                    <div class="dataset-section-heading">
                        <div>
                            <span>ÉQUILIBRE</span>
                            <h2>Distribution globale</h2>
                        </div>
                    </div>

                    <div class="dataset-balance-card">

                        <div
                            class="dataset-balance-ring"
                            style="--safe-angle:174.7deg"
                        >
                            <div>
                                <strong>V2</strong>
                                <small>BINAIRE</small>
                            </div>
                        </div>

                        <div class="dataset-balance-legend">

                            <div>
                                <i class="safe"></i>
                                <span>Non vulnérable</span>
                                <strong>≈ 48,5 %</strong>
                            </div>

                            <div>
                                <i class="danger"></i>
                                <span>Vulnérable</span>
                                <strong>≈ 51,5 %</strong>
                            </div>

                            <p>
                                La répartition reste proche de l'équilibre,
                                ce qui limite la domination d'une classe.
                            </p>

                        </div>

                    </div>

                </article>


                <article class="dataset-section">

                    <div class="dataset-section-heading">
                        <div>
                            <span>PORTÉE SCIENTIFIQUE</span>
                            <h2>Ce que représente le dataset</h2>
                        </div>
                    </div>

                    <div class="dataset-scientific-note">

                        <p>
                            Le dataset V2 alimente un problème de
                            <strong>classification binaire</strong>.
                            Il apprend au modèle à distinguer les contrats
                            associés à la classe vulnérable de ceux associés
                            à la classe non vulnérable.
                        </p>

                        <p>
                            Les catégories précises de vulnérabilités
                            ne constituent pas les classes de sortie
                            du BiLSTM V2.
                        </p>

                        <div class="dataset-labels">
                            <span class="safe">
                                0 · non_vulnerable
                            </span>
                            <span class="danger">
                                1 · vulnerable
                            </span>
                        </div>

                    </div>

                </article>

            </section>


            <div class="dataset-footer-note">
                <strong>
                    SMART BUG · DATASET V2 + CGT
                </strong>

                <p>
                    Les chiffres affichés correspondent aux jeux effectivement
                    utilisés lors de l'entraînement V2 enregistré dans le projet.
                </p>
            </div>

        </section>
    `;
}


if (smartBugDatasetNav) {
    smartBugDatasetNav.addEventListener(
        "click",
        showDatasetView
    );
}


window.showDatasetView =
    showDatasetView;
