
"use strict";

/* =========================================================
   SMART BUG - VUE MODÈLE IA
   ========================================================= */

const smartBugModelNav =
    Array.from(
        document.querySelectorAll(".nav-item")
    ).find(
        item =>
            item.textContent
                .trim()
                .toLowerCase()
                .includes("modèle ia")
    );


function modelEscapeHtml(value) {
    const div =
        document.createElement("div");

    div.textContent =
        value ?? "";

    return div.innerHTML;
}


function modelSetActiveNavigation() {

    document
        .querySelectorAll(".nav-item")
        .forEach(
            item =>
                item.classList.remove(
                    "active"
                )
        );

    if (smartBugModelNav) {
        smartBugModelNav
            .classList
            .add("active");
    }
}


function modelMetricCard(
    label,
    value,
    helper,
    cssClass = ""
) {

    return `
        <article class="model-metric-card ${cssClass}">

            <span>
                ${modelEscapeHtml(label)}
            </span>

            <strong>
                ${modelEscapeHtml(value)}
            </strong>

            <small>
                ${modelEscapeHtml(helper)}
            </small>

        </article>
    `;
}


function architectureLayer(
    index,
    title,
    type,
    shape,
    params,
    description
) {

    return `
        <article class="model-layer-card">

            <div class="model-layer-index">
                ${index}
            </div>

            <div class="model-layer-body">

                <div class="model-layer-head">

                    <div>
                        <strong>
                            ${modelEscapeHtml(title)}
                        </strong>

                        <span>
                            ${modelEscapeHtml(type)}
                        </span>
                    </div>

                    <b>
                        ${modelEscapeHtml(params)}
                    </b>

                </div>

                <div class="model-layer-shape">
                    Sortie :
                    <code>
                        ${modelEscapeHtml(shape)}
                    </code>
                </div>

                <p>
                    ${modelEscapeHtml(description)}
                </p>

            </div>

        </article>
    `;
}


function showModelView() {

    const container =
        document.querySelector(
            ".content"
        );

    if (!container) {
        return;
    }

    modelSetActiveNavigation();

    container.innerHTML = `
        <section class="model-page">

            <div class="model-page-header">

                <div>
                    <span class="model-eyebrow">
                        SMART BUG DEEP LEARNING LAB
                    </span>

                    <h1>
                        Modèle IA — BiLSTM V2
                    </h1>

                    <p>
                        Modèle d'apprentissage profond utilisé par SMART BUG
                        pour la classification binaire des Smart Contracts Solidity.
                    </p>
                </div>

                <div class="model-status-badge">
                    <span></span>
                    MODÈLE ACTIF
                </div>

            </div>


            <section class="model-hero-grid">

                <article class="model-identity-card">

                    <div class="model-brain-icon">
                        ◉
                    </div>

                    <span class="model-card-label">
                        MODÈLE ACTUEL
                    </span>

                    <h2>
                        BiLSTM V2
                    </h2>

                    <p>
                        smart_contract_vulnerability_model_v2.keras
                    </p>

                    <div class="model-identity-tags">
                        <span>TensorFlow 2.21.0</span>
                        <span>Keras</span>
                        <span>Classification binaire</span>
                    </div>

                </article>


                <div class="model-metrics-grid">

                    ${modelMetricCard(
                        "Accuracy",
                        "85.87%",
                        "Évaluation sur le jeu de test",
                        "success"
                    )}

                    ${modelMetricCard(
                        "F1 Macro",
                        "85.86%",
                        "Équilibre global entre précision et rappel",
                        "purple"
                    )}

                    ${modelMetricCard(
                        "Precision Macro",
                        "85.99%",
                        "Précision moyenne des deux classes",
                        "blue"
                    )}

                    ${modelMetricCard(
                        "Recall Macro",
                        "85.94%",
                        "Rappel moyen des deux classes",
                        "cyan"
                    )}

                </div>


                <article class="model-spec-card">

                    <span class="model-card-label">
                        SPÉCIFICATIONS
                    </span>

                    <div class="model-spec-list">

                        <div>
                            <span>Paramètres</span>
                            <strong>346 978</strong>
                        </div>

                        <div>
                            <span>Vocabulaire</span>
                            <strong>5 000</strong>
                        </div>

                        <div>
                            <span>Séquence</span>
                            <strong>512 tokens</strong>
                        </div>

                        <div>
                            <span>Classes</span>
                            <strong>2</strong>
                        </div>

                        <div>
                            <span>Meilleur epoch</span>
                            <strong>4</strong>
                        </div>

                    </div>

                </article>

            </section>


            <section class="model-section">

                <div class="model-section-heading">

                    <div>
                        <span>
                            ARCHITECTURE
                        </span>

                        <h2>
                            Pipeline neuronal
                        </h2>
                    </div>

                    <small>
                        346 978 paramètres entraînables
                    </small>

                </div>


                <div class="model-architecture">

                    ${architectureLayer(
                        "01",
                        "Tokens",
                        "InputLayer",
                        "(None, 512)",
                        "0",
                        "Séquence tokenisée du code Solidity, limitée ou complétée à 512 positions."
                    )}

                    <div class="model-layer-arrow">↓</div>

                    ${architectureLayer(
                        "02",
                        "Embedding",
                        "Embedding",
                        "(None, 512, 64)",
                        "320 000",
                        "Projection des identifiants de tokens dans un espace vectoriel de dimension 64."
                    )}

                    <div class="model-layer-arrow">↓</div>

                    ${architectureLayer(
                        "03",
                        "Bidirectional LSTM",
                        "BiLSTM",
                        "(None, 512, 64)",
                        "24 832",
                        "Lecture bidirectionnelle de la séquence afin de capturer les dépendances contextuelles dans le code."
                    )}

                    <div class="model-layer-arrow">↓</div>

                    ${architectureLayer(
                        "04",
                        "Global Average Pooling",
                        "GlobalAveragePooling1D",
                        "(None, 64)",
                        "0",
                        "Agrégation de la représentation séquentielle en un vecteur compact."
                    )}

                    <div class="model-layer-arrow">↓</div>

                    ${architectureLayer(
                        "05",
                        "Dense",
                        "Dense + ReLU",
                        "(None, 32)",
                        "2 080",
                        "Couche entièrement connectée utilisée pour apprendre une représentation de décision."
                    )}

                    <div class="model-layer-arrow">↓</div>

                    ${architectureLayer(
                        "06",
                        "Classification",
                        "Dense + Softmax",
                        "(None, 2)",
                        "66",
                        "Produit les probabilités des classes non_vulnerable et vulnerable."
                    )}

                </div>


                <div class="model-dropout-note">
                    <strong>
                        Régularisation
                    </strong>

                    <p>
                        Des couches Dropout sont placées après le pooling global
                        et après la couche Dense afin de réduire le surapprentissage.
                    </p>
                </div>

            </section>


            <section class="model-two-column">

                <article class="model-section">

                    <div class="model-section-heading">

                        <div>
                            <span>
                                MATRICE DE CONFUSION
                            </span>

                            <h2>
                                Résultats sur le test
                            </h2>
                        </div>

                        <small>
                            2 363 contrats
                        </small>

                    </div>


                    <div class="model-confusion-wrapper">

                        <div class="model-confusion-y-label">
                            RÉEL
                        </div>

                        <div class="model-confusion-main">

                            <div class="model-confusion-x-label">
                                PRÉDIT
                            </div>

                            <div class="model-confusion-grid">

                                <div class="model-confusion-corner">
                                </div>

                                <div class="model-confusion-header">
                                    Non vuln.
                                </div>

                                <div class="model-confusion-header">
                                    Vuln.
                                </div>

                                <div class="model-confusion-header side">
                                    Non vuln.
                                </div>

                                <div class="model-confusion-cell correct">
                                    <strong>1026</strong>
                                    <span>Vrais négatifs</span>
                                </div>

                                <div class="model-confusion-cell error">
                                    <strong>127</strong>
                                    <span>Faux positifs</span>
                                </div>

                                <div class="model-confusion-header side">
                                    Vuln.
                                </div>

                                <div class="model-confusion-cell error">
                                    <strong>207</strong>
                                    <span>Faux négatifs</span>
                                </div>

                                <div class="model-confusion-cell correct">
                                    <strong>1003</strong>
                                    <span>Vrais positifs</span>
                                </div>

                            </div>

                        </div>

                    </div>

                </article>


                <article class="model-section">

                    <div class="model-section-heading">

                        <div>
                            <span>
                                DONNÉES D'ENTRAÎNEMENT
                            </span>

                            <h2>
                                Répartition V2
                            </h2>
                        </div>

                    </div>


                    <div class="model-split-list">

                        <div class="model-split-item">

                            <div class="model-split-head">
                                <span>TRAIN</span>
                                <strong>19 104</strong>
                            </div>

                            <div class="model-split-track">
                                <span style="width:73.1%"></span>
                            </div>

                            <p>
                                9 182 non vulnérables · 9 922 vulnérables
                            </p>

                        </div>


                        <div class="model-split-item">

                            <div class="model-split-head">
                                <span>VALIDATION</span>
                                <strong>4 667</strong>
                            </div>

                            <div class="model-split-track validation">
                                <span style="width:17.9%"></span>
                            </div>

                            <p>
                                2 298 non vulnérables · 2 369 vulnérables
                            </p>

                        </div>


                        <div class="model-split-item">

                            <div class="model-split-head">
                                <span>TEST</span>
                                <strong>2 363</strong>
                            </div>

                            <div class="model-split-track test">
                                <span style="width:9%"></span>
                            </div>

                            <p>
                                1 153 non vulnérables · 1 210 vulnérables
                            </p>

                        </div>

                    </div>


                    <div class="model-dataset-total">

                        <span>
                            TOTAL V2
                        </span>

                        <strong>
                            26 134
                        </strong>

                        <small>
                            exemples dans les trois splits
                        </small>

                    </div>

                </article>

            </section>


            <section class="model-section">

                <div class="model-section-heading">

                    <div>
                        <span>
                            CLASSIFICATION
                        </span>

                        <h2>
                            Ce que prédit réellement le modèle
                        </h2>
                    </div>

                </div>


                <div class="model-classes-grid">

                    <article class="model-class-card safe">

                        <span class="model-class-id">
                            CLASSE 0
                        </span>

                        <strong>
                            non_vulnerable
                        </strong>

                        <p>
                            Le modèle estime que le contrat appartient
                            à la classe non vulnérable selon les motifs
                            appris pendant l'entraînement.
                        </p>

                    </article>


                    <article class="model-class-card danger">

                        <span class="model-class-id">
                            CLASSE 1
                        </span>

                        <strong>
                            vulnerable
                        </strong>

                        <p>
                            Le modèle estime que le contrat présente
                            des caractéristiques associées aux contrats
                            vulnérables du jeu d'entraînement.
                        </p>

                    </article>

                </div>


                <div class="model-scientific-warning">

                    <div>!</div>

                    <p>
                        <strong>
                            Limite scientifique importante :
                        </strong>
                        le BiLSTM V2 effectue une classification
                        <b>binaire</b>. Il ne détermine pas directement
                        si la vulnérabilité est une réentrance,
                        un problème de contrôle d'accès ou une autre
                        catégorie précise. Les catégories détaillées
                        affichées ailleurs dans SMART BUG proviennent
                        du moteur d'analyse statique heuristique.
                    </p>

                </div>

            </section>


            <section class="model-section">

                <div class="model-section-heading">

                    <div>
                        <span>
                            ENTRAÎNEMENT
                        </span>

                        <h2>
                            Informations expérimentales
                        </h2>
                    </div>

                </div>


                <div class="model-training-grid">

                    ${modelMetricCard(
                        "Epoch retenu",
                        "4",
                        "Poids restaurés depuis le meilleur epoch",
                        "purple"
                    )}

                    ${modelMetricCard(
                        "Test Loss",
                        "0.344680",
                        "Évaluation rapide finale",
                        "blue"
                    )}

                    ${modelMetricCard(
                        "Longueur entrée",
                        "512",
                        "Tokens par contrat",
                        "cyan"
                    )}

                    ${modelMetricCard(
                        "Sortie",
                        "Softmax × 2",
                        "Probabilités des deux classes",
                        "success"
                    )}

                </div>

            </section>


            <div class="model-footer-note">

                <strong>
                    SMART BUG · BiLSTM V2
                </strong>

                <p>
                    Le modèle est un outil d'aide à l'audit.
                    Une prédiction ne remplace pas une analyse manuelle,
                    une analyse statique spécialisée ou un audit de sécurité complet.
                </p>

            </div>

        </section>
    `;
}


if (smartBugModelNav) {

    smartBugModelNav.addEventListener(
        "click",
        showModelView
    );
}


window.showModelView =
    showModelView;
