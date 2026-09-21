
"use strict";

/* =========================================================
   SMART BUG - À PROPOS
   ========================================================= */

function smartBugFindAboutNav() {
    return Array.from(
        document.querySelectorAll(
            ".nav-item"
        )
    ).find(
        item => {
            const text =
                item.textContent
                    .trim()
                    .toLowerCase();

            return (
                text.includes("à propos")
                ||
                text.includes("about")
                ||
                text.includes("über")
            );
        }
    );
}


function aboutSetActiveNavigation() {

    document
        .querySelectorAll(".nav-item")
        .forEach(
            item =>
                item.classList.remove(
                    "active"
                )
        );

    const aboutNav =
        smartBugFindAboutNav();

    if (aboutNav) {
        aboutNav.classList.add(
            "active"
        );
    }
}


function showAboutView() {

    const container =
        document.querySelector(
            ".content"
        );

    if (!container) {
        return;
    }

    aboutSetActiveNavigation();

    container.innerHTML = `
        <section class="about-page">

            <div class="about-page-header">

                <div>
                    <span class="about-eyebrow">
                        SMART BUG SECURITY PLATFORM
                    </span>

                    <h1>
                        À propos de SMART BUG
                    </h1>

                    <p>
                        Plateforme intelligente d'analyse de sécurité
                        des Smart Contracts Solidity.
                    </p>
                </div>

                <div class="about-version-badge">
                    V2 ACTIVE
                </div>

            </div>


            <section class="about-hero">

                <article class="about-brand-card">

                    <div class="about-logo-mark">
                        SB
                    </div>

                    <div>
                        <span>
                            SMART BUG
                        </span>

                        <h2>
                            AI-Powered Smart Contract Security
                        </h2>

                        <p>
                            Détection automatisée, analyse des risques,
                            visualisation et aide à l'audit des contrats Solidity.
                        </p>
                    </div>

                </article>


                <article class="about-mission-card">

                    <span class="about-card-label">
                        MISSION
                    </span>

                    <p>
                        SMART BUG est une plateforme intelligente
                        d'analyse de sécurité des Smart Contracts Solidity.
                        Elle combine apprentissage profond, analyse du code
                        et visualisation interactive afin d'aider à détecter
                        automatiquement les contrats potentiellement vulnérables
                        avant leur déploiement.
                    </p>

                </article>

            </section>


            <section class="about-section">

                <div class="about-section-heading">

                    <div>
                        <span>
                            CAPACITÉS
                        </span>

                        <h2>
                            Ce que fait SMART BUG
                        </h2>
                    </div>

                </div>


                <div class="about-capabilities-grid">

                    <article class="about-capability-card">
                        <div>01</div>
                        <strong>
                            Classification IA
                        </strong>
                        <p>
                            Le BiLSTM V2 classe un contrat en
                            vulnerable ou non_vulnerable
                            et fournit une probabilité pour chaque classe.
                        </p>
                    </article>


                    <article class="about-capability-card">
                        <div>02</div>
                        <strong>
                            Analyse statique
                        </strong>
                        <p>
                            Un moteur heuristique recherche des motifs
                            de risque dans le code Solidity et produit
                            des findings accompagnés de recommandations.
                        </p>
                    </article>


                    <article class="about-capability-card">
                        <div>03</div>
                        <strong>
                            Analyse de performance
                        </strong>
                        <p>
                            SMART BUG estime la complexité,
                            l'efficacité et plusieurs indicateurs
                            potentiels de coût d'exécution.
                        </p>
                    </article>


                    <article class="about-capability-card">
                        <div>04</div>
                        <strong>
                            Historique local
                        </strong>
                        <p>
                            Les analyses effectuées sont enregistrées
                            localement dans SQLite afin de conserver
                            un historique exploitable.
                        </p>
                    </article>


                    <article class="about-capability-card">
                        <div>05</div>
                        <strong>
                            Rapports
                        </strong>
                        <p>
                            Le dernier audit peut être présenté sous
                            forme de rapport lisible, imprimable et
                            enregistrable en PDF.
                        </p>
                    </article>


                    <article class="about-capability-card">
                        <div>06</div>
                        <strong>
                            Visualisation interactive
                        </strong>
                        <p>
                            Les résultats sont regroupés dans un
                            dashboard comprenant risques, modèle,
                            dataset, performances et recommandations.
                        </p>
                    </article>

                </div>

            </section>


            <section class="about-two-column">

                <article class="about-section">

                    <div class="about-section-heading">

                        <div>
                            <span>
                                INTELLIGENCE ARTIFICIELLE
                            </span>

                            <h2>
                                BiLSTM V2
                            </h2>
                        </div>

                    </div>


                    <div class="about-model-summary">

                        <div>
                            <span>
                                Architecture
                            </span>
                            <strong>
                                Embedding → BiLSTM → Pooling → Dense
                            </strong>
                        </div>

                        <div>
                            <span>
                                Paramètres
                            </span>
                            <strong>
                                346 978
                            </strong>
                        </div>

                        <div>
                            <span>
                                Entrée
                            </span>
                            <strong>
                                512 tokens
                            </strong>
                        </div>

                        <div>
                            <span>
                                Vocabulaire
                            </span>
                            <strong>
                                5 000
                            </strong>
                        </div>

                        <div>
                            <span>
                                Classes
                            </span>
                            <strong>
                                2
                            </strong>
                        </div>

                        <div>
                            <span>
                                Accuracy test
                            </span>
                            <strong>
                                85,87 %
                            </strong>
                        </div>

                    </div>

                </article>


                <article class="about-section">

                    <div class="about-section-heading">

                        <div>
                            <span>
                                TECHNOLOGIES
                            </span>

                            <h2>
                                Stack de la plateforme
                            </h2>
                        </div>

                    </div>


                    <div class="about-tech-grid">

                        <span>Python</span>
                        <span>FastAPI</span>
                        <span>TensorFlow</span>
                        <span>Keras</span>
                        <span>SQLite</span>
                        <span>JavaScript</span>
                        <span>HTML5</span>
                        <span>CSS3</span>
                        <span>Solidity</span>

                    </div>


                    <div class="about-local-note">

                        <strong>
                            Fonctionnement local
                        </strong>

                        <p>
                            Le serveur FastAPI, le modèle IA,
                            la base SQLite et les principaux traitements
                            fonctionnent localement sur la machine
                            qui exécute SMART BUG.
                        </p>

                    </div>

                </article>

            </section>


            <section class="about-section">

                <div class="about-section-heading">

                    <div>
                        <span>
                            MÉTHODOLOGIE
                        </span>

                        <h2>
                            Séparation des moteurs
                        </h2>
                    </div>

                </div>


                <div class="about-engines-grid">

                    <article class="about-engine-card ai">

                        <span>
                            MOTEUR 01
                        </span>

                        <strong>
                            BiLSTM V2
                        </strong>

                        <p>
                            Classification binaire du contrat :
                            vulnerable ou non_vulnerable.
                        </p>

                        <small>
                            Apprentissage profond
                        </small>

                    </article>


                    <article class="about-engine-card static">

                        <span>
                            MOTEUR 02
                        </span>

                        <strong>
                            Static Risk Analyzer
                        </strong>

                        <p>
                            Détection heuristique de motifs de risque
                            et production de catégories détaillées.
                        </p>

                        <small>
                            Analyse statique
                        </small>

                    </article>


                    <article class="about-engine-card performance">

                        <span>
                            MOTEUR 03
                        </span>

                        <strong>
                            Performance Analyzer
                        </strong>

                        <p>
                            Analyse heuristique de complexité,
                            efficacité et pression potentielle de coût.
                        </p>

                        <small>
                            Analyse de performance
                        </small>

                    </article>

                </div>

            </section>


            <section class="about-scientific-warning">

                <div>
                    !
                </div>

                <div>

                    <strong>
                        Limite scientifique importante
                    </strong>

                    <p>
                        Le modèle BiLSTM V2 ne prédit pas directement
                        le type exact de vulnérabilité.
                        Il effectue une classification binaire.
                        Les catégories précises présentées dans
                        la Synthèse des risques proviennent du moteur
                        heuristique d'analyse statique.
                    </p>

                </div>

            </section>


            <section class="about-section">

                <div class="about-section-heading">

                    <div>
                        <span>
                            AVERTISSEMENT
                        </span>

                        <h2>
                            Outil d'aide à l'audit
                        </h2>
                    </div>

                </div>


                <div class="about-disclaimer">

                    <p>
                        SMART BUG constitue un outil d'aide à l'audit
                        et ne remplace pas une vérification manuelle
                        approfondie réalisée par un spécialiste
                        en sécurité blockchain.
                    </p>

                    <p>
                        Une prédiction « non vulnérable » ne constitue
                        pas une garantie absolue de sécurité.
                        Avant un déploiement en production,
                        il reste recommandé d'utiliser plusieurs méthodes
                        complémentaires : revue manuelle, analyse statique,
                        tests, compilation, fuzzing et audit spécialisé.
                    </p>

                </div>

            </section>


            <footer class="about-footer">

                <div>
                    <strong>
                        SMART BUG
                    </strong>

                    <span>
                        AI-Powered Smart Contract Security
                    </span>
                </div>

                <div>
                    <span>
                        BiLSTM V2
                    </span>

                    <span>
                        Dataset V2 + CGT
                    </span>

                    <span>
                        Classification binaire
                    </span>
                </div>

            </footer>

        </section>
    `;
}


function bindAboutNavigation() {

    const aboutNav =
        smartBugFindAboutNav();

    if (
        aboutNav
        &&
        !aboutNav.dataset.aboutBound
    ) {

        aboutNav.dataset.aboutBound =
            "true";

        aboutNav.addEventListener(
            "click",
            showAboutView
        );
    }
}


bindAboutNavigation();


window.showAboutView =
    showAboutView;
