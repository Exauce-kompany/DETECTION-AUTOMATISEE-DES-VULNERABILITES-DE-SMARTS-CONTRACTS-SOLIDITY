const uploadForm = document.getElementById("uploadForm");
const fileInput = document.getElementById("contractFile");
const uploadZone = document.getElementById("uploadZone");
const selectedFile = document.getElementById("selectedFile");
const analyzeButton = document.getElementById("analyzeButton");

const loadingSection = document.getElementById("loadingSection");
const errorSection = document.getElementById("errorSection");
const errorMessage = document.getElementById("errorMessage");

const resultPlaceholder = document.getElementById("resultPlaceholder");
const resultContent = document.getElementById("resultContent");

const analysisStatus = document.getElementById("analysisStatus");

const verdictBox = document.getElementById("verdictBox");
const verdictText = document.getElementById("verdictText");
const resultMessage = document.getElementById("resultMessage");

const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");

const riskScore = document.getElementById("riskScore");
const riskLabel = document.getElementById("riskLabel");
const riskRing = document.querySelector(".risk-ring");

const vulnerableProbabilityText = document.getElementById(
    "vulnerableProbabilityText"
);
const vulnerableProbabilityBar = document.getElementById(
    "vulnerableProbabilityBar"
);

const safeProbabilityText = document.getElementById(
    "safeProbabilityText"
);
const safeProbabilityBar = document.getElementById(
    "safeProbabilityBar"
);

const tokensDetected = document.getElementById("tokensDetected");
const tokensUsed = document.getElementById("tokensUsed");
const unknownTokens = document.getElementById("unknownTokens");
const unknownRate = document.getElementById("unknownRate");
const truncated = document.getElementById("truncated");
const fileSize = document.getElementById("fileSize");

const fileName = document.getElementById("fileName");
const codeLineCount = document.getElementById("codeLineCount");
const codePreview = document.getElementById("codePreview");

const findingList = document.getElementById("findingList");

const contractsAnalyzed = document.getElementById("contractsAnalyzed");
const vulnerableCount = document.getElementById("vulnerableCount");
const recentActivity = document.getElementById("recentActivity");

let currentFile = null;
let currentFileContent = "";
let analyzedCount = 0;
let vulnerableDetectedCount = 0;
let recentItems = [];


/* ========================================================= */
/* FILE SELECTION */
/* ========================================================= */

fileInput.addEventListener("change", async () => {
    if (fileInput.files.length === 0) {
        resetSelectedFile();
        return;
    }

    await handleSelectedFile(
        fileInput.files[0]
    );
});


async function handleSelectedFile(file) {

    if (!file) {
        return;
    }

    if (
        !file.name
            .toLowerCase()
            .endsWith(".sol")
    ) {
        showError(
            "Le fichier sélectionné doit avoir l'extension .sol."
        );
        return;
    }

    currentFile = file;

    selectedFile.textContent =
        file.name;

    fileName.textContent =
        file.name;

    fileSize.textContent =
        formatFileSize(file.size);

    try {

        currentFileContent =
            await file.text();

        const lineCount =
            currentFileContent
                .split(/\r?\n/)
                .length;

        codeLineCount.textContent =
            `${lineCount} lignes`;

        codePreview.textContent =
            currentFileContent || "Aucun code chargé.";

    } catch (error) {

        currentFileContent = "";

        codePreview.textContent =
            "Impossible d'afficher le contenu du fichier.";

    }

    hideError();
}


/* ========================================================= */
/* DRAG & DROP */
/* ========================================================= */

[
    "dragenter",
    "dragover"
].forEach(eventName => {

    uploadZone.addEventListener(
        eventName,
        event => {

            event.preventDefault();
            event.stopPropagation();

            uploadZone.classList.add(
                "drag-over"
            );
        }
    );

});


[
    "dragleave",
    "drop"
].forEach(eventName => {

    uploadZone.addEventListener(
        eventName,
        event => {

            event.preventDefault();
            event.stopPropagation();

            uploadZone.classList.remove(
                "drag-over"
            );
        }
    );

});


uploadZone.addEventListener(
    "drop",
    async event => {

        const files =
            event.dataTransfer.files;

        if (
            !files ||
            files.length === 0
        ) {
            return;
        }

        const file =
            files[0];

        if (
            !file.name
                .toLowerCase()
                .endsWith(".sol")
        ) {

            showError(
                "Veuillez déposer un fichier Solidity avec l'extension .sol."
            );

            return;
        }

        currentFile = file;

        await handleSelectedFile(
            file
        );
    }
);


/* ========================================================= */
/* FORM SUBMIT */
/* ========================================================= */

uploadForm.addEventListener(
    "submit",
    async event => {

        event.preventDefault();

        let file = null;

        if (
            fileInput.files.length > 0
        ) {

            file =
                fileInput.files[0];

        } else if (
            currentFile
        ) {

            file =
                currentFile;

        }


        if (!file) {

            showError(
                "Veuillez sélectionner un fichier Solidity .sol."
            );

            return;
        }


        if (
            !file.name
                .toLowerCase()
                .endsWith(".sol")
        ) {

            showError(
                "Le fichier doit avoir l'extension .sol."
            );

            return;
        }


        hideError();

        setAnalysisRunning();


        const formData =
            new FormData();

        formData.append(
            "file",
            file
        );


        try {

            const response =
                await fetch(
                    "/api/analyze",
                    {
                        method:
                            "POST",

                        body:
                            formData
                    }
                );


            const data =
                await response.json();


            if (
                !response.ok
            ) {

                throw new Error(
                    data.detail
                    ||
                    "Erreur pendant l'analyse."
                );

            }


            displayResult(
                data
            );


        } catch (error) {

            setAnalysisIdle();

            showError(
                error.message
            );

        } finally {

            loadingSection.classList.add(
                "hidden"
            );

            analyzeButton.disabled =
                false;

        }

    }
);


/* ========================================================= */
/* ANALYSIS STATES */
/* ========================================================= */

function setAnalysisRunning() {

    loadingSection.classList.remove(
        "hidden"
    );

    analyzeButton.disabled =
        true;

    analysisStatus.textContent =
        "ANALYSE";

    analysisStatus.className =
        "analysis-status running";
}


function setAnalysisIdle() {

    analysisStatus.textContent =
        "EN ATTENTE";

    analysisStatus.className =
        "analysis-status idle";
}


function setAnalysisDone() {

    analysisStatus.textContent =
        "TERMINÉE";

    analysisStatus.className =
        "analysis-status done";
}


/* ========================================================= */
/* DISPLAY RESULT */
/* ========================================================= */

function displayResult(data) {

    resultPlaceholder.classList.add(
        "hidden"
    );

    resultContent.classList.remove(
        "hidden"
    );

    setAnalysisDone();


    /* ----------------------------------------------------- */
    /* COUNTERS */
    /* ----------------------------------------------------- */

    /*
     * Les compteurs du Dashboard sont synchronisés avec
     * SQLite. On ne les incrémente plus uniquement en mémoire,
     * sinon ils repartent à zéro après un rechargement.
     */
    syncDashboardFromDatabase();


    /* ----------------------------------------------------- */
    /* VERDICT */
    /* ----------------------------------------------------- */

    verdictText.textContent =
        data.predicted_label === 1
            ? "VULNÉRABLE"
            : "NON VULNÉRABLE";

    resultMessage.textContent =
        data.message;


    if (
        data.predicted_label === 1
    ) {

        verdictBox.style.borderColor =
            "rgba(255, 80, 103, 0.42)";

        verdictText.style.color =
            "#ff4d67";

        confidenceValue.style.color =
            "#ff5b73";

        confidenceBar.style.background =
            "linear-gradient(90deg, #ff335d, #ff6f90)";

    } else {

        verdictBox.style.borderColor =
            "rgba(30, 215, 96, 0.38)";

        verdictText.style.color =
            "#42e58e";

        confidenceValue.style.color =
            "#42e58e";

        confidenceBar.style.background =
            "linear-gradient(90deg, #22c55e, #61f59c)";
    }


    /* ----------------------------------------------------- */
    /* CONFIDENCE */
    /* ----------------------------------------------------- */

    const confidence =
        Number(
            data.confidence_percent
        );

    confidenceValue.textContent =
        `${confidence.toFixed(2)}%`;

    confidenceBar.style.width =
        `${confidence}%`;


    /* ----------------------------------------------------- */
    /* RISK SCORE */
    /* ----------------------------------------------------- */

    const risk =
        calculateRiskScore(
            data
        );

    riskScore.textContent =
        risk;

    updateRiskRing(
        risk,
        data.predicted_label
    );

    riskLabel.textContent =
        riskLevelLabel(
            risk
        );


    /* ----------------------------------------------------- */
    /* PROBABILITIES */
    /* ----------------------------------------------------- */

    const vulnerableProbability =
        Number(
            data
            .probability_vulnerable_percent
        );

    const safeProbability =
        Number(
            data
            .probability_non_vulnerable_percent
        );


    vulnerableProbabilityText.textContent =
        `${vulnerableProbability.toFixed(2)}%`;

    vulnerableProbabilityBar.style.width =
        `${vulnerableProbability}%`;


    safeProbabilityText.textContent =
        `${safeProbability.toFixed(2)}%`;

    safeProbabilityBar.style.width =
        `${safeProbability}%`;


    /* ----------------------------------------------------- */
    /* TECHNICAL INFO */
    /* ----------------------------------------------------- */

    const preprocessing =
        data.preprocessing;


    tokensDetected.textContent =
        preprocessing.tokens_detected;

    tokensUsed.textContent =
        preprocessing.tokens_used;

    unknownTokens.textContent =
        preprocessing.unknown_tokens;

    unknownRate.textContent =
        `${(
            Number(
                preprocessing.unknown_rate
            ) * 100
        ).toFixed(2)}%`;

    truncated.textContent =
        preprocessing.truncated
            ? "Oui"
            : "Non";


    fileSize.textContent =
        `${data.file.size_kb} Ko`;

    fileName.textContent =
        data.file.filename;


    /* ----------------------------------------------------- */
    /* FINDINGS */
    /* ----------------------------------------------------- */

    updateFindings(
        data
    );


    /* ----------------------------------------------------- */
    /* ACTIVITY */
    /* ----------------------------------------------------- */

    addRecentActivity(
        data
    );


    resultContent.scrollIntoView(
        {
            behavior:
                "smooth",

            block:
                "start"
        }
    );
}


/* ========================================================= */
/* RISK SCORE */
/* ========================================================= */

function calculateRiskScore(data) {

    const vulnerableProbability =
        Number(
            data
            .probability_vulnerable_percent
        );


    if (
        data.predicted_label === 1
    ) {

        return Math.round(
            vulnerableProbability
        );

    }


    const safeProbability =
        Number(
            data
            .probability_non_vulnerable_percent
        );


    return Math.round(
        100 - safeProbability
    );
}


function riskLevelLabel(score) {

    if (
        score >= 85
    ) {
        return "CRITIQUE";
    }

    if (
        score >= 65
    ) {
        return "ÉLEVÉ";
    }

    if (
        score >= 40
    ) {
        return "MODÉRÉ";
    }

    return "FAIBLE";
}


function updateRiskRing(
    score,
    predictedLabel
) {

    const degrees =
        Math.max(
            0,
            Math.min(
                360,
                score * 3.6
            )
        );


    if (
        predictedLabel === 1
    ) {

        riskRing.style.background =
            `conic-gradient(
                #ff445f 0deg,
                #ff864e ${degrees}deg,
                rgba(48, 67, 98, 0.34) ${degrees}deg
            )`;

        riskLabel.style.color =
            "#ff665e";

    } else {

        riskRing.style.background =
            `conic-gradient(
                #24dd7b 0deg,
                #57f39a ${degrees}deg,
                rgba(48, 67, 98, 0.34) ${degrees}deg
            )`;

        riskLabel.style.color =
            "#42e58e";
    }
}


/* ========================================================= */
/* FINDINGS */
/* ========================================================= */

function updateFindings(data) {

    findingList.innerHTML =
        "";


    const preprocessing =
        data.preprocessing;


    if (
        data.predicted_label === 1
    ) {

        addFinding(
            "danger",
            "Le modèle classe ce contrat comme potentiellement vulnérable."
        );

    } else {

        addFinding(
            "success",
            "Aucun signal IA dans le périmètre appris ; cela ne certifie pas la sécurité."
        );
    }


    if (
        preprocessing.truncated
    ) {

        addFinding(
            "warning",
            "Cette analyse historique utilisait une entrée tronquée."
        );

    } else {

        addFinding(
            "success",
            "Le contrat a été analysé sans troncature."
        );
    }


    const unkRate =
        Number(
            preprocessing.unknown_rate
        );


    if (
        unkRate >= 0.15
    ) {

        addFinding(
            "warning",
            `Tokens encodés par repli en octets (contenu conservé) : ${(unkRate * 100).toFixed(2)}%.`
        );

    } else {

        addFinding(
            "info",
            `Tokens encodés par repli en octets : ${(unkRate * 100).toFixed(2)}%.`
        );
    }


    addFinding(
        "info",
        `Score de la classe prédite : ${Number(data.confidence_percent).toFixed(2)}%. Seuil de signalement : ${Number(data.decision_threshold * 100).toFixed(2)}%.`
    );
}


function addFinding(
    type,
    text
) {

    const element =
        document.createElement(
            "div"
        );

    element.className =
        `finding ${type}`;

    element.textContent =
        text;

    findingList.appendChild(
        element
    );
}


/* ========================================================= */
/* RECENT ACTIVITY */
/* ========================================================= */

function addRecentActivity(data) {

    const item = {

        filename:
            data.file.filename,

        vulnerable:
            data.predicted_label === 1,

        confidence:
            Number(
                data.confidence_percent
            ),

        time:
            new Date()
                .toLocaleTimeString(
                    "fr-FR",
                    {
                        hour:
                            "2-digit",

                        minute:
                            "2-digit"
                    }
                )

    };


    recentItems.unshift(
        item
    );


    recentItems =
        recentItems.slice(
            0,
            6
        );


    renderRecentActivity();
}


function renderRecentActivity() {

    recentActivity.innerHTML =
        "";


    if (
        recentItems.length === 0
    ) {

        recentActivity.innerHTML =
            `
            <div class="activity-empty">
                Aucune analyse dans cette session.
            </div>
            `;

        return;
    }


    recentItems.forEach(
        item => {

            const entry =
                document.createElement(
                    "div"
                );

            entry.className =
                "activity-entry";


            const info =
                document.createElement(
                    "div"
                );


            const name =
                document.createElement(
                    "strong"
                );

            name.textContent =
                item.filename;


            const meta =
                document.createElement(
                    "small"
                );

            meta.textContent =
                `${item.time} • Confiance ${item.confidence.toFixed(2)}%`;


            info.appendChild(
                name
            );

            info.appendChild(
                meta
            );


            const badge =
                document.createElement(
                    "span"
                );

            badge.className =
                item.vulnerable
                    ? "activity-badge danger"
                    : "activity-badge safe";

            badge.textContent =
                item.vulnerable
                    ? "VULNÉRABLE"
                    : "SÛR";


            entry.appendChild(
                info
            );

            entry.appendChild(
                badge
            );


            recentActivity.appendChild(
                entry
            );

        }
    );
}



/* ========================================================= */
/* DASHBOARD PERSISTANT - SQLITE */
/* ========================================================= */

/*
 * Recharge les KPI et l'activité récente depuis la base
 * SQLite via les routes FastAPI existantes.
 *
 * Résultat :
 * - les compteurs survivent à Ctrl+F5 ;
 * - ils survivent au retour depuis Historique/Risques ;
 * - ils survivent au redémarrage du navigateur ;
 * - SQLite reste la source de vérité.
 */
async function syncDashboardFromDatabase() {

    try {

        const [
            statsResponse,
            historyResponse
        ] = await Promise.all(
            [
                fetch(
                    "/api/history-statistics",
                    {
                        cache:
                            "no-store"
                    }
                ),

                fetch(
                    "/api/history",
                    {
                        cache:
                            "no-store"
                    }
                )
            ]
        );


        if (
            !statsResponse.ok
            ||
            !historyResponse.ok
        ) {

            throw new Error(
                "Impossible de synchroniser le Dashboard avec l'historique."
            );
        }


        const statsPayload =
            await statsResponse.json();

        const historyPayload =
            await historyResponse.json();


        const stats =
            statsPayload.statistics
            ||
            {};


        analyzedCount =
            Number(
                stats.total_analyses
                ||
                0
            );


        vulnerableDetectedCount =
            Number(
                stats.vulnerable
                ||
                0
            );


        if (
            contractsAnalyzed
        ) {

            contractsAnalyzed.textContent =
                analyzedCount;
        }


        if (
            vulnerableCount
        ) {

            vulnerableCount.textContent =
                vulnerableDetectedCount;
        }


        const analyses =
            Array.isArray(
                historyPayload.analyses
            )
                ? historyPayload.analyses
                : [];


        recentItems =
            analyses
                .slice(
                    0,
                    6
                )
                .map(
                    analysis => {

                        const confidence =
                            Number(
                                analysis.confidence
                                ||
                                0
                            );


                        let time =
                            "-";


                        if (
                            analysis.created_at
                        ) {

                            const parsedDate =
                                new Date(
                                    analysis.created_at
                                );


                            if (
                                !Number.isNaN(
                                    parsedDate.getTime()
                                )
                            ) {

                                time =
                                    parsedDate
                                        .toLocaleTimeString(
                                            "fr-FR",
                                            {
                                                hour:
                                                    "2-digit",

                                                minute:
                                                    "2-digit"
                                            }
                                        );
                            }
                        }


                        return {

                            filename:
                                analysis.filename
                                ||
                                "contract.sol",

                            vulnerable:
                                Number(
                                    analysis.predicted_label
                                )
                                ===
                                1,

                            confidence:
                                Number.isFinite(
                                    confidence
                                )
                                    ? confidence
                                    : 0,

                            time:
                                time

                        };

                    }
                );


        if (
            recentActivity
        ) {

            renderRecentActivity();
        }


    } catch (error) {

        console.error(
            "SMART BUG - Synchronisation Dashboard :",
            error
        );
    }
}


/*
 * Au chargement initial, les KPI du Dashboard sont restaurés
 * depuis SQLite au lieu de repartir de zéro.
 */
syncDashboardFromDatabase();


/* ========================================================= */
/* ERRORS */
/* ========================================================= */

function showError(message) {

    errorMessage.textContent =
        message;

    errorSection.classList.remove(
        "hidden"
    );


    setTimeout(
        () => {

            errorSection.classList.add(
                "hidden"
            );

        },
        6000
    );
}


function hideError() {

    errorSection.classList.add(
        "hidden"
    );
}


/* ========================================================= */
/* UTILITIES */
/* ========================================================= */

function resetSelectedFile() {

    currentFile =
        null;

    currentFileContent =
        "";

    selectedFile.textContent =
        "Aucun fichier sélectionné";

    fileName.textContent =
        "Aucun fichier";

    codeLineCount.textContent =
        "0 lignes";

    codePreview.textContent =
        "Aucun code chargé.";
}


function formatFileSize(bytes) {

    if (
        bytes < 1024
    ) {

        return `${bytes} octets`;

    }


    if (
        bytes <
        1024 * 1024
    ) {

        return `${(
            bytes / 1024
        ).toFixed(2)} Ko`;

    }


    return `${(
        bytes /
        (1024 * 1024)
    ).toFixed(2)} Mo`;
}
/* ========================================================= */
/* SMART BUG - NAVIGATION / HISTORIQUE */
/* ========================================================= */

const contentContainer =
    document.querySelector(".content");

const navItems =
    document.querySelectorAll(".nav-item");


const analysisNav =
    Array.from(navItems).find(
        item =>
            item.textContent
                .trim()
                .includes("Analyse")
            &&
            !item.textContent
                .includes("risques")
    );


const historyNav =
    Array.from(navItems).find(
        item =>
            item.textContent
                .trim()
                .includes("Historique")
    );


let originalDashboardHTML = null;
let currentView = "analysis";


/* ========================================================= */
/* SAUVEGARDE DU DASHBOARD */
/* ========================================================= */

function saveOriginalDashboard() {

    if (!originalDashboardHTML) {

        originalDashboardHTML =
            contentContainer.innerHTML;
    }
}


/* ========================================================= */
/* NAVIGATION */
/* ========================================================= */

function setActiveNavigation(
    activeItem
) {

    navItems.forEach(
        item => {

            item.classList.remove(
                "active"
            );

        }
    );


    if (activeItem) {

        activeItem.classList.add(
            "active"
        );
    }
}


/* ========================================================= */
/* PAGE ANALYSE */
/* ========================================================= */

function showAnalysisView() {

    /*
     * On ne quitte la fonction que si le véritable écran
     * d'analyse est encore présent dans le DOM.
     *
     * La vue "Synthèse des risques" remplace .content mais
     * currentView peut encore valoir "analysis". Dans ce cas,
     * il faut recharger la page pour reconstruire le Dashboard.
     *
     * Les KPI sont ensuite restaurés depuis SQLite par
     * syncDashboardFromDatabase().
     */
    const analysisFormStillVisible =
        document.getElementById(
            "uploadForm"
        );


    if (
        currentView === "analysis"
        &&
        analysisFormStillVisible
    ) {
        return;
    }


    window.location.reload();
}


/* ========================================================= */
/* PAGE HISTORIQUE */
/* ========================================================= */

async function showHistoryView() {

    saveOriginalDashboard();

    currentView = "history";

    setActiveNavigation(
        historyNav
    );


    contentContainer.innerHTML = `
        <section class="history-page">

            <div class="history-header">

                <div>
                    <span class="history-eyebrow">
                        SMART BUG SECURITY CENTER
                    </span>

                    <h1>
                        Historique des analyses
                    </h1>

                    <p>
                        Consultez toutes les analyses de Smart Contracts
                        enregistrées par SMART BUG.
                    </p>
                </div>

                <button
                    id="historyNewAnalysis"
                    class="history-primary-button"
                >
                    + Nouvelle analyse
                </button>

            </div>


            <section
                id="historyStats"
                class="history-stats-grid"
            >

                <div class="history-stat-card">
                    <span>Total des analyses</span>
                    <strong id="historyTotal">0</strong>
                    <small>Contrats analysés</small>
                </div>

                <div class="history-stat-card danger">
                    <span>Vulnérables</span>
                    <strong id="historyVulnerable">0</strong>
                    <small>Risques détectés</small>
                </div>

                <div class="history-stat-card success">
                    <span>Non vulnérables</span>
                    <strong id="historySafe">0</strong>
                    <small>Contrats classés sûrs</small>
                </div>

                <div class="history-stat-card purple">
                    <span>Confiance moyenne</span>
                    <strong id="historyConfidence">0%</strong>
                    <small>Moyenne du modèle IA</small>
                </div>

            </section>


            <section class="history-main-panel">

                <div class="history-toolbar">

                    <div class="history-search-wrapper">

                        <span>
                            ⌕
                        </span>

                        <input
                            id="historySearch"
                            type="search"
                            placeholder="Rechercher un contrat..."
                        >

                    </div>


                    <div class="history-filter-group">

                        <button
                            class="history-filter active"
                            data-filter="all"
                        >
                            Tous
                        </button>

                        <button
                            class="history-filter"
                            data-filter="vulnerable"
                        >
                            Vulnérables
                        </button>

                        <button
                            class="history-filter"
                            data-filter="safe"
                        >
                            Non vulnérables
                        </button>

                    </div>


                    <button
                        id="refreshHistory"
                        class="history-secondary-button"
                    >
                        ↻ Actualiser
                    </button>


                    <button
                        id="clearHistoryButton"
                        class="history-danger-button"
                    >
                        Supprimer tout
                    </button>

                </div>


                <div
                    id="historyLoading"
                    class="history-loading"
                >
                    Chargement de l'historique...
                </div>


                <div
                    id="historyEmpty"
                    class="history-empty hidden"
                >

                    <div class="history-empty-icon">
                        ◷
                    </div>

                    <h3>
                        Aucun historique disponible
                    </h3>

                    <p>
                        Les contrats analysés apparaîtront ici
                        automatiquement.
                    </p>

                </div>


                <div
                    id="historyTableWrapper"
                    class="history-table-wrapper hidden"
                >

                    <table
                        class="history-table"
                    >

                        <thead>

                            <tr>

                                <th>
                                    ID
                                </th>

                                <th>
                                    Contrat
                                </th>

                                <th>
                                    Verdict
                                </th>

                                <th>
                                    Confiance
                                </th>

                                <th>
                                    Risque
                                </th>

                                <th>
                                    Tokens
                                </th>

                                <th>
                                    Date
                                </th>

                                <th>
                                    Actions
                                </th>

                            </tr>

                        </thead>

                        <tbody
                            id="historyTableBody"
                        >
                        </tbody>

                    </table>

                </div>

            </section>


            <section
                id="historyDetailPanel"
                class="history-detail-panel hidden"
            >

                <div class="history-detail-header">

                    <div>

                        <span>
                            DÉTAIL DE L'ANALYSE
                        </span>

                        <h2
                            id="historyDetailFilename"
                        >
                            -
                        </h2>

                    </div>


                    <button
                        id="closeHistoryDetail"
                        class="history-close-button"
                    >
                        ×
                    </button>

                </div>


                <div class="history-detail-grid">


                    <div class="history-detail-card">

                        <span>
                            Verdict
                        </span>

                        <strong
                            id="historyDetailVerdict"
                        >
                            -
                        </strong>

                    </div>


                    <div class="history-detail-card">

                        <span>
                            Confiance IA
                        </span>

                        <strong
                            id="historyDetailConfidence"
                        >
                            -
                        </strong>

                    </div>


                    <div class="history-detail-card">

                        <span>
                            Score de risque
                        </span>

                        <strong
                            id="historyDetailRisk"
                        >
                            -
                        </strong>

                    </div>


                    <div class="history-detail-card">

                        <span>
                            Niveau de risque
                        </span>

                        <strong
                            id="historyDetailRiskLevel"
                        >
                            -
                        </strong>

                    </div>

                </div>


                <div class="history-probabilities">

                    <h3>
                        Probabilités du modèle
                    </h3>


                    <div class="history-probability-row">

                        <span>
                            Vulnérable
                        </span>

                        <div class="history-probability-track">

                            <div
                                id="historyVulnerableBar"
                                class="history-probability-fill danger"
                            >
                            </div>

                        </div>

                        <strong
                            id="historyVulnerableProbability"
                        >
                            0%
                        </strong>

                    </div>


                    <div class="history-probability-row">

                        <span>
                            Non vulnérable
                        </span>

                        <div class="history-probability-track">

                            <div
                                id="historySafeBar"
                                class="history-probability-fill success"
                            >
                            </div>

                        </div>

                        <strong
                            id="historySafeProbability"
                        >
                            0%
                        </strong>

                    </div>

                </div>


                <div class="history-tech-grid">

                    <div>
                        <span>Tokens détectés</span>
                        <strong id="historyDetailTokensDetected">-</strong>
                    </div>

                    <div>
                        <span>Tokens utilisés</span>
                        <strong id="historyDetailTokensUsed">-</strong>
                    </div>

                    <div>
                        <span>Tokens inconnus</span>
                        <strong id="historyDetailUnknown">-</strong>
                    </div>

                    <div>
                        <span>Troncature</span>
                        <strong id="historyDetailTruncated">-</strong>
                    </div>

                </div>


                <div class="history-code-panel">

                    <div class="history-code-header">

                        <h3>
                            Code Solidity enregistré
                        </h3>

                    </div>

                    <pre
                        id="historyDetailCode"
                    ></pre>

                </div>

            </section>

        </section>
    `;


    attachHistoryEvents();

    await loadHistory();
}


/* ========================================================= */
/* CHARGEMENT HISTORIQUE */
/* ========================================================= */

let historyData = [];
let historyFilter = "all";


async function loadHistory() {

    const loading =
        document.getElementById(
            "historyLoading"
        );

    const empty =
        document.getElementById(
            "historyEmpty"
        );

    const tableWrapper =
        document.getElementById(
            "historyTableWrapper"
        );


    if (loading) {
        loading.classList.remove(
            "hidden"
        );
    }


    try {

        const response =
            await fetch(
                "/api/history"
            );


        if (!response.ok) {

            throw new Error(
                "Impossible de charger l'historique."
            );
        }


        const data =
            await response.json();


        historyData =
            data.analyses || [];


        await loadHistoryStatistics();


        renderHistoryTable();


    } catch (error) {

        if (loading) {

            loading.textContent =
                error.message;
        }

        return;
    }


    if (loading) {

        loading.classList.add(
            "hidden"
        );
    }


    if (
        historyData.length === 0
    ) {

        empty.classList.remove(
            "hidden"
        );

        tableWrapper.classList.add(
            "hidden"
        );

    } else {

        empty.classList.add(
            "hidden"
        );

        tableWrapper.classList.remove(
            "hidden"
        );
    }
}


/* ========================================================= */
/* STATISTIQUES HISTORIQUE */
/* ========================================================= */

async function loadHistoryStatistics() {

    try {

        const response =
            await fetch(
                "/api/history-statistics"
            );


        const data =
            await response.json();


        const stats =
            data.statistics;


        document.getElementById(
            "historyTotal"
        ).textContent =
            stats.total_analyses;


        document.getElementById(
            "historyVulnerable"
        ).textContent =
            stats.vulnerable;


        document.getElementById(
            "historySafe"
        ).textContent =
            stats.non_vulnerable;


        document.getElementById(
            "historyConfidence"
        ).textContent =
            `${Number(
                stats.average_confidence
            ).toFixed(2)}%`;


    } catch (error) {

        console.error(
            error
        );
    }
}


/* ========================================================= */
/* TABLE HISTORIQUE */
/* ========================================================= */

function renderHistoryTable() {

    const tbody =
        document.getElementById(
            "historyTableBody"
        );


    const searchInput =
        document.getElementById(
            "historySearch"
        );


    if (!tbody) {
        return;
    }


    const searchValue =
        searchInput
            ? searchInput.value
                .trim()
                .toLowerCase()
            : "";


    const filtered =
        historyData.filter(
            item => {


                const filenameMatch =
                    item.filename
                        .toLowerCase()
                        .includes(
                            searchValue
                        );


                let filterMatch =
                    true;


                if (
                    historyFilter ===
                    "vulnerable"
                ) {

                    filterMatch =
                        item.predicted_label === 1;
                }


                if (
                    historyFilter ===
                    "safe"
                ) {

                    filterMatch =
                        item.predicted_label === 0;
                }


                return (
                    filenameMatch
                    &&
                    filterMatch
                );
            }
        );


    tbody.innerHTML = "";


    filtered.forEach(
        analysis => {


            const row =
                document.createElement(
                    "tr"
                );


            const vulnerable =
                analysis.predicted_label === 1;


            const verdictClass =
                vulnerable
                    ? "danger"
                    : "success";


            const createdAt =
                formatHistoryDate(
                    analysis.created_at
                );


            row.innerHTML = `

                <td>
                    #${analysis.id}
                </td>

                <td>

                    <div class="history-file-cell">

                        <div class="history-file-icon">
                            SOL
                        </div>

                        <div>

                            <strong>
                                ${escapeHtml(
                                    analysis.filename
                                )}
                            </strong>

                            <small>
                                ${formatFileSize(
                                    analysis.file_size_bytes
                                )}
                            </small>

                        </div>

                    </div>

                </td>


                <td>

                    <span
                        class="history-verdict ${verdictClass}"
                    >

                        ${
                            vulnerable
                            ? "VULNÉRABLE"
                            : "NON VULNÉRABLE"
                        }

                    </span>

                </td>


                <td>

                    <div class="history-confidence">

                        <strong>
                            ${Number(
                                analysis.confidence
                            ).toFixed(2)}%
                        </strong>

                        <div>

                            <span
                                style="
                                    width:
                                    ${Math.min(
                                        Number(
                                            analysis.confidence
                                        ),
                                        100
                                    )}%;
                                "
                            >
                            </span>

                        </div>

                    </div>

                </td>


                <td>

                    <span
                        class="
                            history-risk-badge
                            ${getHistoryRiskClass(
                                analysis.risk_score
                            )}
                        "
                    >

                        ${analysis.risk_score}/100

                    </span>

                </td>


                <td>

                    ${analysis.tokens_used}

                </td>


                <td>

                    <span class="history-date">
                        ${createdAt}
                    </span>

                </td>


                <td>

                    <div class="history-actions">

                        <button
                            class="history-action-button view"
                            data-view-id="${analysis.id}"
                            title="Voir les détails"
                        >
                            Voir
                        </button>


                        <button
                            class="history-action-button delete"
                            data-delete-id="${analysis.id}"
                            title="Supprimer"
                        >
                            ×
                        </button>

                    </div>

                </td>
            `;


            tbody.appendChild(
                row
            );
        }
    );


    attachHistoryRowEvents();
}


/* ========================================================= */
/* ÉVÉNEMENTS HISTORIQUE */
/* ========================================================= */

function attachHistoryEvents() {

    const newAnalysisButton =
        document.getElementById(
            "historyNewAnalysis"
        );


    if (newAnalysisButton) {

        newAnalysisButton.addEventListener(
            "click",
            showAnalysisView
        );
    }


    const refresh =
        document.getElementById(
            "refreshHistory"
        );


    if (refresh) {

        refresh.addEventListener(
            "click",
            loadHistory
        );
    }


    const search =
        document.getElementById(
            "historySearch"
        );


    if (search) {

        search.addEventListener(
            "input",
            renderHistoryTable
        );
    }


    document
        .querySelectorAll(
            ".history-filter"
        )
        .forEach(
            button => {


                button.addEventListener(
                    "click",
                    () => {


                        document
                            .querySelectorAll(
                                ".history-filter"
                            )
                            .forEach(
                                item =>
                                    item.classList.remove(
                                        "active"
                                    )
                            );


                        button.classList.add(
                            "active"
                        );


                        historyFilter =
                            button.dataset.filter;


                        renderHistoryTable();
                    }
                );
            }
        );


    const clearButton =
        document.getElementById(
            "clearHistoryButton"
        );


    if (clearButton) {

        clearButton.addEventListener(
            "click",
            clearCompleteHistory
        );
    }
}


/* ========================================================= */
/* ÉVÉNEMENTS LIGNES */
/* ========================================================= */

function attachHistoryRowEvents() {

    document
        .querySelectorAll(
            "[data-view-id]"
        )
        .forEach(
            button => {


                button.addEventListener(
                    "click",
                    () => {

                        showHistoryDetail(
                            button.dataset.viewId
                        );

                    }
                );
            }
        );


    document
        .querySelectorAll(
            "[data-delete-id]"
        )
        .forEach(
            button => {


                button.addEventListener(
                    "click",
                    () => {

                        deleteHistoryEntry(
                            button.dataset.deleteId
                        );

                    }
                );
            }
        );
}


/* ========================================================= */
/* DÉTAIL HISTORIQUE */
/* ========================================================= */

async function showHistoryDetail(
    id
) {

    try {

        const response =
            await fetch(
                `/api/history/${id}`
            );


        if (!response.ok) {

            throw new Error(
                "Impossible de charger cette analyse."
            );
        }


        const data =
            await response.json();


        const analysis =
            data.analysis;


        const panel =
            document.getElementById(
                "historyDetailPanel"
            );


        panel.classList.remove(
            "hidden"
        );


        document.getElementById(
            "historyDetailFilename"
        ).textContent =
            analysis.filename;


        document.getElementById(
            "historyDetailVerdict"
        ).textContent =
            analysis.predicted_label === 1
                ? "VULNÉRABLE"
                : "NON VULNÉRABLE";


        document.getElementById(
            "historyDetailConfidence"
        ).textContent =
            `${Number(
                analysis.confidence
            ).toFixed(2)}%`;


        document.getElementById(
            "historyDetailRisk"
        ).textContent =
            `${analysis.risk_score}/100`;


        document.getElementById(
            "historyDetailRiskLevel"
        ).textContent =
            (
                analysis.risk_level
                ||
                "-"
            ).toUpperCase();


        const vulnerableProbability =
            Number(
                analysis.probability_vulnerable
            );


        const safeProbability =
            Number(
                analysis.probability_non_vulnerable
            );


        document.getElementById(
            "historyVulnerableProbability"
        ).textContent =
            `${vulnerableProbability.toFixed(2)}%`;


        document.getElementById(
            "historySafeProbability"
        ).textContent =
            `${safeProbability.toFixed(2)}%`;


        document.getElementById(
            "historyVulnerableBar"
        ).style.width =
            `${vulnerableProbability}%`;


        document.getElementById(
            "historySafeBar"
        ).style.width =
            `${safeProbability}%`;


        document.getElementById(
            "historyDetailTokensDetected"
        ).textContent =
            analysis.tokens_detected;


        document.getElementById(
            "historyDetailTokensUsed"
        ).textContent =
            analysis.tokens_used;


        document.getElementById(
            "historyDetailUnknown"
        ).textContent =
            analysis.unknown_tokens;


        document.getElementById(
            "historyDetailTruncated"
        ).textContent =
            analysis.truncated
                ? "Oui"
                : "Non";


        document.getElementById(
            "historyDetailCode"
        ).textContent =
            analysis.code
            ||
            "Code indisponible.";


        const closeButton =
            document.getElementById(
                "closeHistoryDetail"
            );


        closeButton.onclick =
            () => {

                panel.classList.add(
                    "hidden"
                );
            };


        panel.scrollIntoView(
            {
                behavior:
                    "smooth",

                block:
                    "start"
            }
        );


    } catch (error) {

        alert(
            error.message
        );
    }
}


/* ========================================================= */
/* SUPPRESSION */
/* ========================================================= */

async function deleteHistoryEntry(
    id
) {

    const confirmed =
        confirm(
            "Voulez-vous supprimer cette analyse de l'historique ?"
        );


    if (!confirmed) {
        return;
    }


    try {

        const response =
            await fetch(
                `/api/history/${id}`,
                {
                    method:
                        "DELETE"
                }
            );


        if (!response.ok) {

            throw new Error(
                "Impossible de supprimer l'analyse."
            );
        }


        await loadHistory();


    } catch (error) {

        alert(
            error.message
        );
    }
}


/* ========================================================= */
/* TOUT SUPPRIMER */
/* ========================================================= */

async function clearCompleteHistory() {

    const confirmed =
        confirm(
            "Attention : toutes les analyses seront supprimées. Continuer ?"
        );


    if (!confirmed) {
        return;
    }


    try {

        const response =
            await fetch(
                "/api/history",
                {
                    method:
                        "DELETE"
                }
            );


        if (!response.ok) {

            throw new Error(
                "Impossible de supprimer l'historique."
            );
        }


        historyData = [];

        await loadHistory();


    } catch (error) {

        alert(
            error.message
        );
    }
}


/* ========================================================= */
/* UTILITAIRES HISTORIQUE */
/* ========================================================= */

function formatHistoryDate(
    value
) {

    if (!value) {
        return "-";
    }


    const date =
        new Date(
            value
        );


    if (
        Number.isNaN(
            date.getTime()
        )
    ) {

        return value;
    }


    return date.toLocaleString(
        "fr-FR",
        {
            day:
                "2-digit",

            month:
                "2-digit",

            year:
                "numeric",

            hour:
                "2-digit",

            minute:
                "2-digit"
        }
    );
}


function getHistoryRiskClass(
    score
) {

    score =
        Number(
            score
        );


    if (
        score >= 85
    ) {
        return "critical";
    }


    if (
        score >= 65
    ) {
        return "high";
    }


    if (
        score >= 40
    ) {
        return "medium";
    }


    return "low";
}


function escapeHtml(
    text
) {

    const div =
        document.createElement(
            "div"
        );


    div.textContent =
        text;


    return div.innerHTML;
}



/*
 * Rend le retour vers la page Analyse accessible aux vues
 * complémentaires chargées après app.js (ex. risk_view.js).
 */
window.showAnalysisView =
    showAnalysisView;


/* ========================================================= */
/* SIDEBAR EVENTS */
/* ========================================================= */

if (historyNav) {

    historyNav.addEventListener(
        "click",
        showHistoryView
    );
}


if (analysisNav) {

    analysisNav.addEventListener(
        "click",
        showAnalysisView
    );
}
