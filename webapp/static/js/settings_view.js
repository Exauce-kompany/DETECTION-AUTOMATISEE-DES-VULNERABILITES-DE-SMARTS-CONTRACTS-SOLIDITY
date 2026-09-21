
"use strict";

/* =========================================================
   SMART BUG - PARAMÈTRES
   ========================================================= */

const SMART_BUG_SETTINGS_KEY =
    "smartBugSettingsV1";


const SMART_BUG_DEFAULT_SETTINGS = {
    language: "fr",
    accent: "blue",
    compactMode: false,
    reducedMotion: false,
    rememberLastAnalysis: true
};


const SMART_BUG_ACCENTS = {
    blue: {
        label: "Bleu",
        primary: "#347fff",
        secondary: "#665eff",
        soft: "rgba(52, 127, 255, 0.12)",
        border: "rgba(74, 132, 255, 0.24)"
    },
    purple: {
        label: "Violet",
        primary: "#845cff",
        secondary: "#b45ef0",
        soft: "rgba(132, 92, 255, 0.12)",
        border: "rgba(145, 101, 255, 0.24)"
    },
    cyan: {
        label: "Cyan",
        primary: "#21b8d8",
        secondary: "#3b8dff",
        soft: "rgba(33, 184, 216, 0.12)",
        border: "rgba(50, 184, 224, 0.24)"
    },
    emerald: {
        label: "Émeraude",
        primary: "#28c77a",
        secondary: "#35d99a",
        soft: "rgba(40, 199, 122, 0.12)",
        border: "rgba(55, 207, 133, 0.24)"
    },
    ruby: {
        label: "Rubis",
        primary: "#eb556c",
        secondary: "#ff7b67",
        soft: "rgba(235, 85, 108, 0.12)",
        border: "rgba(235, 91, 111, 0.24)"
    }
};


const SMART_BUG_TRANSLATIONS = {
    fr: {
        settingsTitle: "Paramètres",
        settingsDescription:
            "Personnalisez l'apparence et le comportement local de SMART BUG.",
        languageTitle: "Langue",
        languageDescription:
            "Choisissez la langue du shell principal de l'application.",
        accentTitle: "Couleur d'accent",
        accentDescription:
            "Appliquez une identité visuelle différente au tableau de bord.",
        displayTitle: "Affichage",
        displayDescription:
            "Réglez la densité et les animations de l'interface.",
        privacyTitle: "Session locale",
        privacyDescription:
            "Contrôlez la conservation locale du dernier résultat.",
        compactLabel: "Mode compact",
        compactHelp: "Réduit légèrement les espacements de l'interface.",
        motionLabel: "Réduire les animations",
        motionHelp: "Désactive la majorité des transitions visuelles.",
        memoryLabel: "Mémoriser la dernière analyse",
        memoryHelp:
            "Permet aux vues Risques, Performances et Rapports de réutiliser le dernier audit.",
        reset: "Restaurer les paramètres",
        saved: "Paramètres enregistrés automatiquement",
        french: "Français",
        english: "English",
        german: "Deutsch",
        dashboard: "Dashboard",
        analysis: "Analyse",
        history: "Historique",
        risks: "Synthèse des risques",
        performance: "Performances",
        model: "Modèle IA",
        datasets: "Jeux de données",
        reports: "Rapports",
        settings: "Paramètres",
        about: "À propos"
    },

    en: {
        settingsTitle: "Settings",
        settingsDescription:
            "Customize the local appearance and behavior of SMART BUG.",
        languageTitle: "Language",
        languageDescription:
            "Choose the language of the application's main shell.",
        accentTitle: "Accent color",
        accentDescription:
            "Apply a different visual identity to the dashboard.",
        displayTitle: "Display",
        displayDescription:
            "Adjust interface density and motion.",
        privacyTitle: "Local session",
        privacyDescription:
            "Control local persistence of the latest result.",
        compactLabel: "Compact mode",
        compactHelp: "Slightly reduces interface spacing.",
        motionLabel: "Reduce motion",
        motionHelp: "Disables most visual transitions.",
        memoryLabel: "Remember latest analysis",
        memoryHelp:
            "Lets Risk, Performance and Reports reuse the most recent audit.",
        reset: "Reset settings",
        saved: "Settings are saved automatically",
        french: "Français",
        english: "English",
        german: "Deutsch",
        dashboard: "Dashboard",
        analysis: "Analysis",
        history: "History",
        risks: "Risk summary",
        performance: "Performance",
        model: "AI Model",
        datasets: "Datasets",
        reports: "Reports",
        settings: "Settings",
        about: "About"
    },

    de: {
        settingsTitle: "Einstellungen",
        settingsDescription:
            "Passen Sie das lokale Erscheinungsbild und Verhalten von SMART BUG an.",
        languageTitle: "Sprache",
        languageDescription:
            "Wählen Sie die Sprache der Hauptoberfläche.",
        accentTitle: "Akzentfarbe",
        accentDescription:
            "Wenden Sie eine andere visuelle Identität auf das Dashboard an.",
        displayTitle: "Anzeige",
        displayDescription:
            "Passen Sie Dichte und Animationen der Oberfläche an.",
        privacyTitle: "Lokale Sitzung",
        privacyDescription:
            "Steuern Sie die lokale Speicherung des letzten Ergebnisses.",
        compactLabel: "Kompaktmodus",
        compactHelp: "Reduziert die Abstände der Oberfläche leicht.",
        motionLabel: "Animationen reduzieren",
        motionHelp: "Deaktiviert die meisten visuellen Übergänge.",
        memoryLabel: "Letzte Analyse merken",
        memoryHelp:
            "Erlaubt Risiko-, Performance- und Berichtsansichten, das letzte Audit wiederzuverwenden.",
        reset: "Einstellungen zurücksetzen",
        saved: "Einstellungen werden automatisch gespeichert",
        french: "Français",
        english: "English",
        german: "Deutsch",
        dashboard: "Dashboard",
        analysis: "Analyse",
        history: "Verlauf",
        risks: "Risikoübersicht",
        performance: "Performance",
        model: "KI-Modell",
        datasets: "Datensätze",
        reports: "Berichte",
        settings: "Einstellungen",
        about: "Über"
    }
};


function smartBugLoadSettings() {
    try {
        const stored =
            localStorage.getItem(
                SMART_BUG_SETTINGS_KEY
            );

        if (!stored) {
            return {
                ...SMART_BUG_DEFAULT_SETTINGS
            };
        }

        return {
            ...SMART_BUG_DEFAULT_SETTINGS,
            ...JSON.parse(stored)
        };

    } catch (error) {

        console.warn(
            "SMART BUG : paramètres locaux illisibles.",
            error
        );

        return {
            ...SMART_BUG_DEFAULT_SETTINGS
        };
    }
}


let smartBugSettings =
    smartBugLoadSettings();


function smartBugSaveSettings() {
    try {
        localStorage.setItem(
            SMART_BUG_SETTINGS_KEY,
            JSON.stringify(
                smartBugSettings
            )
        );
    } catch (error) {
        console.warn(
            "SMART BUG : impossible de sauvegarder les paramètres.",
            error
        );
    }
}


function smartBugTranslation() {
    return (
        SMART_BUG_TRANSLATIONS[
            smartBugSettings.language
        ]
        ||
        SMART_BUG_TRANSLATIONS.fr
    );
}


function smartBugApplyAccent() {

    const accent =
        SMART_BUG_ACCENTS[
            smartBugSettings.accent
        ]
        ||
        SMART_BUG_ACCENTS.blue;

    const root =
        document.documentElement;

    root.style.setProperty(
        "--sb-accent-primary",
        accent.primary
    );

    root.style.setProperty(
        "--sb-accent-secondary",
        accent.secondary
    );

    root.style.setProperty(
        "--sb-accent-soft",
        accent.soft
    );

    root.style.setProperty(
        "--sb-accent-border",
        accent.border
    );

    document.body.dataset.smartBugAccent =
        smartBugSettings.accent;
}


function smartBugApplyDisplaySettings() {

    document.body.classList.toggle(
        "smartbug-compact",
        Boolean(
            smartBugSettings.compactMode
        )
    );

    document.body.classList.toggle(
        "smartbug-reduced-motion",
        Boolean(
            smartBugSettings.reducedMotion
        )
    );
}


function smartBugApplyLanguage() {

    const t =
        smartBugTranslation();

    document.documentElement.lang =
        smartBugSettings.language;

    const navItems =
        Array.from(
            document.querySelectorAll(
                ".nav-item"
            )
        );

    const replacements = [
        ["dashboard", t.dashboard],
        ["analyse", t.analysis],
        ["analysis", t.analysis],
        ["historique", t.history],
        ["history", t.history],
        ["synthèse des risques", t.risks],
        ["risk summary", t.risks],
        ["risikoübersicht", t.risks],
        ["performances", t.performance],
        ["performance", t.performance],
        ["modèle ia", t.model],
        ["ai model", t.model],
        ["ki-modell", t.model],
        ["jeux de données", t.datasets],
        ["datasets", t.datasets],
        ["datensätze", t.datasets],
        ["rapports", t.reports],
        ["reports", t.reports],
        ["berichte", t.reports],
        ["paramètres", t.settings],
        ["settings", t.settings],
        ["einstellungen", t.settings],
        ["à propos", t.about],
        ["about", t.about],
        ["über", t.about]
    ];

    navItems.forEach(
        item => {

            const label =
                item.querySelector(
                    "span:last-child"
                );

            if (!label) {
                return;
            }

            const current =
                label.textContent
                    .trim()
                    .toLowerCase();

            const entry =
                replacements.find(
                    ([source]) =>
                        current === source
                );

            if (entry) {
                label.textContent =
                    entry[1];
            }
        }
    );
}


function smartBugApplySettings() {
    smartBugApplyAccent();
    smartBugApplyDisplaySettings();
    smartBugApplyLanguage();
}


function settingsSetActiveNavigation() {

    const settingsNav =
        Array.from(
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
                    text.includes("paramètres")
                    ||
                    text.includes("settings")
                    ||
                    text.includes("einstellungen")
                );
            }
        );

    document
        .querySelectorAll(".nav-item")
        .forEach(
            item =>
                item.classList.remove(
                    "active"
                )
        );

    if (settingsNav) {
        settingsNav.classList.add(
            "active"
        );
    }
}


function settingsToggleRow(
    id,
    title,
    help,
    checked
) {
    return `
        <label
            class="settings-toggle-row"
            for="${id}"
        >

            <div>
                <strong>
                    ${title}
                </strong>

                <span>
                    ${help}
                </span>
            </div>

            <div class="settings-switch">
                <input
                    id="${id}"
                    type="checkbox"
                    ${checked ? "checked" : ""}
                >

                <span></span>
            </div>

        </label>
    `;
}


function showSettingsView() {

    const container =
        document.querySelector(
            ".content"
        );

    if (!container) {
        return;
    }

    settingsSetActiveNavigation();

    const t =
        smartBugTranslation();


    container.innerHTML = `
        <section class="settings-page">

            <div class="settings-page-header">

                <div>
                    <span class="settings-eyebrow">
                        SMART BUG CONTROL CENTER
                    </span>

                    <h1>
                        ${t.settingsTitle}
                    </h1>

                    <p>
                        ${t.settingsDescription}
                    </p>
                </div>

                <div class="settings-auto-save">
                    <span></span>
                    ${t.saved}
                </div>

            </div>


            <section class="settings-grid">

                <article class="settings-panel">

                    <div class="settings-panel-heading">
                        <span>01</span>

                        <div>
                            <h2>
                                ${t.languageTitle}
                            </h2>

                            <p>
                                ${t.languageDescription}
                            </p>
                        </div>
                    </div>


                    <div class="settings-language-grid">

                        <button
                            class="settings-language-button ${
                                smartBugSettings.language === "fr"
                                    ? "active"
                                    : ""
                            }"
                            data-language="fr"
                            type="button"
                        >
                            <strong>FR</strong>
                            <span>
                                ${t.french}
                            </span>
                        </button>

                        <button
                            class="settings-language-button ${
                                smartBugSettings.language === "en"
                                    ? "active"
                                    : ""
                            }"
                            data-language="en"
                            type="button"
                        >
                            <strong>EN</strong>
                            <span>
                                ${t.english}
                            </span>
                        </button>

                        <button
                            class="settings-language-button ${
                                smartBugSettings.language === "de"
                                    ? "active"
                                    : ""
                            }"
                            data-language="de"
                            type="button"
                        >
                            <strong>DE</strong>
                            <span>
                                ${t.german}
                            </span>
                        </button>

                    </div>

                    <div class="settings-language-note">
                        Les pages techniques existantes restent principalement
                        en français pour cette version ; le shell principal
                        et cette page changent déjà de langue.
                    </div>

                </article>


                <article class="settings-panel">

                    <div class="settings-panel-heading">
                        <span>02</span>

                        <div>
                            <h2>
                                ${t.accentTitle}
                            </h2>

                            <p>
                                ${t.accentDescription}
                            </p>
                        </div>
                    </div>


                    <div class="settings-accent-grid">

                        ${Object.entries(
                            SMART_BUG_ACCENTS
                        ).map(
                            ([key, accent]) => `
                                <button
                                    class="settings-accent-button ${
                                        smartBugSettings.accent === key
                                            ? "active"
                                            : ""
                                    }"
                                    data-accent="${key}"
                                    type="button"
                                    style="
                                        --preview-a:${accent.primary};
                                        --preview-b:${accent.secondary};
                                    "
                                >
                                    <span></span>
                                    <strong>
                                        ${accent.label}
                                    </strong>
                                </button>
                            `
                        ).join("")}

                    </div>

                </article>


                <article class="settings-panel">

                    <div class="settings-panel-heading">
                        <span>03</span>

                        <div>
                            <h2>
                                ${t.displayTitle}
                            </h2>

                            <p>
                                ${t.displayDescription}
                            </p>
                        </div>
                    </div>


                    <div class="settings-toggle-list">

                        ${settingsToggleRow(
                            "settingsCompactMode",
                            t.compactLabel,
                            t.compactHelp,
                            smartBugSettings.compactMode
                        )}

                        ${settingsToggleRow(
                            "settingsReducedMotion",
                            t.motionLabel,
                            t.motionHelp,
                            smartBugSettings.reducedMotion
                        )}

                    </div>

                </article>


                <article class="settings-panel">

                    <div class="settings-panel-heading">
                        <span>04</span>

                        <div>
                            <h2>
                                ${t.privacyTitle}
                            </h2>

                            <p>
                                ${t.privacyDescription}
                            </p>
                        </div>
                    </div>


                    <div class="settings-toggle-list">

                        ${settingsToggleRow(
                            "settingsRememberAnalysis",
                            t.memoryLabel,
                            t.memoryHelp,
                            smartBugSettings.rememberLastAnalysis
                        )}

                    </div>


                    <div class="settings-session-box">

                        <div>
                            <span>
                                STOCKAGE
                            </span>

                            <strong>
                                localStorage + sessionStorage
                            </strong>
                        </div>

                        <p>
                            Les préférences restent locales au navigateur.
                            Les analyses historiques continuent d'être
                            enregistrées dans SQLite par SMART BUG.
                        </p>

                    </div>

                </article>

            </section>


            <div class="settings-footer-actions">

                <button
                    id="settingsResetButton"
                    class="settings-reset-button"
                    type="button"
                >
                    ${t.reset}
                </button>

                <span>
                    SMART BUG · Local preferences
                </span>

            </div>

        </section>
    `;


    document
        .querySelectorAll(
            ".settings-language-button"
        )
        .forEach(
            button => {

                button.addEventListener(
                    "click",
                    () => {

                        smartBugSettings.language =
                            button.dataset.language;

                        smartBugSaveSettings();
                        smartBugApplySettings();
                        showSettingsView();
                    }
                );
            }
        );


    document
        .querySelectorAll(
            ".settings-accent-button"
        )
        .forEach(
            button => {

                button.addEventListener(
                    "click",
                    () => {

                        smartBugSettings.accent =
                            button.dataset.accent;

                        smartBugSaveSettings();
                        smartBugApplySettings();
                        showSettingsView();
                    }
                );
            }
        );


    const compact =
        document.getElementById(
            "settingsCompactMode"
        );

    if (compact) {
        compact.addEventListener(
            "change",
            () => {
                smartBugSettings.compactMode =
                    compact.checked;

                smartBugSaveSettings();
                smartBugApplySettings();
            }
        );
    }


    const motion =
        document.getElementById(
            "settingsReducedMotion"
        );

    if (motion) {
        motion.addEventListener(
            "change",
            () => {
                smartBugSettings.reducedMotion =
                    motion.checked;

                smartBugSaveSettings();
                smartBugApplySettings();
            }
        );
    }


    const remember =
        document.getElementById(
            "settingsRememberAnalysis"
        );

    if (remember) {
        remember.addEventListener(
            "change",
            () => {

                smartBugSettings.rememberLastAnalysis =
                    remember.checked;

                if (!remember.checked) {
                    try {
                        sessionStorage.removeItem(
                            "smartBugLastAnalysis"
                        );
                    } catch (error) {
                        console.warn(error);
                    }
                }

                smartBugSaveSettings();
                smartBugApplySettings();
            }
        );
    }


    const reset =
        document.getElementById(
            "settingsResetButton"
        );

    if (reset) {
        reset.addEventListener(
            "click",
            () => {

                smartBugSettings = {
                    ...SMART_BUG_DEFAULT_SETTINGS
                };

                smartBugSaveSettings();
                smartBugApplySettings();
                showSettingsView();
            }
        );
    }
}


function bindSettingsNavigation() {

    const settingsNav =
        Array.from(
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
                    text.includes("paramètres")
                    ||
                    text.includes("settings")
                    ||
                    text.includes("einstellungen")
                );
            }
        );

    if (
        settingsNav
        &&
        !settingsNav.dataset.settingsBound
    ) {

        settingsNav.dataset.settingsBound =
            "true";

        settingsNav.addEventListener(
            "click",
            showSettingsView
        );
    }
}


smartBugApplySettings();
bindSettingsNavigation();


window.showSettingsView =
    showSettingsView;

window.smartBugApplySettings =
    smartBugApplySettings;

window.smartBugSettings =
    smartBugSettings;
