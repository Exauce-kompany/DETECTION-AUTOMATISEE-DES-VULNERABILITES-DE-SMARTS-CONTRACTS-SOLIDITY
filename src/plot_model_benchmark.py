"""Export French benchmark figures without training, selection or recalibration.

The default mode requires all three evaluations before exporting test figures.
--available-models exports only families already declared evaluated, after
verifying their saved checkpoints, calibration and predictions. It does not
train, select a checkpoint, calibrate again or fabricate pending model scores.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = "cnn-xgboost-tcn-20261005"
MODELS = ("cnn_bilstm", "xgboost", "tcn")
NAMES = {"cnn_bilstm": "CNN–BiLSTM actif", "xgboost": "TF-IDF + XGBoost", "tcn": "TCN depuis zéro"}
COLORS = {"cnn_bilstm": "#0072B2", "xgboost": "#009E73", "tcn": "#E69F00"}
METRICS = ("f1_macro", "precision_vulnerable", "recall_vulnerable", "false_positive_rate", "average_precision")
TEST_PLOTS = ("metriques_principales", "matrices_confusion", "courbe_roc", "courbe_precision_rappel")
EXTRA_PLOTS = ("ecarts_f1_ic95", "pertes_tcn_selectionne")


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def relative(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_bytes(path, sources):
    path = Path(path)
    value = path.read_bytes()
    sources[relative(path)] = hashlib.sha256(value).hexdigest()
    return value


def read_json(path, sources):
    return json.loads(read_bytes(path, sources).decode("utf-8-sig"))


def save_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def configure_style():
    for color in COLORS.values():
        to_rgba(color)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 13, "axes.labelsize": 11,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})


def footer(config, *, test=False):
    label = "SMART BUG · snapshot " + config["dataset_id"][:12]
    if test:
        return label + " · test déjà observé : étude exploratoire\nCheckpoints sélectionnés sur validation ; température et seuil fixés sur calibration."
    return label + " · étude exploratoire · aucune sélection sur le test"


def export(fig, output, name, config, manifest, *, test=False, note=None, fixed_canvas=False):
    text = footer(config, test=test)
    if test and manifest.get("excluded_models"):
        text += "\nModèles non évalués omis : " + ", ".join(NAMES[name] for name in manifest["excluded_models"]) + "."
    if note:
        text += "\n" + note
    fig.text(0.5, 0.025, text, ha="center", va="bottom", fontsize=8, color="#444444")
    files = []
    for extension in ("png", "svg"):
        path = output / (name + "." + extension)
        fig.savefig(path, dpi=300, bbox_inches=None if fixed_canvas else "tight")
        files.append({"path": relative(path), "sha256": file_hash(path), "format": extension,
                      **({"dpi": 300} if extension == "png" else {})})
    plt.close(fig)
    manifest["plots"][name] = {"files": files, "uses_test": test,
                               "canvas_inches": [float(value) for value in fig.get_size_inches()],
                               "fixed_canvas": fixed_canvas}


def plot_architecture(config, output, manifest):
    settings = config["tcn"]
    dilations = [int(value) for value in settings["dilations"]]
    receptive = 1 + int(settings["convolutions_per_block"]) * (int(settings["kernel_size"]) - 1) * sum(dilations)
    if not dilations or any(value <= 0 for value in dilations):
        raise ValueError("Les dilatations doivent être strictement positives.")
    boxes = [
        (1.85, "Tokens Solidity\nnormalisés\nSéquence entière", "#EAF2F8"),
        (1.85, f"Embedding {settings['embedding_dim']}\nPAD = 0 masqué", "#D6EAF8"),
        (3.45, f"{len(dilations)} blocs résiduels\n{settings['convolutions_per_block']} Conv1D causales dilatées / bloc\n{settings['channels']} canaux · noyau {settings['kernel_size']} · ReLU\nd = " + ", ".join(map(str, dilations)), "#FBEBC8"),
        (2.05, "Pooling global masqué\nMoyenne + maximum", "#D5F5E3"),
        (1.90, f"Dense {settings['dense_units']}\nReLU", "#D6EAF8"),
        (1.40, "Logit\ndu contrat", "#FBEBC8"),
    ]
    fig, ax = plt.subplots(figsize=(14.2, 4.1))
    gap, left = 0.3, 0.0
    total = sum(box[0] for box in boxes) + gap * (len(boxes) - 1)
    for index, (width, label, color) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((left, 0.45), width, 1.1, boxstyle="round,pad=0.035", linewidth=1.2,
                                   edgecolor="#485460", facecolor=color))
        ax.text(left + width / 2, 1, label, ha="center", va="center", fontsize=9.3)
        if index < len(boxes) - 1:
            ax.annotate("", xy=(left + width + gap - 0.04, 1), xytext=(left + width + 0.04, 1),
                        arrowprops={"arrowstyle": "->", "color": "#485460", "lw": 1.5})
        left += width + gap
    ax.set(xlim=(-0.12, total + 0.12), ylim=(0, 2.0))
    ax.axis("off")
    ax.set_title("Architecture du TCN — apprentissage depuis zéro", pad=14)
    ax.text(total / 2, 0.08, f"Champ réceptif local : {receptive:,} positions encodées".replace(",", " ") +
            f" · dropout {settings['dropout']:.2f} après les convolutions et avant la tête\n"
            "Perte binaire au contrat · tous les tokens encodés contribuent au pooling", ha="center", va="center", fontsize=9)
    fig.subplots_adjust(bottom=0.20, top=0.82, left=0.02, right=0.98)
    export(fig, output, "architecture_tcn", config, manifest)
    manifest["architecture"] = {"settings": settings, "receptive_field_encoded_positions": receptive,
                                "initialization": "from_scratch", "truncation": False}


def plot_validation(config, result, output, sources, manifest):
    histories, pending = {}, []
    for seed in config["seeds"]:
        path = result / "tcn" / f"seed-{seed}" / "history.json"
        if not path.exists():
            pending.append(int(seed))
            continue
        history = read_json(path, sources)
        if not history:
            pending.append(int(seed))
            continue
        epochs = [int(row["epoch"]) for row in history]
        losses = [float(row["validation_log_loss"]) for row in history]
        if epochs != sorted(set(epochs)) or not np.isfinite(losses).all() or min(losses) < 0:
            raise ValueError(f"Historique de validation invalide pour la graine {seed}.")
        histories[int(seed)] = (epochs, losses)
    manifest["validation"] = {"available_seeds": list(histories), "pending_seeds": pending,
                              "criterion": "minimum_unweighted_validation_log_loss", "best_epochs": {}}
    if not histories:
        manifest["skipped"]["logloss_validation_tcn"] = "Aucune époque de validation enregistrée."
        return
    fig, ax = plt.subplots(figsize=(9, 5.4))
    seed_colors = (COLORS["cnn_bilstm"], COLORS["xgboost"], COLORS["tcn"])
    for index, seed in enumerate(config["seeds"]):
        if int(seed) not in histories:
            continue
        epochs, losses = histories[int(seed)]
        best = int(np.argmin(losses))
        color = seed_colors[index % len(seed_colors)]
        ax.plot(epochs, losses, color=color, marker="o", linewidth=2, label=f"Graine {seed} — meilleure époque {epochs[best]}")
        ax.scatter([epochs[best]], [losses[best]], s=125, facecolors="none", edgecolors=color, linewidths=2, zorder=5)
        manifest["validation"]["best_epochs"][str(seed)] = {"epoch": epochs[best], "validation_log_loss": losses[best]}
    ax.set(title="TCN : log-loss sur validation", xlabel="Époque", ylabel="Log-loss non pondérée (plus bas = meilleur)",
           xlim=(0.7, max(2, max(max(values[0]) for values in histories.values())) + 0.3))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=9)
    fig.subplots_adjust(bottom=0.24, top=0.88)
    note = "Historiques partiels ; les meilleures époques sont provisoires." if pending or not (result / "tcn" / "selection.json").exists() else "La meilleure époque est déterminée sur validation, avant calibration."
    export(fig, output, "logloss_validation_tcn", config, manifest, note=note)


def evaluated_models(summary):
    """Eligibility depends on saved evaluation status, never on test scores."""
    if summary is None:
        return ()
    return tuple(name for name in MODELS if summary.get("models", {}).get(name, {}).get("status") == "evaluated")


def completion_gate(summary, result, models=MODELS, *, require_complete=True):
    """This gate must run before opening any test labels, IDs or logits."""
    reasons = []
    if require_complete and (summary is None or summary.get("complete") is not True):
        reasons.append("summary.json ne déclare pas les trois évaluations terminées.")
    if not models:
        reasons.append("Aucun modèle n'est déclaré évalué.")
    for name in models:
        entry = (summary or {}).get("models", {}).get(name, {})
        if entry.get("status") != "evaluated":
            reasons.append(NAMES[name] + " n'est pas déclaré évalué.")
        if not entry.get("metrics") or not entry.get("selected"):
            reasons.append(NAMES[name] + " n'a pas de métriques et checkpoint enregistrés dans le résumé.")
        for filename in ("evaluation.json", "selection.json", "calibration.json", "test_sample_ids.json", "test_logits.npy"):
            if not (result / name / filename).is_file():
                reasons.append(f"Fichier manquant : {name}/{filename}.")
    return reasons


def load_complete_test(config, result, summary, sources, manifest, models=MODELS):
    """Validate stored selection/calibration, then load the identical test set."""
    from scipy.special import expit
    from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

    evaluations = {}
    for name in models:
        evaluation = read_json(result / name / "evaluation.json", sources)
        selection = read_json(result / name / "selection.json", sources)
        calibration = read_json(result / name / "calibration.json", sources)
        if (evaluation.get("model") != name or evaluation["selected"] != selection["selected"]
                or evaluation["selected"] != summary["models"][name]["selected"]):
            raise ValueError(f"Checkpoint sélectionné incohérent : {name}.")
        if selection.get("fit_split") != "validation":
            raise ValueError(f"La sélection de {name} n'est pas issue de validation.")
        if evaluation["calibration"] != calibration or calibration.get("fit_split") != "calibration":
            raise ValueError(f"Calibration incohérente : {name}.")
        temperature, threshold = float(calibration["temperature"]), float(calibration["threshold"])
        if not np.isfinite([temperature, threshold]).all() or temperature <= 0 or not 0 <= threshold <= 1:
            raise ValueError(f"Température ou seuil invalide : {name}.")
        point = evaluation["splits"]["test"]["calibrated"]
        if float(point["threshold"]) != threshold:
            raise ValueError(f"Le seuil évalué diffère du seuil calibré : {name}.")
        for key in METRICS:
            value = float(point[key])
            if not np.isfinite(value) or not 0 <= value <= 1 or not np.isclose(value, summary["models"][name]["metrics"][key], rtol=0, atol=1e-12):
                raise ValueError(f"Métrique ou résumé incohérent : {name}/{key}.")
        for artifact in ("model", "vectorizer"):
            selected = evaluation["selected"]
            if artifact + "_path" not in selected:
                continue
            path = ROOT / selected[artifact + "_path"]
            actual = file_hash(path)
            if actual != selected[artifact + "_sha256"]:
                raise ValueError(f"Artefact sélectionné modifié : {name}/{artifact}.")
            sources[relative(path)] = actual
        evaluations[name] = evaluation
    dataset = ROOT / config["dataset_dir"]
    dataset_manifest = read_json(dataset / "manifest.json", sources)
    if dataset_manifest["dataset_id"] != config["dataset_id"]:
        raise ValueError("Le snapshot ne correspond pas au protocole.")
    test_path = dataset / "test.jsonl.gz"
    compressed = read_bytes(test_path, sources)
    if hashlib.sha256(compressed).hexdigest() != dataset_manifest["files"]["test.jsonl.gz"]:
        raise ValueError("Le fichier de test diffère du snapshot.")
    # This is the first operation that accesses individual test annotations.
    rows = [json.loads(line) for line in gzip.decompress(compressed).decode("utf-8").splitlines() if line.strip()]
    sample_ids = [row["sample_id"] for row in rows]
    labels = np.asarray([row["label"] for row in rows], dtype=np.int32)
    if not len(labels) or len(set(sample_ids)) != len(sample_ids) or set(labels.tolist()) != {0, 1}:
        raise ValueError("Le test doit contenir des identifiants uniques et les deux classes.")
    probabilities = {}
    manifest["selected_checkpoints"] = {}
    manifest["metrics"] = {}
    for name, evaluation in evaluations.items():
        saved_ids = read_json(result / name / "test_sample_ids.json", sources)
        if saved_ids != sample_ids:
            raise ValueError(f"Ordre des identifiants du test incohérent : {name}.")
        logits = np.load(BytesIO(read_bytes(result / name / "test_logits.npy", sources)), allow_pickle=False)
        if logits.shape != labels.shape or not np.isfinite(logits).all():
            raise ValueError(f"Prédictions de test invalides : {name}.")
        calibration = evaluation["calibration"]
        probability = expit(logits.astype(np.float64) / float(calibration["temperature"]))
        predictions = (probability >= calibration["threshold"]).astype(np.int32)
        matrix = confusion_matrix(labels, predictions, labels=[0, 1])
        tn, fp, _, _ = matrix.ravel()
        calculated = {"f1_macro": f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0),
                      "precision_vulnerable": precision_score(labels, predictions, zero_division=0),
                      "recall_vulnerable": recall_score(labels, predictions, zero_division=0),
                      "false_positive_rate": fp / (tn + fp), "average_precision": average_precision_score(labels, probability),
                      "roc_auc": roc_auc_score(labels, probability)}
        point = evaluation["splits"]["test"]["calibrated"]
        if not np.array_equal(matrix, np.asarray(point["confusion_matrix"])) or int(point["samples"]) != len(labels):
            raise ValueError(f"Matrice ou nombre de contrats incohérent : {name}.")
        for key, value in calculated.items():
            if not np.isclose(value, point[key], rtol=0, atol=1e-10):
                raise ValueError(f"Logits, calibration et métriques ne concordent pas : {name}/{key}.")
        probabilities[name] = probability
        manifest["selected_checkpoints"][name] = {"selected": evaluation["selected"], "temperature": calibration["temperature"],
                                                  "threshold": calibration["threshold"], "threshold_fit_split": "calibration"}
        manifest["metrics"][name] = {key: float(point[key]) for key in (*METRICS, "roc_auc")}
        manifest["metrics"][name]["confusion_matrix"] = matrix.tolist()
    manifest["test"] = {"samples": len(labels), "sample_ids_sha256": hashlib.sha256(json.dumps(sample_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest(),
                        "order_verified_for": list(models), "prediction_rule": "sigmoid(logit / temperature) >= calibration_threshold"}
    return labels, probabilities, evaluations


def legend_name(name, evaluation):
    return NAMES[name] + f" (graine {evaluation['selected']['seed']})"


def plot_paired_differences(config, result, output, sources, manifest, summary):
    """Plot the saved cluster bootstrap; never perform a new bootstrap here."""
    path = result / "paired_comparisons.json"
    if not path.exists():
        manifest["skipped"]["ecarts_f1_ic95"] = "Intervalles appariés non encore enregistrés."
        return
    paired = read_json(path, sources)
    evaluation_hashes = {name: sources[relative(result / name / "evaluation.json")] for name in MODELS}
    fingerprint = hashlib.sha256(json.dumps(evaluation_hashes, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    if paired["evaluations_id"] != fingerprint or paired != summary.get("paired_comparisons"):
        raise ValueError("Les intervalles appariés ne correspondent pas aux évaluations complètes.")
    if paired.get("sampling_unit") != "group_id" or int(paired["repetitions"]) <= 0:
        raise ValueError("Le bootstrap enregistré doit utiliser les groupes du dataset.")
    pairs = paired["pairs"]
    expected = {frozenset((left, right)) for index, left in enumerate(MODELS) for right in MODELS[index + 1:]}
    if len(pairs) != 3 or {frozenset((pair["left"], pair["right"])) for pair in pairs} != expected:
        raise ValueError("Les trois paires attendues ne sont pas présentes.")
    rows = []
    for pair in pairs:
        left, right = pair["left"], pair["right"]
        delta = float(pair["f1_macro"]["difference"])
        interval = np.asarray(pair["f1_macro"]["ci95"], dtype=float)
        if interval.shape != (2,) or not np.isfinite(interval).all() or interval[0] > interval[1] or np.any(abs(interval) > 1):
            raise ValueError("Intervalle de F1 invalide.")
        expected_delta = manifest["metrics"][left]["f1_macro"] - manifest["metrics"][right]["f1_macro"]
        if not np.isclose(delta, expected_delta, rtol=0, atol=1e-10):
            raise ValueError("L'écart enregistré ne correspond pas aux checkpoints sélectionnés.")
        rows.append((left, right, 100 * delta, 100 * interval))
    fig, (ax, annotations) = plt.subplots(1, 2, figsize=(9, 4.1), gridspec_kw={"width_ratios": [2.1, 1.15]}, sharey=True)
    ys = np.arange(len(rows))[::-1]
    extent = max(0.5, max(abs(value) for _, _, delta, interval in rows for value in (delta, *interval))) * 1.15
    for y, (left, right, delta, interval) in zip(ys, rows):
        color = COLORS[left]
        ax.hlines(y, interval[0], interval[1], color=color, linewidth=2.4)
        ax.vlines(interval, y - 0.08, y + 0.08, color=color, linewidth=1.5)
        ax.scatter([delta], [y], color=color, s=55, zorder=3)
        annotations.text(0, y, f"{delta:+.2f} [{interval[0]:+.2f} ; {interval[1]:+.2f}]", va="center", fontsize=9)
    ax.axvline(0, color="#555555", linestyle="--", linewidth=1)
    ax.set_yticks(ys, [NAMES[left] + " − " + NAMES[right] for left, right, _, _ in rows], fontsize=8.7)
    ax.set(xlim=(-extent, extent), ylim=(-0.5, 2.6), xlabel="Différence de F1 macro (points)")
    ax.grid(axis="x", alpha=0.2)
    annotations.set_xlim(0, 1)
    annotations.axis("off")
    annotations.text(0, 2.53, "Écart [IC 95 %], points", fontsize=8.7, fontweight="bold")
    fig.suptitle("Écarts appariés de F1 macro — checkpoints sélectionnés", fontsize=12)
    fig.subplots_adjust(left=0.30, right=0.98, bottom=0.33, top=0.82, wspace=0.10)
    export(fig, output, "ecarts_f1_ic95", config, manifest, test=True,
           note=f"Bootstrap par groupe déjà calculé ({paired['repetitions']} réplications) ; checkpoints fixés, sans correction des comparaisons multiples.",
           fixed_canvas=True)
    manifest["paired_comparisons"] = paired


def plot_selected_loss(config, result, output, sources, manifest, evaluations):
    selected = evaluations["tcn"]["selected"]
    seed, best_epoch = int(selected["seed"]), int(selected["best_epoch"])
    path = result / "tcn" / f"seed-{seed}" / "history.json"
    if not path.exists():
        manifest["skipped"]["pertes_tcn_selectionne"] = "Historique de la graine TCN sélectionnée absent."
        return
    history = read_json(path, sources)
    if not history:
        raise ValueError("Le TCN retenu doit avoir un historique d'apprentissage.")
    epochs = np.asarray([row["epoch"] for row in history], dtype=int)
    training = np.asarray([row["loss"] for row in history], dtype=float)
    validation = np.asarray([row["validation_log_loss"] for row in history], dtype=float)
    if list(epochs) != sorted(set(epochs)) or not np.isfinite([training, validation]).all() or np.any(training < 0) or np.any(validation < 0):
        raise ValueError("Historique de pertes TCN invalide.")
    index = int(np.argmin(validation))
    if int(epochs[index]) != best_epoch or not np.isclose(validation[index], selected["validation"]["log_loss"], rtol=0, atol=1e-10):
        raise ValueError("La meilleure époque ne correspond pas au TCN sélectionné.")
    fig, ax = plt.subplots(figsize=(9, 4.1))
    color = COLORS["tcn"]
    ax.plot(epochs, training, color=color, linestyle="--", linewidth=2, alpha=0.6,
            label="Train : BCE pondérée, mode apprentissage")
    ax.plot(epochs, validation, color=color, linestyle="-", marker="o", linewidth=2,
            label="Validation : log-loss non pondérée, mode évaluation")
    ax.axvline(best_epoch, color="#555555", linestyle=":", linewidth=1.3,
               label=f"Meilleure époque sur validation : {best_epoch}")
    ax.scatter([best_epoch], [validation[index]], s=115, facecolors="none", edgecolors=color, linewidths=2, zorder=5)
    ax.set(title=f"TCN retenu (graine {seed}) — pertes par époque", xlabel="Époque", ylabel="Perte au contrat",
           xlim=(0.7, max(2, int(epochs.max())) + 0.3), ylim=(0, None))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8.3, loc="best")
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.29, top=0.85)
    export(fig, output, "pertes_tcn_selectionne", config, manifest,
           note="Pertes de nature distincte : pondération et dropout actifs sur train ; validation non pondérée sans dropout.",
           fixed_canvas=True)
    manifest["selected_loss"] = {"seed": seed, "best_epoch": best_epoch,
                                  "training_loss": "mean_weighted_BCE_in_training_mode",
                                  "validation_loss": "unweighted_log_loss_in_evaluation_mode"}


def draw_confusion(ax, name, matrix, calibration, maximum):
    classes = ["Sans vulnérabilité\nannotée (0)", "Vulnérabilité\nannotée (1)"]
    cmap = LinearSegmentedColormap.from_list(name, ["#FFFFFF", COLORS[name]])
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=maximum)
    for (row, column), value in np.ndenumerate(matrix):
        symbol = (("VN", "FP"), ("FN", "VP"))[row][column]
        ax.text(column, row, f"{symbol}\n{value}", ha="center", va="center", fontsize=13,
                color="white" if value > 0.6 * maximum else "#222222")
    ax.set_xticks([0, 1], classes, fontsize=8)
    ax.set_yticks([0, 1], classes, fontsize=8)
    ax.set(xlabel="Classe prédite", ylabel="Annotation réelle")
    ax.set_title(NAMES[name] + f"\nT = {calibration['temperature']:.3f} · seuil = {calibration['threshold']:.3f}", fontsize=11)
    ax.tick_params(length=0)


def plot_test(config, output, manifest, labels, probabilities, evaluations, models=MODELS):
    from sklearn.metrics import precision_recall_curve, roc_curve

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    x, width = np.arange(len(METRICS)), 0.24 if len(models) == 3 else 0.32
    for index, name in enumerate(models):
        values = [manifest["metrics"][name][key] for key in METRICS]
        bars = ax.bar(x + (index - (len(models) - 1) / 2) * width, values, width, color=COLORS[name], label=legend_name(name, evaluations[name]))
        for bar, value in zip(bars, values):
            high = value > 0.94
            ax.text(bar.get_x() + bar.get_width() / 2, value - 0.055 if high else value + 0.013,
                    f"{value:.3f}", ha="center", va="center" if high else "bottom", fontsize=8,
                    color="white" if high else "#333333")
    ax.set_xticks(x, ["F1 macro ↑", "Précision\npositive ↑", "Rappel\npositif ↑", "FPR ↓", "PR-AUC\n(AP) ↑"])
    ax.set(ylim=(0, 1), ylabel="Valeur", title="Comparaison des checkpoints sélectionnés sur le même test")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=len(models), fontsize=9)
    fig.subplots_adjust(bottom=0.25, top=0.77)
    export(fig, output, "metriques_principales", config, manifest, test=True,
           note="↑ : plus haut = meilleur · FPR ↓ : plus bas = meilleur · PR-AUC = average precision (AP).")

    fig, axes = plt.subplots(1, len(models), figsize=(4.4 * len(models) + 0.3, 5.1), squeeze=False)
    matrices = [np.asarray(manifest["metrics"][name]["confusion_matrix"]) for name in models]
    maximum = max(matrix.max() for matrix in matrices)
    for ax, name, matrix in zip(axes.ravel(), models, matrices):
        draw_confusion(ax, name, matrix, evaluations[name]["calibration"], maximum)
    fig.suptitle("Matrices de confusion — effectifs de contrats", fontsize=14)
    fig.subplots_adjust(bottom=0.25, top=0.79, wspace=0.55)
    confusion_note = f"{len(labels):,} contrats · VN/VP : vrais négatifs/positifs · FP/FN : faux positifs/négatifs.".replace(",", " ")
    export(fig, output, "matrices_confusion", config, manifest, test=True, note=confusion_note)
    for name, matrix in zip(models, matrices):
        fig, ax = plt.subplots(figsize=(6.5, 6.0))
        draw_confusion(ax, name, matrix, evaluations[name]["calibration"], maximum)
        fig.subplots_adjust(left=0.25, right=0.96, bottom=0.27, top=0.87)
        export(fig, output, "matrice_confusion_" + name, config, manifest, test=True, note=confusion_note)

    for kind in ("roc", "pr"):
        fig, ax = plt.subplots(figsize=(7.8, 6.4))
        for name in models:
            point = manifest["metrics"][name]
            if kind == "roc":
                fpr, recall, _ = roc_curve(labels, probabilities[name])
                ax.plot(fpr, recall, color=COLORS[name], linewidth=2, label=legend_name(name, evaluations[name]) + f" · AUC {point['roc_auc']:.3f}")
                marker_x, marker_y = point["false_positive_rate"], point["recall_vulnerable"]
            else:
                precision, recall, _ = precision_recall_curve(labels, probabilities[name])
                ax.step(recall, precision, where="post", color=COLORS[name], linewidth=2,
                        label=legend_name(name, evaluations[name]) + f" · AP {point['average_precision']:.3f}")
                marker_x, marker_y = point["recall_vulnerable"], point["precision_vulnerable"]
            ax.scatter([marker_x], [marker_y], color=COLORS[name], edgecolor="white", s=70, zorder=5)
        if kind == "roc":
            ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1, label="Classement aléatoire")
            ax.set(xlabel="Taux de faux positifs (FPR)", ylabel="Rappel positif (TPR)", title="Courbes ROC — test commun")
            name = "courbe_roc"
        else:
            prevalence = float(labels.mean())
            ax.axhline(prevalence, linestyle="--", color="#777777", linewidth=1, label=f"Prévalence positive : {prevalence:.3f}")
            ax.set(xlabel="Rappel positif", ylabel="Précision positive", title="Courbes précision–rappel — test commun")
            name = "courbe_precision_rappel"
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right" if kind == "roc" else "lower left", fontsize=8.3)
        fig.subplots_adjust(bottom=0.23, top=0.90, left=0.12, right=0.97)
        export(fig, output, name, config, manifest, test=True,
               note="Points : seuil principal ajusté sur calibration ; les courbes du test restent descriptives.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN, help="Dossier sous results/benchmark ; défaut : %(default)s")
    parser.add_argument("--config", type=Path, default=ROOT / "config/benchmark_models.json", help="Configuration du protocole")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--architecture-only", action="store_true", help="Exporter uniquement le schéma, sans lire les évaluations ni le test")
    mode.add_argument("--available-models", action="store_true", help="Exporter dans plots/modeles_evalues les modèles déjà évalués, même si le TCN est en pause")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        parser.error("--run-id doit être un seul nom de dossier.")
    result = ROOT / "results/benchmark" / args.run_id
    if not result.is_dir():
        parser.error("Le dossier du run n'existe pas.")
    sources = {}
    config = read_json(args.config.resolve(), sources)
    if tuple(config["models"]) != MODELS or config["threshold_fit_split"] != "calibration":
        raise ValueError("Le générateur requiert CNN–BiLSTM / XGBoost / TCN, avec calibration séparée.")
    protocol = read_json(result / "protocol.json", sources)
    if protocol["config"] != config:
        raise ValueError("La configuration diffère du protocole figé de ce run.")
    configure_style()
    output = result / "plots" / "modeles_evalues" if args.available_models else result / "plots"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "run_id": args.run_id,
                "dataset_id": config["dataset_id"], "protocol_id": protocol.get("protocol_id"),
                "test_role": config["test_role"], "exploratory": True,
                "selection": config["selection"], "threshold_fit_split": "calibration",
                "generator": {"path": relative(__file__), "sha256": file_hash(__file__)},
                "frozen_sources_sha256": protocol["sources_sha256"],
                "palette": COLORS, "environment": {"matplotlib": matplotlib.__version__, "numpy": np.__version__},
                "status": "architecture_only" if args.architecture_only else "incomplete",
                "test_labels_read": False, "plots": {}, "skipped": {}, "sources_sha256": sources}
    if args.available_models:
        summary_path = result / "summary.json"
        summary = read_json(summary_path, sources) if summary_path.exists() else None
        models = evaluated_models(summary)
        reasons = completion_gate(summary, result, models, require_complete=False)
        if reasons:
            raise ValueError("Impossible de tracer les modèles évalués : " + " ".join(reasons))
        manifest.update({"status": "evaluated_models_only", "compared_models": list(models),
                         "final_comparison": bool(summary.get("complete") and models == MODELS),
                         "excluded_models": {name: summary.get("models", {}).get(name, {}).get("status", "not_evaluated") for name in MODELS if name not in models}})
        labels, probabilities, evaluations = load_complete_test(config, result, summary, sources, manifest, models)
        manifest["test_labels_read"] = True
        plot_test(config, output, manifest, labels, probabilities, evaluations, models)
        for name in EXTRA_PLOTS:
            manifest["skipped"][name] = "Mode modèles évalués uniquement ; graphiques finaux à trois modèles exclus."
    elif args.architecture_only:
        plot_architecture(config, output, manifest)
        for name in (*TEST_PLOTS, *EXTRA_PLOTS, "logloss_validation_tcn"):
            manifest["skipped"][name] = "Option --architecture-only."
    else:
        plot_architecture(config, output, manifest)
        plot_validation(config, result, output, sources, manifest)
        summary_path = result / "summary.json"
        summary = read_json(summary_path, sources) if summary_path.exists() else None
        reasons = completion_gate(summary, result)
        if reasons:
            manifest["incomplete_reasons"] = reasons
            for name in TEST_PLOTS:
                manifest["skipped"][name] = "Comparaison incomplète ; aucun graphique de test généré."
            for name in EXTRA_PLOTS:
                manifest["skipped"][name] = "Attente des trois évaluations complètes et de la sélection finale."
        else:
            labels, probabilities, evaluations = load_complete_test(config, result, summary, sources, manifest)
            manifest["test_labels_read"] = True
            plot_test(config, output, manifest, labels, probabilities, evaluations)
            plot_paired_differences(config, result, output, sources, manifest, summary)
            plot_selected_loss(config, result, output, sources, manifest, evaluations)
            manifest["status"] = "complete"
    save_json(output / "plot_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "plots": list(manifest["plots"]),
                      "skipped": list(manifest["skipped"]), "test_labels_read": manifest["test_labels_read"],
                      "manifest": relative(output / "plot_manifest.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
