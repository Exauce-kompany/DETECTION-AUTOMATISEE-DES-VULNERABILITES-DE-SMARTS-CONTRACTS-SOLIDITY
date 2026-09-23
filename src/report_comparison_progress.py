"""Report finished validation runs without opening test predictions or selecting a winner."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_curve
from .comparison_data import save_json
from .preprocessing_v3 import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="comparison-v1-20260922")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        raise ValueError("Invalid run ID")
    result = ROOT / "results/comparison" / args.run_id
    protocol = json.loads((result / "protocol.json").read_text())
    config = protocol["config"]
    with np.load(ROOT / config["dataset_dir"] / "validation.npz") as data:
        labels = data["labels"]
    records = []
    for directory in sorted(result.iterdir()):
        if not directory.is_dir():
            continue
        paths = sorted(directory.glob("seed-*/run.json")) or list(directory.glob("run.json"))
        if not paths:
            continue
        runs = [json.loads(path.read_text()) for path in paths]
        recalls = []
        for path in paths:
            fpr, tpr, _ = roc_curve(labels, np.load(path.parent / "validation_logits.npy"))
            recalls.append(float(tpr[fpr <= config["target_fpr"]].max()))
        f1 = [run["validation"]["f1_macro"] for run in runs]
        records.append({"variant": directory.name, "completed_runs": len(runs), "f1_macro_mean": float(np.mean(f1)), "f1_macro_std": float(np.std(f1, ddof=1)) if len(f1) > 1 else 0.0, "roc_auc_mean": float(np.mean([run["validation"]["roc_auc"] for run in runs])), "recall_at_fpr_0_10_mean": float(np.mean(recalls))})
    state = {"generated_utc": datetime.now(timezone.utc).isoformat(), "partition": "validation_only", "final_selection": False, "rows": records}
    save_json(result / "validation_progress.json", state)
    lines = ["# Comparaison provisoire sur validation", "", f"Mise à jour UTC : {state['generated_utc']}. Exécution `{args.run_id}`.", "", "Ce tableau utilise uniquement les checkpoints terminés et les 1 709 contrats de validation. CodeT5 et l'évaluation finale peuvent encore être en cours. Il ne constitue pas le classement final et ne présente aucun résultat du test.", "", "F1 macro au seuil brut 0,5 ; rappel calculé sur la courbe ROC de validation à FPR ≤ 10 %. Les deux colonnes n'utilisent donc pas le même seuil. Moyenne ± écart-type entre graines pour les réseaux ; TF-IDF a un seul ajustement déterministe.", "", "| Variante | Entraînements terminés | F1 macro validation | AUC validation | Rappel validation à FPR ≤ 10 % |", "|---|---:|---:|---:|---:|"]
    for row in records:
        lines.append(f"| {row['variant']} | {row['completed_runs']} | {100*row['f1_macro_mean']:.2f} % ± {100*row['f1_macro_std']:.2f} | {row['roc_auc_mean']:.4f} | {100*row['recall_at_fpr_0_10_mean']:.2f} % |")
    lines += ["", "Le choix final sera figé après l'entraînement de tous les candidats, sur validation. Température et seuils seront ensuite ajustés exclusivement sur calibration. Les résultats du test V3 déjà consulté seront présentés comme exploratoires, avec un diagnostic commun excluant les collisions de préfixes induites par la troncature.", "", "`full` conserve tous les tokens ; `first512` conserve les 512 premiers tokens lexicaux normalisés, avant encodage. Les résultats de validation reflètent les labels historiques et ne prouvent pas une généralisation à de nouveaux projets.", ""]
    (result / "validation_progress.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(result / "validation_progress.md")


if __name__ == "__main__":
    main()
