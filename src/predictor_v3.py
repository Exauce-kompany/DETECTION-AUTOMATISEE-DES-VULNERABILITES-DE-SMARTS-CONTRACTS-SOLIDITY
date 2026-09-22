"""Single production inference implementation for the web application and CLI."""
import json
from pathlib import Path
from threading import RLock

import numpy as np
from scipy.special import expit

from .preprocessing_v3 import ROOT, VERSION, file_digest, prepare_code


class SmartContractPredictor:
    def __init__(self, manifest_path=None):
        self.manifest_path = Path(manifest_path) if manifest_path else ROOT / "models/active_model.json"
        self.model = None
        self.bundle = None
        self.vocabulary = None
        self._lock = RLock()

    def _artifact(self, path):
        resolved = (ROOT / path).resolve()
        if not resolved.is_relative_to(ROOT):
            raise ValueError("Artifact path escapes project root.")
        return resolved

    def read_bundle(self):
        if not self.manifest_path.is_file():
            raise RuntimeError("Le modèle V3 n'est pas encore prêt. Exécuter python -m src.train_v3.")
        bundle = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if bundle.get("preprocessing_version") != VERSION:
            raise ValueError("La version du prétraitement ne correspond pas au modèle.")
        calibration = bundle.get("calibration", {})
        if calibration.get("fit_split") != "calibration" or not np.isfinite(calibration.get("temperature", np.nan)) or calibration["temperature"] <= 0:
            raise ValueError("Le modèle ne possède pas de calibration valide.")
        if not 0 <= calibration.get("threshold", -1) <= 1:
            raise ValueError("Seuil de décision invalide.")
        return bundle

    def load_resources(self):
        with self._lock:
            if self.model is not None:
                return
            bundle = self.read_bundle()
            model_path = self._artifact(bundle["model_path"])
            vocabulary_path = self._artifact(bundle["vocabulary_path"])
            if file_digest(model_path) != bundle["model_sha256"] or file_digest(vocabulary_path) != bundle["vocabulary_sha256"]:
                raise ValueError("Le modèle ou son vocabulaire a changé depuis l'évaluation.")
            import tensorflow as tf
            from . import model_v3  # Register custom layers before deserialization.
            model = tf.keras.models.load_model(model_path, compile=False)
            vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
            if model.input_shape[-1] != bundle["config"]["window_size"] or model.output_shape[-1] != 1:
                raise ValueError("Architecture incompatible avec la configuration du modèle.")
            self.bundle, self.vocabulary, self.model = bundle, vocabulary, model

    def predict(self, code):
        self.load_resources()
        windows, preprocessing = prepare_code(code, self.vocabulary, self.bundle["config"]["window_size"])
        with self._lock:
            logit = float(np.asarray(self.model(windows[None, ...], training=False)).reshape(-1)[0])
        if not np.isfinite(logit):
            raise ValueError("Le modèle a produit un score non fini.")
        calibration = self.bundle["calibration"]
        probability = float(expit(logit / calibration["temperature"]))
        label = int(probability >= calibration["threshold"])
        confidence = probability if label else 1.0 - probability
        return {
            "success": True, "predicted_label": label,
            "predicted_class": "vulnerable" if label else "non_vulnerable",
            "verdict": "Contrat potentiellement vulnérable" if label else "Aucune vulnérabilité signalée dans le périmètre appris",
            "message": "La décision porte sur les annotations et familles de code du jeu d'apprentissage ; elle ne certifie pas la sécurité du contrat.",
            "confidence": confidence, "confidence_percent": round(confidence * 100, 2),
            "confidence_level": "Score calibré sur le jeu dédié",
            "probability_non_vulnerable": 1.0 - probability,
            "probability_non_vulnerable_percent": round((1.0 - probability) * 100, 2),
            "probability_vulnerable": probability,
            "probability_vulnerable_percent": round(probability * 100, 2),
            "decision_threshold": calibration["threshold"],
            "calibration": {"method": calibration["method"], "fit_split": calibration["fit_split"], "temperature": calibration["temperature"], "target_recall": calibration["target_recall"]},
            "preprocessing": preprocessing,
            "model_info": {"version": "V3", "architecture": "CNN par fenêtre + BiLSTM hiérarchique", "sequence_length": None, "window_size": self.bundle["config"]["window_size"], "vocabulary_size": len(self.vocabulary), "run_id": self.bundle["run_id"], "dataset_id": self.bundle["dataset_id"]},
            "warning": "Le code complet est traité. Le résultat reste une aide à l'audit et une absence de signal ne constitue pas une preuve de sécurité.",
        }

    def model_report(self):
        bundle = self.bundle or self.read_bundle()
        result_dir = self._artifact(bundle["results_path"])
        return {"run_id": bundle["run_id"], "dataset_id": bundle["dataset_id"], "architecture": "CNN par fenêtre + BiLSTM hiérarchique", "version": "V3", "config": bundle["config"], "selected": bundle["selected"], "calibration": bundle["calibration"], "evaluation": json.loads((result_dir / "evaluation.json").read_text(encoding="utf-8")), "experiment": json.loads((result_dir / "experiment.json").read_text(encoding="utf-8")), "seeds": json.loads((result_dir / "seed_results.json").read_text(encoding="utf-8"))}

    def dataset_report(self):
        bundle = self.bundle or self.read_bundle()
        return {"dataset_id": bundle["dataset_id"], "config": bundle["config"], **json.loads((self._artifact(bundle["dataset_path"]) / "audit.json").read_text(encoding="utf-8"))}
