# Archive des graphiques — partie 1

**CNN–BiLSTM actif et TF-IDF + XGBoost**, sur les mêmes **1 709 contrats de test**. Ces graphiques ont été sauvegardés pendant la pause du TCN, dans le commit `72034bd`, et conservent cet état historique à deux modèles. Le TCN est maintenant terminé : les [matrices finales des trois modèles](../matrices_confusion.png) et les autres figures se trouvent dans le dossier parent. Cette archive conserve les checkpoints sélectionnés sur validation, ainsi que les températures et seuils ajustés sur calibration.

![Matrices de confusion CNN–BiLSTM et XGBoost](matrices_confusion.png)

Les lignes correspondent aux annotations réelles et les colonnes aux classes prédites. La classe 0 signifie « sans vulnérabilité annotée » ; elle ne certifie pas la sécurité. VN/VP : vrais négatifs/positifs ; FP/FN : faux positifs/négatifs.

| Graphique | PNG | SVG |
|---|---|---|
| Matrices comparées | [Ouvrir](matrices_confusion.png) | [Ouvrir](matrices_confusion.svg) |
| Matrice CNN–BiLSTM | [Ouvrir](matrice_confusion_cnn_bilstm.png) | [Ouvrir](matrice_confusion_cnn_bilstm.svg) |
| Matrice XGBoost | [Ouvrir](matrice_confusion_xgboost.png) | [Ouvrir](matrice_confusion_xgboost.svg) |
| F1, précision, rappel, FPR et PR-AUC | [Ouvrir](metriques_principales.png) | [Ouvrir](metriques_principales.svg) |
| Courbes ROC | [Ouvrir](courbe_roc.png) | [Ouvrir](courbe_roc.svg) |
| Courbes précision–rappel | [Ouvrir](courbe_precision_rappel.png) | [Ouvrir](courbe_precision_rappel.svg) |

PNG à 300 dpi ; SVG vectoriels. Les sources, empreintes, matrices et seuils exacts figurent dans [plot_manifest.json](plot_manifest.json). Les métriques ont été recalculées à partir des logits archivés et comparées aux évaluations enregistrées avant l'export. Les six PNG ont été inspectés visuellement.

La commande utilisée pour cet export partiel était :

```powershell
python -m src.plot_model_benchmark --available-models
```

Depuis la fin du TCN, cette option inclurait les trois modèles évalués et remplacerait les graphiques de ce sous-dossier. Pour générer les figures finales dans le dossier parent, utiliser `python -m src.plot_model_benchmark`. Le manifeste de cette archive référence les sources de partie 1 conservées dans l'historique Git ; les résultats actuels ont leur propre manifeste dans le dossier parent. Le test a déjà été consulté : les comparaisons restent exploratoires.
