# Comparaison : CNN–BiLSTM, XGBoost et CodeBERT

**Archive :** ce protocole décrit l'option initiale et son pilote. Le choix du TCN est décrit dans le [protocole actuel](protocole_cnn_xgboost_tcn.md). Les commandes CodeBERT ci-dessous correspondent à l'ancien code et à la configuration figés dans les snapshots historiques.

La comparaison principale du mémoire porte sur trois modèles exécutés sur SMART BUG. La revue de littérature présente leurs fondements ; aucun score publié sur un autre corpus ne sert à classer notre modèle. Les anciennes expériences restent archivées.

| Modèle | Apprentissage | Entrée et décision |
|---|---|---|
| CNN–BiLSTM actif | Depuis zéro ; checkpoint TensorFlow graine 73 déjà sélectionné sur validation | Code complet ; CNN par fenêtre puis BiLSTM hiérarchique |
| TF-IDF + XGBoost | Arbres appris depuis zéro | Tous les tokens normalisés ; unigrammes/bigrammes ; score au contrat |
| CodeBERT-base | Encodeur Microsoft préentraîné, puis adaptation de tous ses paramètres et de sa tête | Toutes les fenêtres BPE ; moyenne des logits CLS ; perte au contrat |

Le CNN est celui de l'application (F1 macro archivé : 82,10 %). Le CNN–BiLSTM PyTorch de l'ancienne étude est une autre implémentation ; sa moyenne de 84,54 % n'entre pas dans ce tableau. Les nouveaux modèles ont trois graines (42, 73, 101), avec choix par log-loss minimale sur validation. Le tableau principal compare des checkpoints sélectionnés ; il ne mélange pas un score individuel avec une moyenne de graines.

## Données et prévention des fuites

Snapshot : `dataset/benchmark` ; identifiant `69395a7f7f7d66a9066687445aca249e9d67a994a12b1466920fb4e64102b465`.

| Partition | Exemples | Usage |
|---|---:|---|
| Train | 11 959 | Paramètres et vocabulaire TF-IDF |
| Validation | 1 709 | Arrêt anticipé et sélection |
| Calibration | 1 709 | Température et seuils |
| Test | 1 709 | Évaluation après sélection |
| Source réservée | 1 152 | Diagnostic entre sources |

Les groupes sont disjoints selon l'audit existant. Les labels et tokens normalisés sont communs ; chaque modèle a sa représentation déclarée. CodeBERT conserve tous les segments, sans troncature aux 512 premiers sous-tokens. Un contrat positif ne rend pas chacune de ses fenêtres positive.

Le train contient 17 450 564 tokens lexicaux. Les pondérations communes équilibrent les strates source/label à partir du train uniquement (`src.experiment.training_weights`). Validation et évaluation sont non pondérées. Aucun diagnostic d'analyseur statique n'est ajouté aux entrées XGBoost.

## Réglages fixés

Valeurs exactes : `config/benchmark_models.json`.

- **CNN–BiLSTM** : poids conservés ; embedding 32, 24 filtres de largeur 5, 24 unités BiLSTM par sens, dropout 0,30 ; Adam à 0,001, maximum historique de 12 époques, patience 3.
- **XGBoost** : TF-IDF 1–2 grammes, au plus 20 000 caractéristiques, `min_df=2`, fréquence sous-linéaire ; au plus 600 arbres, profondeur 6, taux 0,05, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=2`, `reg_lambda=1`, méthode `hist`, objectif `binary:logistic`. Arrêt après 40 itérations sans amélioration de la log-loss validation ; inférence jusqu'à la meilleure itération.
- **CodeBERT** : `microsoft/codebert-base`, révision `3b0952feddeffad0063f274080e3c23d75e7eb39` ; encodeur entier et tête linéaire entraînables ; AdamW à 0,00002, weight decay 0,01, au plus 3 époques, patience 2. Fenêtres de 256 sous-tokens incluant les marqueurs spéciaux, micro-lots de 4 fenêtres, accumulation sur 8 contrats. Moyenne des logits CLS, puis BCE globale du contrat. Deux passes calculent le gradient exact en conservant un micro-lot en mémoire ; dropout désactivé pour leur déterminisme. Checkpoints de reprise avec optimiseur et RNG.

Solidity ne fait pas partie des six langages du préentraînement CodeBERT ; son transfert est évalué localement. Les budgets et tailles de modèles diffèrent : ce protocole compare des solutions déclarées et n'isole pas l'architecture à budget identique.

## Calibration et indicateurs

Après sélection, la température est ajustée exclusivement sur calibration. Le seuil principal maximise la précision avec une cible de rappel de 90 % sur calibration. Un seuil alternatif vise FPR ≤ 10 % sur calibration. Ces objectifs peuvent ne pas être atteints sur le test.

Les JSON contiennent exactitude, F1 macro, précision/rappel positifs, confusion, FPR, ROC-AUC, PR-AUC (average precision), Brier, ECE et résultats par source/granularité/confiance d'annotation. Les métriques portent sur les contrats, dans l'ordre des identifiants enregistrés.

Les temps d'inférence par lot ne sont pas des latences unitaires. XGBoost inclut le chargement de ses artefacts dans le timer, CNN et CodeBERT le font avant ; ces durées ne servent pas à comparer la vitesse. Les durées murales peuvent inclure une suspension du PC. Le pilote CodeBERT estime le coût, sans produire de score final.

## Fichiers et exécution

| Fichier | Fonction |
|---|---|
| `config/benchmark_models.json` | Dataset, modèles, graines, hyperparamètres et sélection |
| `src/run_model_benchmark.py` | Vérification ; XGBoost ; sélection, calibration, test et rapport |
| `src/benchmark_codebert.py` | Tokenisation complète ; fine-tuning au contrat ; pilote et reprise |
| `requirements-benchmark.txt` | Dépendances de l'environnement expérimental |
| `results/benchmark/<run>/protocol.json` | Configuration figée, empreintes et environnement |
| `results/benchmark/<run>/<modèle>/` | Historiques, sélection, calibration, logits, identifiants et métriques |
| `models/benchmark/<run>/` | Poids XGBoost/CodeBERT et vocabulaire TF-IDF |

Dans l'environnement scientifique existant, installer `requirements-benchmark.txt`, puis :

```powershell
python -m src.run_model_benchmark --stage cnn
python -m src.run_model_benchmark --stage xgboost
python -m src.run_model_benchmark --stage codebert-pilot
python -m src.run_model_benchmark --stage codebert
python -m src.run_model_benchmark --stage report
```

Le pilote est distinct. `--stage codebert` entraîne les trois graines et évalue après sélection ; il reprend les checkpoints disponibles. Changer la configuration ou le code figé demande un nouveau `--run-id`. Aucun modèle n'est activé dans l'application.

## Statut et limites

Le [rapport calculé](../results/benchmark/cnn-xgboost-codebert-20261005-final/report.md) est la source des scores. Les modèles sans évaluation restent « non évalués ». Une conclusion finale exige les trois évaluations terminées.

Le test a déjà été observé : comparaison **exploratoire**, à confirmer sur un corpus indépendant. Les annotations sont en partie faibles ; la classe 0 ne certifie pas la sécurité. La source réservée ne contient que six négatifs : son FPR est fragile. La normalisation des identifiants peut défavoriser CodeBERT.

## Références

- Feng et al. (2020). *CodeBERT: A Pre-Trained Model for Programming and Natural Languages*. [Article original](https://aclanthology.org/2020.findings-emnlp.139/).
- Chen et Guestrin (2016). *XGBoost: A Scalable Tree Boosting System*. [Article original](https://arxiv.org/abs/1603.02754).
- Microsoft. [Fiche et poids de CodeBERT-base](https://huggingface.co/microsoft/codebert-base).
- XGBoost. [Documentation officielle](https://xgboost.readthedocs.io/en/stable/python/python_api.html).
