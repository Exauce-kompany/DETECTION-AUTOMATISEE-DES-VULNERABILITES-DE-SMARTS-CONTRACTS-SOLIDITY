# Comparaison principale — état du 5 octobre 2026

**Archive :** l'utilisateur a ensuite choisi le TCN à la place de CodeBERT. La [comparaison principale actuelle](comparaison_cnn_xgboost_tcn.md) conserve les résultats CNN et XGBoost avec leur provenance.

La présentation du mémoire a été recentrée sur **CNN–BiLSTM, XGBoost et CodeBERT**. La revue de littérature conserve les références scientifiques et ne classe pas notre modèle d'après des scores publiés sur d'autres données. L'ancienne comparaison CNN/BiLSTM/CodeT5/TF-IDF reste une archive expérimentale.

## Résultats obtenus

Les deux checkpoints évalués utilisent les mêmes 1 709 exemples de test et des seuils appris exclusivement sur calibration (cible de rappel de 90 %). Ces scores concernent des checkpoints choisis sur validation, pas une moyenne mélangée au modèle actif.

| Modèle | F1 macro | Précision positive | Rappel positif | Faux positifs | PR-AUC |
|---|---:|---:|---:|---:|---:|
| CNN–BiLSTM actif, graine 73 | 82,10 % | 77,94 % | 88,95 % | 24,45 % | 0,8967 |
| TF-IDF + XGBoost, graine 101 | 89,41 % | 89,11 % | 89,43 % | 10,61 % | 0,9534 |
| CodeBERT avec fine-tuning complet | Non évalué | Non évalué | Non évalué | Non évalué | Non évalué |

XGBoost a été entraîné sur les trois graines 42, 73, 101 ; la graine 101 est retenue par log-loss de validation. Meilleure itération 597 (index commençant à zéro), soit 598 arbres utilisés. Sa confusion test donne 775 vrais négatifs, 92 faux positifs, 89 faux négatifs et 753 vrais positifs. Le CNN donne 655, 212, 93 et 749 respectivement.

L'écart de F1 macro XGBoost−CNN vaut **+7,31 points**. L'intervalle bootstrap exploratoire à 95 %, par groupes de contrats et sur 1 000 réplications, va de **+4,94 à +10,80 points**. Il conditionne sur les deux checkpoints sélectionnés et n'inclut pas toute la variabilité d'entraînement. L'écart de rappel (+0,48 point) a un intervalle comprenant zéro.

Le test a déjà été consulté pendant le développement. Ces constats sont exploratoires ; ils ne prouvent pas une supériorité générale de XGBoost sur le deep learning. Les annotations historiques, souvent issues d'analyseurs, peuvent favoriser des régularités lexicales. Un corpus indépendant avec annotations vérifiées reste nécessaire.

## CodeBERT : étape restante

L'encodeur et la tête sont entièrement entraînables : **124 055 809 paramètres**. La tokenisation conserve 140 510 fenêtres d'entraînement et 20 723 fenêtres de validation, au lieu de tronquer les contrats. Une perte BCE est calculée au contrat après moyenne des logits des fenêtres ; les fenêtres d'un contrat positif ne reçoivent pas toutes un label positif.

Le pilote a mesuré deux contrats près des quantiles 50 % et 90 % des longueurs. L'extrapolation linéaire donne **environ 5,2 jours par graine et 15,5 jours pour les trois graines sur CPU**, pour trois époques. Elle exclut les sauvegardes, le test et les pauses, et peut être raccourcie par l'arrêt anticipé. L'entraînement complet n'a pas été lancé ; aucun score CodeBERT n'est inventé. À cette étape, plusieurs remplacements étaient envisagés. L'utilisateur a ensuite retenu le **TCN**, décrit dans le [protocole actuel](protocole_cnn_xgboost_tcn.md).

Les sauvegardes de reprise et le moteur d'entraînement sont prêts. Les tests vérifient notamment l'équivalence du gradient en deux passes, la contribution des tokens après le préfixe de 512, et la reprise identique sur une expérience de contrôle.

## Traçabilité

Le [protocole](protocole_cnn_xgboost_codebert.md) précise hyperparamètres, partitions, règles de sélection et commandes. Les valeurs viennent du [rapport calculé](../results/benchmark/cnn-xgboost-codebert-20261005-final/report.md), des évaluations JSON et de `paired_cnn_xgboost.json`. Les poids de l'application et le dataset sont conservés.

Le premier calcul XGBoost est conservé dans `results/benchmark/cnn-xgboost-codebert-20261005/`, avec snapshot du code. Ses résultats ont été importés dans le run final après corrections des descriptions de timing, de la provenance et des écritures JSON. `imported_xgboost.json` vérifie que poids, logits et métriques n'ont pas changé. Il n'y a qu'une copie des poids XGBoost, référencée par les manifestes.

Les temps par lot ne comparent pas la vitesse : XGBoost inclut le rechargement de ses artefacts dans le timer, CNN et CodeBERT chargent leurs poids avant.

## Références

- Feng et al. (2020), [CodeBERT: A Pre-Trained Model for Programming and Natural Languages](https://aclanthology.org/2020.findings-emnlp.139/).
- Chen et Guestrin (2016), [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754).
