# CNN–BiLSTM, XGBoost et CodeBERT

**État : comparaison incomplète ; aucun classement final.**

Les résultats proviennent des mêmes partitions SMART BUG. Le test a déjà été consulté ; cette étude reste exploratoire. Aucun score publié d'un autre corpus n'entre dans ce tableau.

Un checkpoint par famille est sélectionné sur validation. Le CNN est le checkpoint TensorFlow actif, pas le réseau PyTorch de l'ancienne étude. Les nouveaux modèles ont trois graines ; les métriques principales concernent leur checkpoint sélectionné, pas une moyenne mélangée au CNN actif.

| Modèle | État | F1 macro | Précision positive | Rappel positif | FPR | PR-AUC |
|---|---|---:|---:|---:|---:|---:|
| CNN–BiLSTM actif | Évalué | 0.8210 | 0.7794 | 0.8895 | 0.2445 | 0.8967 |
| TF-IDF + XGBoost | Évalué | 0.8941 | 0.8911 | 0.8943 | 0.1061 | 0.9534 |
| CodeBERT fine-tuné | Pilote terminé ; entraînement complet à lancer | — | — | — | — | — |

Les températures et les seuils sont ajustés exclusivement sur calibration, avec une cible de rappel de 90 %. La cible ne garantit pas 90 % sur le test. Le seuil alternatif visant FPR ≤ 10 % sur calibration figure dans les JSON d'évaluation.

XGBoost utilise TF-IDF appris uniquement sur train (1–2 grammes, 20 000 caractéristiques max.). CodeBERT doit adapter l'encodeur et sa tête sur tous les segments du contrat ; sa perte porte sur le contrat entier. Aucune fenêtre positive n'est étiquetée automatiquement comme vulnérable.

## Variabilité des nouveaux modèles

Les scores ci-dessous sont ceux de validation avant calibration ; ils documentent la sélection, sans utiliser le test pour choisir une graine.

| Modèle | Graine | Log-loss validation | F1 macro validation à 0,5 |
|---|---:|---:|---:|
| TF-IDF + XGBoost | 42 | 0.3244 | 0.8665 |
| TF-IDF + XGBoost | 73 | 0.3271 | 0.8712 |
| TF-IDF + XGBoost | 101 | 0.3197 | 0.8682 |

## Limites

Les cibles sont des annotations historiques, en partie issues d'analyseurs statiques. La classe 0 ne certifie pas l'absence de vulnérabilité. La source réservée contient seulement six négatifs : son taux de faux positifs est trop fragile pour une conclusion générale.

Le CNN et CodeBERT conservent l'ordre dans leurs représentations ; TF-IDF le réduit aux n-grammes. Les identifiants sont normalisés pour les trois modèles, ce qui peut défavoriser CodeBERT préentraîné sur d'autres langages. Les budgets d'entraînement et les tailles de modèles diffèrent et sont archivés explicitement.

Les durées d'inférence par lot ne sont pas des latences unitaires : XGBoost inclut le rechargement de ses artefacts, CNN et CodeBERT chargent leurs poids avant le chronométrage. Ces chiffres ne servent pas à comparer la vitesse des modèles. Les expériences historiques restent dans results/comparison mais ne sont plus la comparaison principale du mémoire.

## Références

- Feng et al. (2020), [CodeBERT](https://aclanthology.org/2020.findings-emnlp.139/).
- Chen et Guestrin (2016), [XGBoost](https://arxiv.org/abs/1603.02754).
