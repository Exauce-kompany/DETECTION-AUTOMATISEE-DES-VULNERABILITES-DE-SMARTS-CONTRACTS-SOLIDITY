# Comparaison provisoire sur validation

Mise à jour UTC : 2026-09-23T07:06:48.633848+00:00. Exécution `comparison-v1-20260922`.

Ce tableau utilise uniquement les checkpoints terminés et les 1 709 contrats de validation. CodeT5 et l'évaluation finale peuvent encore être en cours. Il ne constitue pas le classement final et ne présente aucun résultat du test.

F1 macro au seuil brut 0,5 ; rappel calculé sur la courbe ROC de validation à FPR ≤ 10 %. Les deux colonnes n'utilisent donc pas le même seuil. Moyenne ± écart-type entre graines pour les réseaux ; TF-IDF a un seul ajustement déterministe.

| Variante | Entraînements terminés | F1 macro validation | AUC validation | Rappel validation à FPR ≤ 10 % |
|---|---:|---:|---:|---:|
| bilstm_first512 | 3 | 75.94 % ± 0.37 | 0.8399 | 61.68 % |
| bilstm_full | 3 | 80.54 % ± 0.73 | 0.8969 | 68.69 % |
| cnn_bilstm_first512 | 3 | 75.48 % ± 0.91 | 0.8370 | 60.85 % |
| cnn_bilstm_full | 3 | 81.73 % ± 2.00 | 0.9063 | 72.09 % |
| cnn_first512 | 3 | 75.85 % ± 1.32 | 0.8349 | 60.13 % |
| cnn_full | 3 | 80.33 % ± 1.48 | 0.9044 | 71.89 % |
| tfidf_first512 | 1 | 70.27 % ± 0.00 | 0.8328 | 60.57 % |
| tfidf_full | 1 | 78.98 % ± 0.00 | 0.8961 | 73.63 % |

Le choix final sera figé après l'entraînement de tous les candidats, sur validation. Température et seuils seront ensuite ajustés exclusivement sur calibration. Les résultats du test V3 déjà consulté seront présentés comme exploratoires, avec un diagnostic commun excluant les collisions de préfixes induites par la troncature.

`full` conserve tous les tokens ; `first512` conserve les 512 premiers tokens lexicaux normalisés, avant encodage. Les résultats de validation reflètent les labels historiques et ne prouvent pas une généralisation à de nouveaux projets.
