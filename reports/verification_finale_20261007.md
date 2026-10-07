# Vérification du travail — 7 octobre 2026

La comparaison CNN–BiLSTM / TF-IDF + XGBoost / TCN est terminée. Aucun défaut bloquant n'a été détecté dans les contrôles effectués. Cette conclusion porte sur les fichiers, calculs et régressions vérifiés ; elle ne constitue pas une certification de sécurité ni une preuve de généralisation à tous les contrats Solidity.

## Fin des entraînements

| Modèle | Graines 42 / 73 / 101 | Version sélectionnée | F1 macro sur test |
|---|---|---|---:|
| CNN–BiLSTM | 6 / 6 / 8 époques, terminées | Graine 73, époque 3 | 82,10 % |
| TF-IDF + XGBoost | 600 / 600 / 600 itérations, terminées | Graine 101, meilleure itération 597 en indexation à partir de zéro, soit 598 arbres utilisés | 89,41 % |
| TCN | 10 / 5 / 10 époques, terminées | Graine 42, époque 7 | 84,77 % |

Le CNN et XGBoost proviennent d'entraînements antérieurs, réutilisés avec leurs empreintes. Le TCN a terminé sa dernière graine après la reprise du 6 octobre. Les scores principaux concernent un checkpoint par famille, pas une moyenne des trois graines sur test.

L'arrêt avant 12 époques est prévu par la patience de trois époques sans amélioration. Les meilleures époques et les longueurs des historiques concordent. XGBoost a atteint son budget de 600 itérations pour chaque graine ; ses prédictions utilisent automatiquement la meilleure itération avec l'API `XGBClassifier`.

## Contrôles réalisés

- **194 vérifications de preuves réussies** : configuration figée, sept instantanés du code d'origine, import des baselines, choix sur validation, intégrité des poids, états TCN, séparation de la calibration, identifiants et métriques recalculées depuis les logits enregistrés. Les petites différences de log-loss XGBoost, d'environ 10⁻⁸ entre le calcul natif et NumPy, relèvent de l'arrondi numérique.
- **92 tests Python réussis** en 23,086 secondes : données, modèles, checkpoints, calibration, inférence, application HTTP et persistance. Une première tentative restreinte rencontrait des refus d'accès Windows aux dossiers temporaires ; l'exécution complète hors de cette restriction a réussi.
- **45 assertions de l'interface réussies** dans V8 avec un DOM simulé ; aucun contrôle visuel dans un navigateur n'est revendiqué.
- **Ruff 0.16.10** : contrôles de code et de formatage réussis, 65 fichiers déjà formatés. Le prétraitement figé reste exclu conformément à la configuration.
- **18 238 contrats audités** : empreintes, labels et encodages cohérents ; aucun recouvrement entre partitions selon les critères de l'auditeur. Cela ne démontre pas l'absence de tous les clones approximatifs.
- Les trois archives Keras sont lisibles et correspondent aux empreintes Git LFS enregistrées. Les trois modèles XGBoost se chargent, contiennent 600 arbres et conservent leurs meilleures itérations. Les poids TCN concordent avec leurs métadonnées et meilleurs états sauvegardés.
- Les métriques et matrices des trois modèles ont été recalculées sur les **1 709 contrats de test**, ainsi que sur la source réservée, pour les seuils principal et alternatif. Les températures et seuils conservés sont issus de calibration.
- Les trois différences appariées correspondent aux métriques et à l'empreinte des évaluations. Les intervalles existants sont vérifiés comme fichiers de provenance ; aucun nouveau bootstrap n'a été lancé.
- Les **11 figures, soit 22 exports PNG/SVG**, et leurs **28 fichiers d'entrée** conservent les empreintes annoncées.
- Le PDF compte **44 pages** et contient les trois scores corrects. Word, PDF, résumé et notes d'exécution correspondent aux empreintes du QA documentaire. Les remerciements bénéficient du contrôle antérieur en texte et en pixels, attaché à ces mêmes livrables ; aucun nouveau rendu n'a été nécessaire. Les scripts documentaires actuels sont syntaxiquement valides.

## Points à conserver dans les conclusions

1. Le test a déjà été consulté : la comparaison reste **exploratoire**, à confirmer sur un corpus indépendant.
2. Les labels sont en partie issus d'analyseurs statiques. Une classe négative ne garantit pas la sécurité d'un contrat.
3. La source réservée ne contient que **six négatifs** ; son taux de faux positifs est fragile.
4. Les budgets, représentations et historiques de sélection diffèrent entre modèles. Les temps enregistrés ne permettent pas un classement équitable des vitesses.
5. Le TCN traite toutes les positions encodées, mais son champ réceptif local est limité à 2 045 positions ; le pooling global ne modélise pas toutes les relations éloignées.

XGBoost obtient le meilleur F1 macro sur ce test. Le TCN améliore le CNN–BiLSTM de 2,67 points. Ces résultats ne prouvent pas qu'une architecture serait universellement supérieure.

## Réorganisation actuelle du code

Le dossier de travail contient une réorganisation non enregistrée dans Git. Les anciens modules ont été déplacés dans `src/data`, `src/training`, `src/evaluation`, `src/inference`, `src/experiments` et `src/visualization`. Les nouveaux imports passent les tests. Les données, configurations, poids, résultats scientifiques et le prétraitement figé n'ont pas de modification signalée par Git.

Six sources de l'ancien protocole ont changé ou été réparties dans les nouveaux modules ; leurs instantanés d'origine sont intègres. Ce décalage est attendu et ne falsifie pas les résultats historiques. Une nouvelle campagne exige un **nouveau `--run-id`** ; il ne faut pas modifier les empreintes anciennes pour contourner ce contrôle. Le générateur de figures a également changé de chemin : les exports historiques restent valides, tandis qu'une régénération devra produire son propre manifeste avec le générateur actuel.

Cette revue n'a pas relancé d'entraînement, modifié de poids, réajusté de seuil, remplacé de figure ni changé le modèle actif. Elle n'a pas enregistré la réorganisation existante dans GitHub.

## Fichiers de preuve

- [Audit détaillé avec les 194 contrôles](verification_finale_20261007.json).
- [Résumé de la comparaison](../results/benchmark/cnn-xgboost-tcn-20261005/summary.json).
- [Rapport final de comparaison](rapport_final_partie_2.md).
- [Contrôle documentaire](../docs/memoire/qa.json).
- Journal local des tests : `.tmp-tests/audit-20261007-tests-full.log`.

L'auditeur local utilisé pour cette revue est `.cache-comparison/audit_final_20261007.py`. Ces deux chemins de travail sont ignorés par Git.
