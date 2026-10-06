# Rapport final — partie 2

Édition finale du **6 octobre 2026**, après reprise et fin de la campagne `cnn-xgboost-tcn-20261005`. La comparaison **CNN–BiLSTM actif / TF-IDF + XGBoost / TCN depuis zéro** est complète : `summary.complete=true` et les trois évaluations sont enregistrées.

La [partie 1](rapport_final_partie_1.md) conserve le snapshot historique de la pause, publié dans les commits `008981d` et `72034bd`. Cette partie 2 regroupe les résultats achevés, les graphiques vérifiés et l'édition finale du mémoire.

## Entraînement terminé et sélection

La graine TCN 101 a repris depuis la sauvegarde durable de la partie 1, après deux époques et 800 lots de la troisième, avec poids, optimiseur et états aléatoires. Elle a terminé dix époques et 12 000 mises à jour ; son meilleur checkpoint est de la septième époque. Les graines 42 et 73, déjà terminées, ont été conservées.

Les trois log-loss de validation non pondérées sont 0,392392 (graine 42), 0,463125 (73) et 0,401111 (101). **Le TCN retenu est la graine 42, époque 7**, sélectionnée sur validation avant calibration et test. Ce modèle contient 290 505 paramètres ; il a été entraîné depuis zéro sur toute la séquence encodée, sans troncature.

Les données, la configuration, les sept sources figées et le CNN actif restent identiques au protocole. Le dataset conserve l'identifiant `69395a7f7f7d66a9066687445aca249e9d67a994a12b1466920fb4e64102b465`. Les interventions d'exécution, la pause, la reprise et la fin figurent dans [runtime_notes.json](../results/benchmark/cnn-xgboost-tcn-20261005/runtime_notes.json).

## Résultats finaux

Les mêmes **1 709 contrats de test** sont utilisés, dans le même ordre : 867 négatifs et 842 positifs annotés. Les valeurs proviennent des JSON d'évaluation, après température et seuil principal ajustés exclusivement sur calibration.

| Modèle | Graine retenue | F1 macro | Précision positive | Rappel positif | FPR ↓ | PR-AUC (AP) |
|---|---:|---:|---:|---:|---:|---:|
| CNN–BiLSTM actif | 73, historique | 82,10 % | 77,94 % | 88,95 % | 24,45 % | 0,8967 |
| TF-IDF + XGBoost | 101 | **89,41 %** | **89,11 %** | **89,43 %** | **10,61 %** | **0,9534** |
| TCN depuis zéro | 42, époque 7 | 84,77 % | 81,56 % | 89,31 % | 19,61 % | 0,9234 |

PR-AUC correspond à l'average precision, pas à une intégration trapézoïdale. Les matrices, annotations réelles en lignes et prédictions en colonnes dans l'ordre `[0,1]`, sont : CNN `[[655,212],[93,749]]`, XGBoost `[[775,92],[89,753]]`, TCN `[[697,170],[90,752]]`.

Sur ce corpus, XGBoost conserve le meilleur F1 et réduit le plus les faux positifs au seuil principal. Le TCN améliore le F1 du CNN actif de **2,67 points**, en réduisant les faux positifs de 212 à 170, mais reste **4,64 points** sous XGBoost. Ces résultats n'établissent pas une supériorité générale de l'apprentissage profond.

## Incertitude et portée

Le bootstrap apparié existant utilise les groupes du dataset, 1 000 répétitions et la graine 42. Ses intervalles sont conditionnés aux checkpoints sélectionnés, exploratoires et sans correction des comparaisons multiples.

| Différence de F1 macro | Écart, points | IC 95 %, points |
|---|---:|---:|
| XGBoost − CNN–BiLSTM | +7,31 | [+4,87 ; +10,65] |
| TCN − CNN–BiLSTM | +2,67 | [+0,27 ; +5,60] |
| TCN − XGBoost | −4,64 | [−7,43 ; −2,61] |

Les intervalles de rappel incluent zéro pour les trois paires, sans prouver une équivalence. L'ancien intervalle XGBoost − CNN de [+4,94 ; +10,80] utilisait la graine de bootstrap 20261005 ; les mesures ponctuelles des deux baselines sont inchangées. Le [rapport détaillé](comparaison_cnn_xgboost_tcn.md) explique cette provenance et les seuils alternatifs.

Le test avait déjà été observé. Une confirmation sur corpus indépendant reste nécessaire. Les labels historiques sont en partie issus d'analyseurs statiques ; une annotation négative ne certifie pas la sécurité. La source réservée n'a que six négatifs, ce qui rend son FPR fragile. Les budgets et représentations diffèrent, et le checkpoint CNN ainsi que les calculs XGBoost sont réutilisés avec provenance : la comparaison n'isole pas l'architecture à budget égal.

La pause demandée, du 6 octobre vers **00 h 50 à 18 h 52 (UTC+1)** avec processus arrêté, est exclue du temps cumulé de la graine 101. Les segments actifs et les anciennes graines peuvent inclure concurrence CPU et suspensions du système. Les timers d'inférence n'incluent pas les mêmes opérations. Aucun classement de vitesse n'est présenté.

## Livrables et vérifications

- [Résultats complets et checkpoints sélectionnés](../results/benchmark/cnn-xgboost-tcn-20261005/summary.json), [rapport calculé](../results/benchmark/cnn-xgboost-tcn-20261005/report.md) et [différences appariées](../results/benchmark/cnn-xgboost-tcn-20261005/paired_comparisons.json).
- [Comparaison rédigée](comparaison_cnn_xgboost_tcn.md) et [protocole actualisé](protocole_cnn_xgboost_tcn.md), avec définition de l'AP, architecture adaptée, origine des modèles, limites et traitement des durées.
- **11 figures finales en couleurs**, chacune en PNG 300 dpi et SVG : architecture TCN, validation des trois graines, métriques, matrices combinée et individuelles, ROC, précision–rappel, écarts appariés et pertes du TCN retenu. [Manifeste des 22 exports](../results/benchmark/cnn-xgboost-tcn-20261005/plots/plot_manifest.json).
- [Mémoire PDF final](../output/pdf/Memoire_SMART_BUG_Exauce_Kompani.pdf) et [version Word modifiable](../output/documents/Memoire_SMART_BUG_Exauce_Kompani.docx) : 44 pages, 15 figures, 8 tableaux, 14 références et 44 abréviations. Les valeurs des trois modèles et les trois intervalles correspondent aux résultats archivés ; les remerciements restent identiques dans les sources, le texte et les pixels des deux pages.
- [Sources et régénération du mémoire](../docs/memoire/README.md), [contrôles finaux](../docs/memoire/qa.json) et [inspection des 44 pages](../docs/memoire/visual_inspection_final.json), avec les empreintes des livrables vérifiés.

Le générateur vérifie sélection, calibration, empreintes des poids et du TF-IDF, identité du test, ordre des identifiants, prédictions finies, matrices et métriques avant les graphiques. Les empreintes des 28 entrées et des 22 exports ont été relues, les 11 PNG inspectés et les cinq tests ciblés du générateur réussis après la correction de contraste. L'audit indépendant confirme aussi les 15 fichiers du dataset, les sept sources figées, la sélection des trois graines et les métriques recalculées depuis les logits sauvegardés, sans nouvel entraînement ni bootstrap.

Les figures des intervalles et des pertes sélectionnées ont un format fixe de 9 × 4,1 pouces pour la mise en page du mémoire. Les anciennes figures à deux modèles restent une archive de partie 1. Le Word a été rendu avec le renderer documentaire canonique, LibreOffice et Poppler ; les 44 pages ont été inspectées à leur résolution d'origine. Les derniers ajustements ont été rendus et contrôlés à nouveau, avec vérification des hashes des pages inchangées. Aucun nouveau modèle n'est activé automatiquement dans SMART BUG.

Les références primaires de Bai et al. (2018), Gopali et al. (2022, TCN sur opcodes EVM) et Chen et Guestrin (2016) figurent dans le [rapport détaillé](comparaison_cnn_xgboost_tcn.md). Le TCN local sur tokens Solidity est une adaptation, et aucun score externe n'entre dans les tableaux de cette campagne.
