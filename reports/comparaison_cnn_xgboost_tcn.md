# Comparaison expérimentale : CNN–BiLSTM, XGBoost et TCN

Les **trois modèles sont évalués** dans le run `cnn-xgboost-tcn-20261005`, terminé le 6 octobre 2026. Sur les 1 709 contrats du test commun, XGBoost obtient le meilleur F1 macro (89,41 %), suivi du TCN (84,77 %) puis du CNN–BiLSTM actif (82,10 %). Ces résultats décrivent ce corpus et les checkpoints sélectionnés ; ils ne démontrent pas une supériorité générale d'une famille de modèles.

Les valeurs ci-dessous proviennent de `splits.test.calibrated` dans les trois fichiers `evaluation.json`, repris dans [summary.json](../results/benchmark/cnn-xgboost-tcn-20261005/summary.json). Le [protocole](protocole_cnn_xgboost_tcn.md) fixe les données, hyperparamètres, règles de sélection et limites. Le [rapport produit par le runner](../results/benchmark/cnn-xgboost-tcn-20261005/report.md) conserve les résultats calculés.

## Comparabilité et sélection

Le snapshot conserve l'identifiant `69395a7f7f7d66a9066687445aca249e9d67a994a12b1466920fb4e64102b465` : 11 959 contrats de train, puis 1 709 de validation, de calibration et de test, avec une source réservée de 1 152 contrats. Le test contient 867 négatifs et 842 positifs annotés. L'ordre des identifiants est identique pour les trois modèles et correspond au fichier canonique du snapshot.

Le CNN est le checkpoint **TensorFlow actif de graine 73**, historiquement sélectionné sur validation. Il ne s'agit pas du réseau PyTorch de l'ancienne étude. XGBoost réutilise les entraînements et le TF-IDF appris sur train de la campagne précédente, avec [provenance et empreintes](../results/benchmark/cnn-xgboost-tcn-20261005/imported_baselines.json). Son checkpoint retenu est de graine 101. Le TCN est appris depuis zéro avec les graines 42, 73 et 101. Chaque nouveau modèle est sélectionné par log-loss non pondérée de validation minimale, puis par graine en cas d'égalité ; aucun score de test ne sert à cette sélection.

| Graine TCN | Époques exécutées | Meilleure époque | Log-loss validation du checkpoint |
|---|---:|---:|---:|
| **42, retenue** | 10 | 7 | **0,392392** |
| 73 | 5 | 2 | 0,463125 |
| 101 | 10 | 7 | 0,401111 |

Le TCN compte 290 505 paramètres. Il traite la séquence encodée entière par neuf blocs résiduels de deux convolutions causales dilatées, sans weight normalization, puis un pooling moyenne et maximum masqués. Son champ réceptif local est de 2 045 **positions encodées**, distinctes des tokens lexicaux lorsqu'un token inconnu est échappé en octets. Le pooling inclut toutes les positions, sans créer des interactions temporelles arbitrairement lointaines. Le vocabulaire est appris uniquement sur train ; l'encodage conserve les tokens après normalisation, et non le code source original.

Après sélection, température et seuil sont ajustés uniquement sur calibration. Le seuil principal maximise la précision sous une cible de rappel de 90 % sur calibration. Cette cible n'est pas une garantie de rappel sur test. Les tableaux et les points marqués sur ROC/PR utilisent la règle `sigmoid(logit / température) >= seuil`.

| Modèle sélectionné | Température | Seuil principal |
|---|---:|---:|
| CNN–BiLSTM, graine 73 | 0,936841 | 0,269474 |
| TF-IDF + XGBoost, graine 101 | 1,246681 | 0,593873 |
| TCN, graine 42, époque 7 | 1,181506 | 0,467699 |

## Résultats sur le test commun

| Modèle | F1 macro | Précision positive | Rappel positif | FPR ↓ | PR-AUC (AP) | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| CNN–BiLSTM actif | 82,10 % | 77,94 % | 88,95 % | 24,45 % | 0,8967 | 0,9069 |
| TF-IDF + XGBoost | **89,41 %** | **89,11 %** | **89,43 %** | **10,61 %** | **0,9534** | **0,9569** |
| TCN depuis zéro | 84,77 % | 81,56 % | 89,31 % | 19,61 % | 0,9234 | 0,9244 |

PR-AUC désigne ici l'**average precision (AP)**, pas l'aire trapézoïdale. FPR est la proportion de négatifs annotés prédits positifs ; plus bas est meilleur. Les matrices ont les annotations réelles en lignes et les classes prédites en colonnes, dans l'ordre `[0, 1]`.

| Modèle | Vrais négatifs | Faux positifs | Faux négatifs | Vrais positifs |
|---|---:|---:|---:|---:|
| CNN–BiLSTM actif | 655 | 212 | 93 | 749 |
| TF-IDF + XGBoost | 775 | 92 | 89 | 753 |
| TCN depuis zéro | 697 | 170 | 90 | 752 |

Le gain du TCN sur le CNN provient surtout de 42 faux positifs de moins, avec trois positifs annotés supplémentaires détectés. XGBoost réduit encore les faux positifs à 92 et détecte 753 positifs. Les rappels principaux sont proches ; cette campagne ne justifie pas d'affirmer que l'apprentissage profond domine TF-IDF + XGBoost.

Un **seuil alternatif**, également fixé sur calibration, vise FPR ≤ 10 %. Il correspond à un autre compromis que le tableau principal ; la courbe ROC de test ne sert pas à choisir un seuil de déploiement.

| Modèle | Seuil alternatif | Rappel sur test | FPR sur test |
|---|---:|---:|---:|
| CNN–BiLSTM actif | 0,718204 | 62,95 % | 7,15 % |
| TF-IDF + XGBoost | 0,744112 | 79,93 % | 6,57 % |
| TCN depuis zéro | 0,709805 | 67,70 % | 6,46 % |

## Différences appariées

Les intervalles ci-dessous viennent de [paired_comparisons.json](../results/benchmark/cnn-xgboost-tcn-20261005/paired_comparisons.json) : bootstrap apparié par `group_id`, 1 000 répétitions, graine 42, checkpoints sélectionnés fixés. L'empreinte des trois évaluations correspond aux fichiers actuels. Aucun bootstrap supplémentaire n'est exécuté pour les graphiques ou ce rapport.

| Différence, gauche − droite | F1 macro, points | IC 95 % du F1, points | Rappel, points | IC 95 % du rappel, points |
|---|---:|---:|---:|---:|
| XGBoost − CNN–BiLSTM | +7,31 | [+4,87 ; +10,65] | +0,48 | [−2,16 ; +2,85] |
| TCN − CNN–BiLSTM | +2,67 | [+0,27 ; +5,60] | +0,36 | [−3,57 ; +3,65] |
| TCN − XGBoost | −4,64 | [−7,43 ; −2,61] | −0,12 | [−3,76 ; +2,98] |

Les trois intervalles de F1 excluent zéro dans le sens du classement observé. Les intervalles de rappel incluent zéro : une différence de rappel n'est pas établie par ces intervalles, ce qui ne prouve pas l'équivalence. Ils restent exploratoires, ne couvrent pas toute la variabilité d'un nouvel entraînement et ne sont pas corrigés pour comparaisons multiples.

L'archive à deux modèles utilisait une graine de bootstrap **20261005** et donnait pour XGBoost − CNN un IC de [+4,94 ; +10,80]. La campagne actuelle utilise la graine **42** figée dans sa configuration : les rééchantillonnages diffèrent, tandis que l'écart ponctuel reste +7,31. Le code du bootstrap est identique ; chaque paire réinitialise son générateur aléatoire. Les deux archives conservent leur provenance respective.

## Diagnostic entre sources et limites

Sur la source réservée, le rappel positif vaut 89,01 % pour CNN–BiLSTM, 78,10 % pour XGBoost et 85,34 % pour TCN. Cette partition contient **1 146 positifs et seulement six négatifs** : son FPR et son F1 macro sont fragiles, et une AP élevée y bénéficie d'une prévalence presque entièrement positive. Ce diagnostic ne confirme pas un classement général identique à celui du test principal.

Le test a déjà été observé : l'étude doit être confirmée sur un corpus indépendant. Les annotations sont en partie issues d'analyseurs statiques, et la classe 0 ne certifie pas la sécurité. Les représentations, tailles, optimiseurs et budgets de sélection diffèrent ; la réutilisation du CNN actif et des entraînements XGBoost est déclarée. Il ne s'agit pas d'une comparaison à budget égal isolant le seul effet architectural.

La graine 101 a repris le 6 octobre depuis le checkpoint sauvegardé à la pause. La période d'arrêt demandée par l'utilisateur, vers 00 h 50–18 h 52 (UTC+1), est exclue de son temps cumulé enregistré. Les segments actifs et les anciennes graines peuvent inclure concurrence CPU et suspensions du système. Les timers d'inférence sont également différents : XGBoost inclut le chargement de ses artefacts. **Aucun classement de vitesse** n'est tiré de ces durées. La pause, la reprise et la fin sont consignées dans [runtime_notes.json](../results/benchmark/cnn-xgboost-tcn-20261005/runtime_notes.json).

## Graphiques et traçabilité

Les [11 graphiques finaux](../results/benchmark/cnn-xgboost-tcn-20261005/plots/plot_manifest.json) sont exportés en PNG à 300 dpi et en SVG : architecture, validation des trois graines, cinq indicateurs principaux, matrices combinée et individuelles, ROC, précision–rappel, écarts appariés de F1 et pertes du TCN sélectionné. Les deux courbes de perte distinguent explicitement BCE de train pondérée avec dropout et log-loss de validation non pondérée sans dropout ; leur écart n'est pas une mesure directe de surapprentissage.

Le manifeste enregistre les 28 fichiers d'entrée, les 22 exports, le générateur, le protocole, les checkpoints, températures, seuils, métriques et couleurs. Les empreintes ont été vérifiées et les 11 PNG inspectés. Les graphiques historiques de deux modèles restent conservés dans `plots/modeles_evalues/`.

Reproduction des exports, sans entraînement ni nouvelle calibration :

```powershell
.venv-comparison/Scripts/python.exe -m src.plot_model_benchmark
```

Cette comparaison n'active pas automatiquement un nouveau modèle dans SMART BUG.

## Références primaires

- Bai, S., Kolter, J. Z. et Koltun, V. (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*. [arXiv:1803.01271](https://arxiv.org/abs/1803.01271). Les convolutions causales dilatées et blocs résiduels motivent cette architecture adaptée ; l'implémentation locale n'en est pas une reproduction exacte.
- Gopali, S., Khan, Z. A., Chhetri, B., Karki, B. et Namin, A. S. (2022). *Vulnerability Detection in Smart Contracts Using Deep Learning*. IEEE COMPSAC, p. 1249–1255. [DOI 10.1109/COMPSAC54236.2022.00197](https://doi.org/10.1109/COMPSAC54236.2022.00197). Cette étude applique un TCN à des séquences d'opcodes EVM ; notre entrée en tokens Solidity normalisés et notre tâche sont différentes. Aucun score du papier n'est mélangé aux mesures locales.
- Chen, T. et Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. [arXiv:1603.02754](https://arxiv.org/abs/1603.02754).
