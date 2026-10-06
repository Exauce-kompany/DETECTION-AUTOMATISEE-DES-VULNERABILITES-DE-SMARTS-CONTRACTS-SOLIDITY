# Comparaison : CNN–BiLSTM, XGBoost et TCN

Le TCN remplace CodeBERT dans la comparaison principale, à la demande de l'utilisateur le 5 octobre 2026. Le pilote CodeBERT reste une archive de l'option abandonnée pour son coût CPU. La revue de littérature explique les méthodes ; seuls les scores calculés sur les mêmes contrats servent à comparer les modèles.

| Modèle | Apprentissage | Représentation |
|---|---|---|
| CNN–BiLSTM actif | Depuis zéro ; checkpoint TensorFlow historique de graine 73 | CNN sur fenêtres de 256 positions encodées, puis BiLSTM sur les fenêtres |
| TF-IDF + XGBoost | Depuis zéro ; trois graines | Fréquences de tous les tokens normalisés, unigrammes et bigrammes |
| TCN | Depuis zéro ; trois graines | Convolutions causales dilatées résiduelles sur toute la séquence encodée, puis pooling global |

## Données et prévention des fuites

Le snapshot `dataset/benchmark` conserve l'identifiant `69395a7f7f7d66a9066687445aca249e9d67a994a12b1466920fb4e64102b465`.

| Partition | Contrats ou représentations | Usage |
|---|---:|---|
| Train | 11 959 | Paramètres et vocabulaires |
| Validation | 1 709 | Meilleure époque et sélection de graine |
| Calibration | 1 709 | Température et seuils |
| Test | 1 709 | Évaluation après sélection |
| Source réservée | 1 152 | Diagnostic entre sources |

Les groupes restent disjoints selon l'audit existant. Les contrats et labels binaires sont communs. Le vocabulaire neuronal de 8 000 entrées est construit uniquement sur train. Les tokens inconnus sont encodés en octets UTF-8 avec marqueurs. L'encodage conserve les tokens **après normalisation**, pas les commentaires, noms et chaînes transformés en amont.

Le TCN utilise tous les tokens encodés, sans troncature aux 512 premières positions. Sa sortie et sa BCE portent sur le contrat complet. Les pondérations communes de `src.experiment.training_weights` équilibrent les strates source/label à partir du train uniquement. Validation, calibration et test sont non pondérés.

XGBoost et TCN utilisent les graines 42, 73 et 101. La sélection se fait par log-loss validation minimale, puis par graine en cas d'égalité. Le CNN garde son checkpoint précédemment sélectionné. Le tableau principal compare des checkpoints sélectionnés, sans mélanger un score individuel avec une moyenne de graines.

## Hyperparamètres

Valeurs exactes : `config/benchmark_models.json`.

- **CNN–BiLSTM** : poids existants ; embedding 32, 24 filtres de largeur 5, 24 unités BiLSTM par sens, dropout 0,30 ; Adam à 0,001, maximum historique de 12 époques et patience 3.
- **XGBoost** : TF-IDF appris sur train, n-grammes 1–2, au plus 20 000 caractéristiques, `min_df=2`, fréquence sous-linéaire. Au plus 600 arbres, profondeur 6, taux 0,05, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=2`, `reg_lambda=1`, méthode `hist`, objectif `binary:logistic`, log-loss validation et patience 40. Les calculs terminés sont réutilisés avec empreintes et provenance.
- **TCN** : embedding 32 ; 24 canaux ; neuf blocs résiduels de deux convolutions causales chacun ; noyau 3 ; dilations `[1,2,4,8,16,32,64,128,256]` ; ReLU ; projection 1×1 quand les dimensions diffèrent ; dropout élémentaire de 0,20 après les convolutions et avant la tête ; sans weight normalization ; pooling moyenne et maximum masqués ; dense 32 puis logit binaire. AdamW à 0,001, `weight_decay=0.0001`, clipping à 1,0 ; au plus 12 époques, patience 3. Mini-lots regroupés par longueur, au plus 32 contrats et budget de padding de 16 384 positions, sauf contrat individuel plus long. CPU à six threads. Reprise atomique avec optimiseur et états aléatoires.

Le champ réceptif du TCN vaut `1 + 2 × (3−1) × (1+2+4+8+16+32+64+128+256) = 2 045` **positions encodées**. Chaque position voit au plus ce contexte passé. Le pooling global agrège toute la séquence, mais ne permet pas des interactions arbitraires entre instructions éloignées. Les PAD sont masqués à chaque couche et exclus du pooling.

Il s'agit d'une **adaptation**, pas d'une reproduction exacte de Gopali et al. : leur étude travaille sur des opcodes EVM et d'autres familles de vulnérabilités. Les tailles, budgets et optimisateurs des trois solutions diffèrent ; cette comparaison n'isole pas l'effet architectural à budget égal.

## Calibration et indicateurs

Après sélection, la température est ajustée exclusivement sur calibration. Le seuil principal maximise la précision avec une cible de rappel de 90 % sur calibration ; un seuil alternatif vise FPR ≤ 10 %. Ces objectifs ne garantissent pas les mêmes valeurs sur test.

Les résultats comprennent exactitude, F1 macro, précision/rappel positifs, confusion, FPR, ROC-AUC, PR-AUC, Brier, ECE et diagnostics par annotations. Dans ces rapports, **PR-AUC désigne l'average precision (AP)**, et non une intégration trapézoïdale de la courbe précision–rappel. Les identifiants de test sont vérifiés avant les comparaisons appariées. Un bootstrap par groupe, 1 000 répétitions et graine 42, estime les différences entre checkpoints sélectionnés : intervalles exploratoires, conditionnés aux entraînements retenus, non corrigés pour comparaisons multiples.

Les temps d'inférence enregistrés portent sur des lots complets. Le timer XGBoost inclut son chargement, alors que CNN et TCN chargent leurs poids avant ; ces chiffres ne permettent pas un classement de vitesse. Le pilote TCN estime le coût des époques prévues dans ses conditions de mesure ; ce n'est pas une borne garantie de durée réelle. Le temps d'entraînement cumulé de la graine 101 additionne les segments enregistrés du processus : **la pause demandée par l'utilisateur, durant laquelle le processus était arrêté du 6 octobre vers 00 h 50 à 18 h 52 (UTC+1), en est exclue**. Les segments actifs et les anciennes graines peuvent inclure concurrence CPU, suspensions du système et attentes d'affichage. La campagne a été reprise le 6 octobre depuis les sauvegardes disponibles, avec journaux locaux sur disque ; les hyperparamètres et règles de sélection ont été conservés. Un autre entraînement a occupé une grande partie du CPU. La priorité du seul processus TCN a été temporairement augmentée avant la pause ; cette intervention sur l'ordonnancement, la pause, la reprise et la fin sont consignées dans `runtime_notes.json` et ne modifient ni l'architecture ni les six threads de calcul. Les durées ne permettent donc pas de comparer équitablement la vitesse d'entraînement.

## Fichiers et exécution

| Fichier | Fonction |
|---|---|
| `config/benchmark_models.json` | Modèles, graines, partitions et hyperparamètres |
| `src/run_model_benchmark.py` | Vérification, import des baselines, sélection, calibration, évaluation et rapport |
| `src/benchmark_tcn.py` | Architecture TCN, pilote, entraînement complet et reprise |
| `results/benchmark/cnn-xgboost-tcn-20261005/protocol.json` | Configuration figée, empreintes du code et environnement |
| `results/benchmark/cnn-xgboost-tcn-20261005/imported_baselines.json` | Origine et intégrité des résultats CNN/XGBoost réutilisés |
| `results/benchmark/cnn-xgboost-tcn-20261005/runtime_notes.json` | Priorité CPU, pause, reprise, fin et limites des mesures de durée |
| `results/benchmark/cnn-xgboost-tcn-20261005/<modèle>/` | Historiques, sélection, calibration, prédictions et métriques |
| `models/benchmark/cnn-xgboost-tcn-20261005/tcn/` | Poids et checkpoints locaux de reprise TCN |

Dans l'environnement scientifique dédié :

```powershell
python -m src.run_model_benchmark --stage import-baselines
python -m src.run_model_benchmark --stage tcn-pilot
python -m src.run_model_benchmark --stage tcn
python -m src.run_model_benchmark --stage report
```

Modifier la configuration ou le code figé exige un nouveau `--run-id`. Aucun modèle de cette comparaison n'est automatiquement activé dans SMART BUG.

## Limites et références

Le test a déjà été observé : étude **exploratoire**, à confirmer sur un corpus indépendant. Les annotations sont en partie issues d'analyseurs statiques ; la classe 0 ne certifie pas la sécurité. La source réservée ne contient que six négatifs : son FPR est fragile.

- Bai, Kolter et Koltun (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*. [Article original](https://arxiv.org/abs/1803.01271).
- Gopali et al. (2022). *Vulnerability Detection in Smart Contracts Using Deep Learning*. IEEE COMPSAC, p. 1249–1255. [DOI](https://doi.org/10.1109/COMPSAC54236.2022.00197).
- Chen et Guestrin (2016). *XGBoost: A Scalable Tree Boosting System*. [Article original](https://arxiv.org/abs/1603.02754).
