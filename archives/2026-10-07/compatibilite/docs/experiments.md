# Expériences, résultats et reproduction

Ce guide reprend les résultats scientifiques et les consignes du README antérieur. Les valeurs et conclusions sont conservées ; les commandes utilisent la nouvelle organisation des modules. Exécuter toutes les commandes depuis la racine du dépôt, dans l'environnement Python approprié.

La réorganisation change les empreintes du code scientifique. Pour de nouvelles campagnes de comparaison ou de benchmark, fournir un identifiant inédit avec `--run-id`. La reprise des anciennes campagnes doit continuer à refuser un code dont les empreintes diffèrent. Utiliser `--config` avec une copie de la configuration si les paramètres changent. Les résultats et modèles déjà publiés restent des artefacts historiques consultables. Voir [l'architecture](architecture.md#provenance-et-compatibilité).

## Reproduire et vérifier

Les données et poids sont déjà fournis. Pour vérifier le dataset puis entraîner une nouvelle expérience indépendante :

```powershell
python -m src.data.audit_dataset
python -m src.training.train
python -m unittest discover -s tests -v
```

Chaque entraînement écrit un nouveau dossier et active son modèle uniquement après la fin de l'évaluation. Les graines et hyperparamètres sont figés avant l'accès au test. Ne pas ajuster les paramètres selon les résultats de ce test : une nouvelle recherche nécessiterait une nouvelle réserve de test.

Pour reconstruire les données depuis les sources brutes conservées dans `dataset/sources/`, sans écraser le snapshot distribué :

```powershell
python -m src.data.build_dataset --output dataset/benchmark_rebuilt
python -m src.data.audit_dataset --dataset dataset/benchmark_rebuilt
```

Le constructeur refuse un dossier de destination non vide. Pour entraîner sur un autre dossier, utiliser une copie de la configuration dont `dataset_dir` pointe vers ce dossier, et fournir la même configuration à la construction et à l'entraînement via `--config`.

Les dépendances de l'expérience sont enregistrées dans `experiment.json` et `requirements.txt`. Le déterminisme est activé ; l'identité bit à bit entre systèmes, versions de bibliothèques et processeurs différents n'est pas garantie.

## Résultats SMART BUG

Expérience : `smartbug-403bb881-20260922T101754Z`. Modèle sélectionné sur validation : graine **73**, époque **3**, **277 017 paramètres**. Température : **0,936841** ; seuil choisi sur la calibration : **0,269474**.

| Modèle / décision | Exactitude test | F1 macro test | Rappel positif test |
|---|---:|---:|---:|
| CNN + BiLSTM, score brut, seuil 0,5 | 82,45 % | 82,41 % | 79,10 % |
| CNN + BiLSTM, calibration et seuil actif | **82,15 %** | **82,10 %** | **88,95 %** |
| TF-IDF + régression logistique, calibrée | 81,39 % | 81,39 % | 84,20 % |

Le test contient 1 709 contrats ; la décision active produit 212 faux positifs et 93 faux négatifs. L'objectif de rappel de 90 % est choisi sur la calibration et n'est pas garanti sur le test. La calibration améliore légèrement le Brier (0,125856 → 0,125753) et l'ECE à 10 classes (5,65 % → 5,12 %) sur ce test ; elle ne rend pas les scores universellement fiables.

Les 1 152 contrats de la source réservée comprennent **1 146 positifs et seulement 6 négatifs** : le rappel positif est de 89,01 %, mais l'estimation des faux positifs est très fragile et l'ECE atteint 29,84 %. Ces données ne prouvent pas une généralisation suffisante à toutes les sources. Des négatifs vérifiés et comparables restent à recueillir.

Zéro recouvrement détecté entre les cinq partitions selon les six critères de l'audit. Cela ne prouve pas l'absence de tous les clones approximatifs. Les labels contradictoires sont exclus, pas corrigés arbitrairement. La classe 0 signifie une absence de signal selon les annotations apprises, **pas une certification de sécurité**.

Les scores de ce tableau concernent le modèle TensorFlow actif, et doivent être distingués de ceux des réseaux PyTorch entraînés pour la comparaison ci-dessous.

## Graphiques du modèle SMART BUG actif

Les cinq figures (exactitude, perte, matrice de confusion, métriques globales et métriques par classe) sont disponibles dans [`results/training/smartbug-403bb881-20260922T101754Z/plots/`](../results/training/smartbug-403bb881-20260922T101754Z/plots/), en PNG et SVG. Elles sont calculées à partir des résultats enregistrés ; aucun réentraînement n’est nécessaire. Le manifeste des figures précise leurs fichiers sources et leurs empreintes.

```powershell
python -m src.visualization.plot_results
```

## Comparaison principale : CNN–BiLSTM, XGBoost et TCN

Le [protocole](../reports/protocole_cnn_xgboost_tcn.md) compare le **CNN–BiLSTM actif**, **TF-IDF + XGBoost** et un **TCN entraîné depuis zéro** sur les mêmes partitions. Le TCN analyse tous les tokens encodés par des convolutions causales dilatées résiduelles, puis une agrégation globale. La revue de littérature reste qualitative ; les scores d'autres publications ne servent pas à classer ces modèles.

Réglages : `config/benchmark_models.json`. Exécution : `src/experiments/benchmark/run_model_benchmark.py`, avec `src/experiments/benchmark/benchmark_tcn.py` pour le TCN. Installer `requirements-benchmark.txt` dans l'environnement scientifique dédié.

```powershell
python -m src.experiments.benchmark.run_model_benchmark --run-id cnn-xgboost-tcn-nouvelle-campagne --stage import-baselines
python -m src.experiments.benchmark.run_model_benchmark --run-id cnn-xgboost-tcn-nouvelle-campagne --stage tcn-pilot
python -m src.experiments.benchmark.run_model_benchmark --run-id cnn-xgboost-tcn-nouvelle-campagne --stage tcn
python -m src.experiments.benchmark.run_model_benchmark --run-id cnn-xgboost-tcn-nouvelle-campagne --stage report
python -m src.visualization.plot_model_benchmark --run-id cnn-xgboost-tcn-nouvelle-campagne
```

Les trois modèles sont désormais évalués sur les mêmes **1 709 contrats de test**. Chaque ligne rapporte le checkpoint sélectionné sur validation, avec une température et un seuil ajustés sur calibration ; il ne s'agit pas d'une moyenne de graines sur test.

| Modèle | F1 macro | Précision positive | Rappel positif | FPR | AP |
|---|---:|---:|---:|---:|---:|
| CNN–BiLSTM actif | 82,10 % | 77,94 % | 88,95 % | 24,45 % | 89,67 % |
| TF-IDF + XGBoost | **89,41 %** | **89,11 %** | **89,43 %** | **10,61 %** | **95,34 %** |
| TCN depuis zéro | 84,77 % | 81,56 % | 89,31 % | 19,61 % | 92,34 % |

Le TCN retenu est celui de la **graine 42, époque 7**, avec **290 505 paramètres**. Ses trois graines ont terminé 10, 5 et 10 époques. XGBoost obtient le meilleur F1 macro sur ce test exploratoire. Le TCN dépasse le CNN de 2,67 points, avec un IC bootstrap à 95 % de [0,27 ; 5,60] points, et reste 4,64 points sous XGBoost. Ces intervalles sont conditionnés aux checkpoints retenus et ne mesurent pas toute la variabilité des réentraînements. Les budgets diffèrent et les labels sont en partie issus d'analyseurs statiques : ces résultats demandent une confirmation indépendante.

Les [matrices de confusion des trois modèles](../results/benchmark/cnn-xgboost-tcn-20261005/plots/matrices_confusion.png), [métriques](../results/benchmark/cnn-xgboost-tcn-20261005/plots/metriques_principales.png), [courbes ROC](../results/benchmark/cnn-xgboost-tcn-20261005/plots/courbe_roc.png), [précision–rappel](../results/benchmark/cnn-xgboost-tcn-20261005/plots/courbe_precision_rappel.png) et [écarts de F1 avec intervalles](../results/benchmark/cnn-xgboost-tcn-20261005/plots/ecarts_f1_ic95.png) sont disponibles en couleur, en PNG et SVG. Le [manifeste des onze figures](../results/benchmark/cnn-xgboost-tcn-20261005/plots/plot_manifest.json) précise les sources, empreintes, seuils et matrices. L'indicateur AP est l'**average precision**, également noté PR-AUC (AP) dans les graphiques ; ce n'est pas une intégration trapézoïdale de la courbe.

Le mode suivant trace tous les modèles dont l'évaluation est terminée, même pour une autre campagne encore incomplète :

```powershell
python -m src.visualization.plot_model_benchmark --available-models
```

Les [graphiques partiels CNN–BiLSTM/XGBoost de partie 1](../results/benchmark/cnn-xgboost-tcn-20261005/plots/modeles_evalues/README.md) restent une archive de la pause. La commande principale sans option produit les figures finales dans le dossier parent. Les commandes de tracé utilisent les checkpoints et seuils enregistrés.

Le [rapport des calculs](../results/benchmark/cnn-xgboost-tcn-20261005/report.md) contient les résultats complets ; le [rapport d'analyse](../reports/comparaison_cnn_xgboost_tcn.md) en explique les écarts et les limites. Le test conserve son rôle exploratoire. Les calculs CNN et XGBoost antérieurs sont réutilisés avec leurs empreintes et sans dupliquer leurs poids. Le pilote CodeBERT reste une archive de l'option remplacée pour son coût sur CPU. Les pauses et la concurrence CPU sont consignées dans les notes d'exécution ; les durées ne permettent pas un classement équitable des vitesses. L'application SMART BUG conserve son modèle actif désigné par `models/active_model.json`.

## Archive de l'expérience CNN/BiLSTM/CodeT5/TF-IDF

Le [protocole de comparaison](../reports/protocole_comparaison_v1.md) couvre CNN,
BiLSTM, CNN-BiLSTM, CodeT5-small figé et TF-IDF, avec entrées complètes ou tronquées.
Les expériences sont isolées de l'application et n'activent aucun nouveau modèle.
Le suffixe `v1` de cette expérience désigne la première version du protocole comparatif sur SMART BUG, et non le modèle historique V1. Ces identifiants sont conservés pour respecter les manifestes figés.
Elles utilisent `config/comparison_v1.json` et un environnement dédié avec les
dépendances de `requirements-comparison.txt`, en complément de `requirements.txt`.
La commande `python -u -m src.experiments.comparison.run_comparison --run-id comparison-nouvelle-campagne` enchaîne la progression ; l'état et
les résultats sont écrits sous `results/comparison/`. Le test SMART BUG étant déjà
consulté, cette comparaison est exploratoire et demande une confirmation indépendante.

Comparaison terminée le 23 septembre 2026 : **26 entraînements et évaluations**
(18 réseaux depuis zéro, six têtes CodeT5 et deux références TF-IDF).
Voir la [conclusion finale de l'expérience](../reports/comparaison_v1_etat.md), le
[rapport complet](../results/comparison/comparison-v1-20260922/report.md) et le
[tableau des résultats](../results/comparison/comparison-v1-20260922/summary.csv).

CodeT5 complet est la variante sélectionnée selon le critère fixé sur validation.
Sur le test exploratoire, le CNN-BiLSTM complet atteint **84,54 % de F1 macro**
et **89,07 % de rappel**, contre **83,40 %** et **86,18 %** pour CodeT5 complet.
Les latences médianes mesurées sont respectivement 3,00 ms et 558,48 ms par contrat.
L'écart de F1 entre ces deux méthodes n'est pas établi de façon concluante par
l'intervalle bootstrap exploratoire. La sélection sur validation reste enregistrée ;
aucun nouveau modèle n'a été activé dans SmartBug.

Les checkpoints, les têtes CodeT5 et leurs scalers sont sauvegardés via Git LFS.
L'encodeur CodeT5 reste un téléchargement depuis Hugging Face à la révision
enregistrée ; les caches locaux ne sont pas versionnés. Le rapport distingue
les réseaux nouvellement entraînés pour la comparaison du modèle SMART BUG actif.

## Sources et provenance antérieure

Les scripts, modèles et résultats historiques V1/V2 ont été retirés du dossier de travail. Ils restent accessibles dans l’historique Git, qui n’a pas été réécrit.

Les trois JSON bruts utilisés par SMART BUG ont été déplacés vers `dataset/sources/`, sans modifier leurs octets. `dataset/sources/manifest.json` enregistre leurs SHA-256 et la correspondance avec leurs anciens chemins. `src/data/source_paths.py` résout ces références et vérifie les empreintes lors de la reconstruction. Les mentions de l’ancien chemin dans les configurations et manifestes figés documentent l’expérience d’origine ; elles ne désignent pas une dépendance à un ancien modèle. Les hyperparamètres, les poids numériques et les résultats du modèle actif restent inchangés.

Le nettoyage antérieur a donné aux modules, chemins et métadonnées le nom SMART BUG. Ce renommage avait entraîné une mise à jour explicite des empreintes et des identifiants des manifestes, sans recalcul des résultats ni changement des données. Les champs `metadata_migration` renvoient au commit d’origine. Le [rapport final](../reports/rapport_final.md) documente cette opération et ses vérifications. Une nouvelle reconstruction produit son propre manifeste.

Le sous-module `dataset/raw/smartbugs-curated` est conservé comme source tierce, avec sa révision, son README et sa licence. Les droits sur les sources Solidity restent ceux de leurs auteurs respectifs.
