# SMART BUG — Détection de vulnérabilités Solidity

Le modèle actif **V3** est un CNN + BiLSTM hiérarchique, entraîné **depuis zéro** avec TensorFlow/Keras. Il traite toutes les fenêtres du code et prédit un label binaire issu des annotations du corpus. Aucun modèle pré-entraîné n'est utilisé et aucun fine-tuning n'est réalisé. L'analyse statique et les indications de performance sont des moteurs heuristiques distincts.

Les corrections de l'audit et leurs limites sont détaillées dans [le rapport V3](reports/corrections_v3.md). Les artefacts V1/V2 et [l'audit historique](reports/audit_modele_v2.md) sont conservés pour la traçabilité.

## Installation

Installer Git et [Git LFS](https://git-lfs.com/), puis :

```powershell
git lfs install
git clone --recurse-submodules https://github.com/Exauce-kompany/DETECTION-AUTOMATISEE-DES-VULNERABILITES-DE-SMARTS-CONTRACTS-SOLIDITY.git
cd DETECTION-AUTOMATISEE-DES-VULNERABILITES-DE-SMARTS-CONTRACTS-SOLIDITY
git lfs pull
conda create -n smartcontract python=3.12
conda activate smartcontract
python -m pip install -r requirements.txt
python -m uvicorn webapp.app:app --host 127.0.0.1 --port 8000
```

Ouvrir <http://127.0.0.1:8000>. Le modèle actif est désigné par `models/active_model.json` ; l'application vérifie les empreintes des poids et du vocabulaire avant de les charger. La base SQLite d'historique reste locale et n'est pas versionnée. `SMARTBUG_DATABASE_PATH` permet de choisir une base isolée pour les tests.

Pour un clone existant : `git submodule update --init --recursive`, puis `git lfs pull`.

## Fichiers du protocole actif

| Fichier | Rôle |
|---|---|
| `config/v3.json` | Configuration des données, hyperparamètres et graines |
| `src/preprocessing_v3.py` | Analyse lexicale, retrait des commentaires, normalisation, vocabulaire et fenêtres complètes |
| `src/build_dataset_v3.py` | Quarantaine, déduplication, regroupement, partitionnement et manifestes |
| `src/audit_dataset_v3.py` | Audit indépendant des fichiers, encodages, labels et recouvrements |
| `src/model_v3.py` | Architecture CNN + BiLSTM hiérarchique et masquage |
| `src/experiment_v3.py` | Lots, pondérations, métriques et calibration |
| `src/train_v3.py` | Entraînement sur trois graines, validation, référence TF-IDF, calibration et test final |
| `src/predictor_v3.py` | Inférence partagée par le web et la commande CLI |
| `src/predict_v3.py` | `python -m src.predict_v3 chemin/contrat.sol` |
| `dataset/v3/` | Partitions complètes, entrées encodées, quarantaines et audit |
| `models/v3/<run>/` | Meilleurs modèles par graine, référence et manifeste du modèle sélectionné |
| `results/v3/<run>/` | Historiques, environnement, métriques et prédictions |
| `tests/` | Régressions des données, du modèle et de l'API |

## Reproduire et vérifier

Les données et poids sont déjà fournis. Pour vérifier le dataset puis entraîner une nouvelle expérience indépendante :

```powershell
python -m src.audit_dataset_v3
python -m src.train_v3
python -m unittest discover -s tests -v
```

Chaque entraînement écrit un nouveau dossier et active son modèle uniquement après la fin de l'évaluation. Les graines et hyperparamètres sont figés avant l'accès au test. Ne pas ajuster les paramètres selon les résultats de ce test : une nouvelle recherche nécessiterait une nouvelle réserve de test.

Pour reconstruire les données depuis les JSON V2 sans écraser le snapshot distribué :

```powershell
python -m src.build_dataset_v3 --output dataset/v3_rebuilt
python -m src.audit_dataset_v3 --dataset dataset/v3_rebuilt
```

Le constructeur refuse un dossier de destination non vide. Pour entraîner sur un autre dossier, utiliser une copie de la configuration dont `dataset_dir` pointe vers ce dossier, et fournir la même configuration à la construction et à l'entraînement via `--config`.

Les dépendances de l'expérience sont enregistrées dans `experiment.json` et `requirements.txt`. Le déterminisme est activé ; l'identité bit à bit entre systèmes, versions de bibliothèques et processeurs différents n'est pas garantie.

## Résultats V3

Expérience : `v3-403bb881-20260922T101754Z`. Modèle sélectionné sur validation : graine **73**, époque **3**, **277 017 paramètres**. Température : **0,936841** ; seuil choisi sur la calibration : **0,269474**.

| Modèle / décision | Exactitude test | F1 macro test | Rappel positif test |
|---|---:|---:|---:|
| CNN + BiLSTM, score brut, seuil 0,5 | 82,45 % | 82,41 % | 79,10 % |
| CNN + BiLSTM, calibration et seuil actif | **82,15 %** | **82,10 %** | **88,95 %** |
| TF-IDF + régression logistique, calibrée | 81,39 % | 81,39 % | 84,20 % |

Le test contient 1 709 contrats ; la décision active produit 212 faux positifs et 93 faux négatifs. L'objectif de rappel de 90 % est choisi sur la calibration et n'est pas garanti sur le test. La calibration améliore légèrement le Brier (0,125856 → 0,125753) et l'ECE à 10 classes (5,65 % → 5,12 %) sur ce test ; elle ne rend pas les scores universellement fiables.

Les 1 152 contrats de la source réservée comprennent **1 146 positifs et seulement 6 négatifs** : le rappel positif est de 89,01 %, mais l'estimation des faux positifs est très fragile et l'ECE atteint 29,84 %. Ces données ne prouvent pas une généralisation suffisante à toutes les sources. Des négatifs vérifiés et comparables restent à recueillir.

Zéro recouvrement détecté entre les cinq partitions selon les six critères de l'audit. Cela ne prouve pas l'absence de tous les clones approximatifs. Les labels contradictoires sont exclus, pas corrigés arbitrairement. La classe 0 signifie une absence de signal selon les annotations apprises, **pas une certification de sécurité**.

L'ancien score V2 de 85,87 % utilisait un autre test avec recouvrements. Il ne constitue pas une comparaison directe avec V3.

## Comparaison expérimentale après V3

Le [protocole de comparaison](reports/protocole_comparaison_v1.md) couvre CNN,
BiLSTM, CNN-BiLSTM, CodeT5-small figé et TF-IDF, avec entrées complètes ou tronquées.
Les expériences sont isolées de l'application et n'activent aucun nouveau modèle.
Elles utilisent `config/comparison_v1.json` et un environnement dédié avec les
dépendances de `requirements-comparison.txt`, en complément de `requirements.txt`.
La commande `python -u -m src.run_comparison` enchaîne la progression ; l'état et
les résultats sont écrits sous `results/comparison/`. Le test V3 étant déjà
consulté, cette comparaison est exploratoire et demande une confirmation indépendante.

Comparaison terminée le 23 septembre 2026 : **26 entraînements et évaluations**
(18 réseaux depuis zéro, six têtes CodeT5 et deux références TF-IDF).
Voir la [conclusion finale de l'expérience](reports/comparaison_v1_etat.md), le
[rapport complet](results/comparison/comparison-v1-20260922/report.md) et le
[tableau des résultats](results/comparison/comparison-v1-20260922/summary.csv).

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
les réseaux nouvellement entraînés pour la comparaison du modèle V3 actif.

## Archives et provenance

Les scripts `*_v2.py` restent historiques. `build_dataset_v2.py` assemble les données et n'entraîne pas le réseau ; sa reconstruction nécessite les dépôts voisins et chemins CGT de l'expérience d'origine. V3 utilise les partitions V2 présentes dans ce dépôt et ne dépend plus de ces chemins externes pour reconstruire son benchmark.

Le sous-module `dataset/raw/smartbugs-curated` pointe vers [SmartBugs Curated](https://github.com/smartbugs/smartbugs-curated) à une révision fixée. Son README et sa licence restent dans le sous-module. Les références des exemples V3 permettent de retrouver leur source V2 ; les droits sur les sources Solidity restent ceux de leurs auteurs respectifs.
