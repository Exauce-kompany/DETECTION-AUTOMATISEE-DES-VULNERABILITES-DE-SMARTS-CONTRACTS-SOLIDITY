# DETECTION-AUTOMATISEE-DES-VULNERABILITES-DE-SMARTS-CONTRACTS-SOLIDITY

SMART BUG est un projet de détection automatisée de vulnérabilités dans du code
Solidity. Il associe un classifieur neuronal binaire BiLSTM, une analyse statique
heuristique et une interface web FastAPI.

Le modèle V2 est entraîné à partir de zéro avec TensorFlow/Keras. Ses deux
classes sont `non_vulnerable` et `vulnerable`. Les catégories et lignes de
vulnérabilités affichées par l'interface proviennent du moteur heuristique
distinct, pas de la sortie binaire du réseau.

## Récupérer le projet

Installer Git et [Git LFS](https://git-lfs.com/) avant le clonage. Les modèles,
vocabulaires, tableaux NumPy et partitions brutes V2 sont stockés avec Git LFS.
Le jeu de données SmartBugs Curated est référencé comme sous-module Git.

```powershell
git lfs install
git clone --recurse-submodules https://github.com/Exauce-kompany/DETECTION-AUTOMATISEE-DES-VULNERABILITES-DE-SMARTS-CONTRACTS-SOLIDITY.git
cd DETECTION-AUTOMATISEE-DES-VULNERABILITES-DE-SMARTS-CONTRACTS-SOLIDITY
git lfs pull
```

Pour compléter un clonage déjà effectué :

```powershell
git submodule update --init --recursive
git lfs pull
```

## Lancer l'application

L'environnement utilisé par le projet repose sur Python 3.12 et TensorFlow
2.21.0. Depuis la racine du dépôt :

```powershell
conda create -n smartcontract python=3.12
conda activate smartcontract
python -m pip install -r requirements.txt
python -m pip install Jinja2==3.1.6 python-multipart==0.0.32
python -m uvicorn webapp.app:app --host 127.0.0.1 --port 8000
```

Les deux dépendances complémentaires servent aux templates HTML et aux envois
de fichiers. Ouvrir ensuite <http://127.0.0.1:8000>.

La base SQLite de l'historique des analyses est créée au démarrage. Elle reste
locale et n'est pas versionnée. `environment.yml` conserve également un export
de l'environnement Windows d'origine ; son champ `prefix` correspond au poste
sur lequel cet export a été réalisé.

## Organisation

| Chemin | Contenu |
|---|---|
| `src/build_dataset_v2.py` | Assemblage du dataset V2 à partir de V1 et de CGT |
| `src/prepare_data_v2.py` | Tokenisation, vocabulaire et encodage |
| `src/train_v2.py` | Entraînement et validation du modèle V2 |
| `src/evaluate_v2.py` | Évaluation détaillée sur le test |
| `src/predict_v2.py` | Prédiction en ligne de commande |
| `src/plot_results_v2.py` | Visualisation des résultats |
| `webapp/` | Application FastAPI, interface et moteurs d'analyse |
| `models/` | Modèles Keras V1 et V2 |
| `dataset/v2/` | Partitions brutes et données préparées V2 |
| `dataset/processed/prepared/` | Données préparées V1 pour comparaison |
| `results/` | Résultats d'évaluation, historiques et audits |
| `reports/` | Rapport d'audit du modèle et mesures associées |

Les scripts sans suffixe V2 permettent de reproduire la référence V1. Les
fichiers V1 et V2 constituent des expériences distinctes.

## Évaluation et entraînement

Les données préparées et les modèles sauvegardés permettent d'évaluer V2 :

```powershell
python src/evaluate_v2.py
```

Pour reconstruire les entrées à partir des partitions brutes, puis entraîner
le réseau :

```powershell
python src/prepare_data_v2.py
python src/train_v2.py
python src/evaluate_v2.py
python src/plot_results_v2.py
```

Ces commandes écrivent dans les dossiers de données, modèles et résultats V2.
L'assemblage initial via `build_dataset_v2.py` nécessite aussi les dépôts voisins
`smart-contract-vuln-dataset` et `smart-contract-cgt`, les partitions V1 attendues
et les chemins sources CGT indiqués dans les CSV d'audit. Ces chemins doivent
être adaptés sur une autre machine. Cette étape n'est pas nécessaire pour
utiliser le modèle et les données V2 déjà fournis.

## Résultats et limites

Le modèle V2 comporte 346 978 paramètres, un vocabulaire de 5 000 tokens et une
entrée limitée à 512 tokens. Le test enregistré contient 2 363 exemples et donne
85,87 % d'accuracy, avec 127 faux positifs et 207 faux négatifs.

L'audit a reproduit ces prédictions, mais identifié des limites importantes :
recouvrement de représentations numériques entre entraînement et test,
annotations contradictoires, troncature du code et commentaires révélant des
annotations de vulnérabilité. Ces résultats ne constituent donc pas une
garantie de sécurité sur de nouveaux contrats.

Consulter le [rapport d'audit](reports/audit_modele_v2.md) et les mesures JSON
associées avant d'interpréter les performances.

## Provenance des données

Le sous-module `dataset/raw/smartbugs-curated` pointe vers
[SmartBugs Curated](https://github.com/smartbugs/smartbugs-curated) à une révision
fixée. Son README et sa licence restent disponibles dans ce sous-module.
Les autres partitions conservent leurs métadonnées de provenance ; les droits
applicables aux sources Solidity restent ceux de leurs auteurs respectifs.
