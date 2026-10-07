# SMART BUG — Détection de vulnérabilités Solidity

SMART BUG associe un modèle CNN + BiLSTM entraîné depuis zéro, des analyses heuristiques du code Solidity et une application FastAPI. Le modèle prédit un label binaire appris sur les annotations du corpus ; un résultat négatif ne certifie pas la sécurité d'un contrat.

## Démarrer l'application

Installer Git et [Git LFS](https://git-lfs.com/), puis exécuter :

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

Ouvrir <http://127.0.0.1:8000>. Pour un clone existant, récupérer les sources et poids avec `git submodule update --init --recursive`, puis `git lfs pull`. Le manifeste [models/active_model.json](models/active_model.json) désigne le modèle actif ; ses poids et son vocabulaire sont vérifiés avant chargement. `SMARTBUG_DATABASE_PATH` permet de choisir le fichier SQLite local de l'historique.

Pour analyser un fichier en ligne de commande :

```powershell
python -m src.inference.predict contracts/mon_contrat.sol
```

Remplacer le chemin d'exemple par celui du contrat à analyser.

## Se repérer

| Besoin | Emplacement |
|---|---|
| Comprendre les modules et les dépendances | [Architecture](docs/architecture.md) |
| Comprendre les défauts corrigés et les travaux consultés | [Audit de qualité](docs/audit_qualite.md) |
| Construire ou auditer les données | [src/data/](src/data/) |
| Modifier l'entraînement ou le réseau actif | [src/training/](src/training/) |
| Modifier les métriques et la calibration | [src/evaluation/](src/evaluation/) |
| Charger le modèle et prédire | [src/inference/](src/inference/) |
| Modifier les heuristiques de risque ou de performance | [src/analysis/](src/analysis/) |
| Comparer des modèles sans changer l'application | [src/experiments/](src/experiments/) |
| Générer les figures | [src/visualization/](src/visualization/) |
| Modifier l'interface, les routes ou l'historique | [webapp/](webapp/) |
| Reproduire les expériences et retrouver leurs résultats | [Guide des expériences](docs/experiments.md) |
| Exécuter les régressions | [tests/](tests/) : `python -m unittest discover -s tests -v` |

Les [configurations](config/), [données](dataset/), [modèles](models/) et [résultats](results/) restent séparés du code. Les commandes ci-dessus utilisent les nouveaux sous-packages ; la [table de migration](docs/architecture.md#migration-des-commandes) précise les anciens et nouveaux chemins.

Pour contrôler le code avant une modification, installer les outils avec `python -m pip install -r requirements-dev.txt`, puis exécuter `python -m ruff check src webapp tests` et `python -m ruff format --check src webapp tests`. La configuration exclut le prétraitement dont l'empreinte est figée par le dataset.

Les régressions du rapport imprimable sont séparées des tests Python : avec Node.js installé, exécuter `node tests/frontend/report_view.test.js`. Elles contrôlent le rendu des alertes, des scores et des états d'indisponibilité avec un DOM simulé.

## Rapports et mémoire

La [méthodologie SMART BUG](reports/methodologie.md), le [rapport final de comparaison](reports/rapport_final_partie_2.md) et le [rapport des calculs](results/benchmark/cnn-xgboost-tcn-20261005/report.md) décrivent les résultats et leurs limites. L'[état intermédiaire](reports/rapport_final_partie_1.md) et le [bilan du nettoyage précédent](reports/rapport_final.md) sont conservés comme documents historiques.

Le mémoire est disponible en [PDF](output/pdf/Memoire_SMART_BUG_Exauce_Kompani.pdf) et en [Word](output/documents/Memoire_SMART_BUG_Exauce_Kompani.docx). Ses [sources et contrôles](docs/memoire/README.md) permettent de le régénérer. La [synthèse antérieure](output/pdf/Synthese_memoire_SmartContractSecurity.pdf) reste disponible.
