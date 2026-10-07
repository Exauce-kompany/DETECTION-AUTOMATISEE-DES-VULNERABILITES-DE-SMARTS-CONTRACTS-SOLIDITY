# Architecture du projet

Le dépôt réunit trois usages : reconstruire un corpus traçable, entraîner et comparer des modèles, puis servir le modèle sélectionné dans une application web. L'organisation sépare ces responsabilités dans une seule application Python. Les expériences ne deviennent pas des dépendances de l'interface web.

## Raisons du découpage

Parnas recommande de choisir les modules selon les décisions de conception qu'ils doivent isoler. Ici, le stockage SQLite, le chargement des poids, les règles heuristiques et les protocoles expérimentaux ont des raisons différentes d'évoluer : ils ont donc des frontières distinctes. Ce principe motive le découpage ; il ne prescrit pas les noms exacts des dossiers. [Parnas, *On the Criteria To Be Used in Decomposing Systems into Modules*, 1972](https://www.cs.lafayette.edu/~gexia/cs301/resources/parnas.html).

La démarche de DrivenData distingue données sources, transformations reproductibles et produits d'analyse. Nous l'adaptons aux répertoires déjà référencés par les manifestes : `dataset/`, `models/` et `results/` conservent leurs chemins. Déplacer ces artefacts uniquement pour copier un modèle d'arborescence compliquerait leur traçabilité. [Cookiecutter Data Science, *Opinions*](https://cookiecutter-data-science.drivendata.org/opinions/).

La documentation FastAPI propose des `APIRouter` pour regrouper les routes de fonctions distinctes. L'application assemble donc des routeurs dédiés à l'analyse, l'historique et les rapports. La préparation des résultats et l'accès à SQLite ont leurs propres modules. [FastAPI, *Bigger Applications — Multiple Files*](https://fastapi.tiangolo.com/tutorial/bigger-applications/).

Ces choix correspondent au contenu du dépôt : ils ne constituent pas une preuve qu'une architecture serait la meilleure pour tous les projets d'apprentissage automatique.

## Arborescence utile

```text
SmartContractSecurity/
├── src/                         # Package Python scientifique
│   ├── data/                    # Construction, audit, artefacts et lots NumPy
│   ├── training/                # Réseau et jeux de données TensorFlow, entraînement
│   ├── evaluation/              # Calibration et métriques partagées
│   ├── inference/               # Chargement vérifié des poids et commande de prédiction
│   ├── analysis/                # Heuristiques Solidity de risque et performance
│   ├── experiments/
│   │   ├── comparison/          # Campagne CNN/BiLSTM/CodeT5/TF-IDF
│   │   └── benchmark/           # Campagnes XGBoost/TCN et pilote CodeBERT
│   ├── visualization/           # Figures produites depuis les résultats enregistrés
│   ├── preprocessing.py         # Contrat lexical partagé et versionné par empreinte
│   └── provenance.py            # Résolution des noms de sources des expériences
├── webapp/
│   ├── app.py                   # Fabrique de l'application et cycle de vie
│   ├── routes/                  # HTTP : analysis, history, reports
│   ├── services/analysis.py     # Orchestration et résultat d'une analyse
│   ├── persistence/database.py  # Connexions et requêtes SQLite
│   ├── predictor.py             # Instance partagée du prédicteur scientifique
│   ├── templates/              # Vues Jinja
│   └── static/                 # Styles et ressources du navigateur
├── tests/                       # Régressions fonctionnelles
├── config/                      # Paramètres et protocoles enregistrés
├── contracts/                   # Contrats d'exemple
├── dataset/                     # Sources, partitions et audits
├── models/                      # Manifestes et poids
├── results/                     # Sorties des expériences et figures
├── reports/                     # Rapports scientifiques historiques
├── docs/                        # Architecture, audit, reproduction et sources du mémoire
├── output/                      # Livrables PDF et Word
└── notebooks/                   # Exploration
```

`src` reste le nom du package importé. Le dépôt n'introduit pas une nouvelle distribution Python ni un second niveau `src/smartbug/`.

## Frontières des modules

| Module | Responsabilité | Règle de modification |
|---|---|---|
| `data` | Transformer les sources, charger les artefacts, préparer les lots NumPy | Produire un nouveau dossier pour une reconstruction |
| `training` | Définir et entraîner le modèle actif, adapter les lots à TensorFlow | Les choix sont fixés avant l'évaluation de test |
| `evaluation` | Calculer les métriques et calibrer les prédictions | Les expériences réutilisent ces calculs |
| `inference` | Charger le modèle publié et prédire sur toutes les fenêtres | Partagé par la CLI et le web |
| `analysis` | Produire des indications heuristiques sur Solidity | Ne pas les présenter comme des probabilités du réseau |
| `experiments` | Exécuter des protocoles comparatifs séparés | Ne pas activer implicitement un nouveau modèle web |
| `visualization` | Lire les résultats et produire les figures | Aucun réentraînement nécessaire |
| `webapp.routes` | Valider les entrées HTTP et choisir les réponses | Déléguer calcul et persistance |
| `webapp.services` | Coordonner les analyses et préparer leurs résultats | Ne pas ouvrir de connexion SQLite |
| `webapp.persistence` | Encapsuler les opérations de stockage | Fermer la connexion même en cas d'échec |

L'ancien module `experiment.py` mélangeait les lectures de fichiers, les lots, TensorFlow et les métriques. Ces responsabilités sont désormais portées par `data/artifacts.py`, `data/batching.py`, `training/datasets.py` et `evaluation/metrics.py`. Les expériences PyTorch peuvent ainsi utiliser les données et les métriques sans importer le réseau TensorFlow.

Une fonctionnalité nouvelle rejoint le domaine dont elle dépend. Un dossier générique `utils/` n'est pas nécessaire tant que l'on peut nommer clairement la responsabilité partagée.

## Provenance et compatibilité

`src/preprocessing.py` conserve son emplacement et ses octets : les manifestes du dataset en enregistrent l'empreinte. Le déplacer ou le reformater sans migrer le protocole rendrait cette vérification incohérente.

`src/provenance.py` résout les noms des fichiers scientifiques vers leurs nouveaux chemins pour le calcul des empreintes. Les manifestes et instantanés de sources déjà publiés restent des preuves de l'exécution d'origine. Un changement de code reste un changement de provenance : cette table n'autorise pas à reprendre une expérience avec des empreintes différentes. Les anciennes campagnes de comparaison et de benchmark sont donc conservées ; une nouvelle campagne exige un identifiant inédit passé avec `--run-id`. Utiliser `--config` avec une copie de la configuration pour modifier ses paramètres.

Les données, configurations expérimentales, poids, métriques enregistrées et documents historiques ne sont pas réécrits par la réorganisation. Un chemin ancien dans un rapport d'expérience décrit son état historique ; les commandes actuelles sont dans le [guide des expériences](experiments.md).

## Migration des commandes

Exécuter les modules depuis la racine du dépôt. Les anciens points d'entrée `src.<module>` sont remplacés par les chemins ci-dessous ; il n'y a pas de doublons maintenus dans l'ancien dossier.

| Ancienne commande | Commande actuelle |
|---|---|
| `python -m src.audit_dataset` | `python -m src.data.audit_dataset` |
| `python -m src.build_dataset` | `python -m src.data.build_dataset` |
| `python -m src.train` | `python -m src.training.train` |
| `python -m src.predict` | `python -m src.inference.predict` |
| `python -m src.run_comparison` | `python -m src.experiments.comparison.run_comparison` |
| `python -m src.run_model_benchmark` | `python -m src.experiments.benchmark.run_model_benchmark` |
| `python -m src.plot_results` | `python -m src.visualization.plot_results` |
| `python -m src.plot_comparison` | `python -m src.visualization.plot_comparison` |
| `python -m src.plot_model_benchmark` | `python -m src.visualization.plot_model_benchmark` |

Le lancement web demeure `python -m uvicorn webapp.app:app`. Les sous-packages possèdent leurs `__init__.py` ; les imports internes suivent les mêmes frontières que les commandes.
