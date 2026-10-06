# Dossier final SMART BUG — nettoyage du 27 septembre 2026

Mise à jour du 5 octobre 2026 : la comparaison principale du mémoire porte désormais sur CNN–BiLSTM, XGBoost et TCN, selon le [nouveau protocole](protocole_cnn_xgboost_tcn.md). Le présent rapport conserve le compte rendu du nettoyage de septembre ; les [nouveaux calculs](../results/benchmark/cnn-xgboost-tcn-20261005/report.md) ont leur propre état d'avancement. Le pilote CodeBERT est archivé après le choix du TCN.

Le dossier de travail contient désormais le modèle actif SMART BUG, son jeu de données, ses résultats et la comparaison expérimentale CNN/BiLSTM/CNN-BiLSTM/CodeT5/TF-IDF réalisée sur les partitions SMART BUG.

## Suppressions

79 fichiers historiques ont été retirés, soit environ 118,6 Mo dans le dossier de travail : anciens modèles Keras V1/V2, données préparées, métriques, prédictions, graphiques, audits spécifiques et 18 scripts obsolètes. Les dossiers devenus vides ont été retirés. Les quatre tests exclusivement consacrés aux scripts historiques ont été supprimés avec ces scripts.

L’historique Git et ses objets LFS sont conservés. Ce nettoyage ne réécrit pas les anciens commits et ne supprime pas leurs copies historiques. Les documents Word/PDF de synthèse restent dans `output/`.

## Sources brutes nécessaires à SMART BUG

Les trois fichiers JSON d’origine ont été déplacés, sans modification de contenu :

| Ancienne référence | Emplacement actuel |
|---|---|
| `dataset/v2/raw/train.json` | `dataset/sources/train.json` |
| `dataset/v2/raw/val.json` | `dataset/sources/val.json` |
| `dataset/v2/raw/test.json` | `dataset/sources/test.json` |

Ces fichiers servent à reconstruire les partitions SMART BUG et à vérifier une prédiction à partir du code original. Ils ne représentent pas des résultats V2 à utiliser dans le mémoire. Leurs SHA-256 correspondent aux empreintes du manifeste SMART BUG initial. `dataset/sources/manifest.json` conserve cette correspondance ; les règles Git LFS suivent les nouveaux chemins.

Les références historiques aux sources brutes sont conservées ; les chemins et appellations du modèle ont été actualisés dans les manifestes. Le constructeur SMART BUG utilise désormais `src/source_paths.py` pour résoudre les chemins déplacés et vérifier leur intégrité. Les commandes habituelles continuent de fonctionner :

```powershell
python -m src.audit_dataset
python -m src.build_dataset --output dataset/benchmark_rebuilt
```

Le constructeur a été adapté au déplacement des sources. Le renommage du projet a ensuite modifié les chemins, les imports et les empreintes des modules. Les partitions, leurs encodages et leur vocabulaire conservent leur contenu original ; les métadonnées de provenance ont été migrées explicitement.

Le sous-module SmartBugs Curated, sa révision et sa licence sont conservés comme provenance tierce. Les fichiers de comparaison contenant `v1` dans leur nom appartiennent à la première version du protocole comparatif **sur SMART BUG** ; ils ne sont pas les anciens résultats du modèle V1. Les versions des moteurs heuristiques et des réglages de l’application sont également indépendantes de la version du modèle neuronal.

## Graphiques et application

`src/plot_results.py` exporte cinq figures en PNG et SVG dans `results/training/smartbug-403bb881-20260922T101754Z/plots/` :

- exactitude d’entraînement et de validation de la graine sélectionnée ;
- perte d’entraînement et de validation, avec indication du checkpoint retenu ;
- matrice de confusion sur le test, au seuil actif ;
- métriques globales du réseau brut, du réseau calibré et de la référence TF-IDF ;
- précision, rappel et F1 par classe du modèle actif.

Un manifeste enregistre les sources et empreintes des graphiques. Les figures ont été inspectées visuellement. Le graphique comparatif des nouveaux modèles PyTorch reste dans `results/comparison/comparison-v1-20260922/`.

Le rapport heuristique de SMART BUG indiquait encore « modèle IA V2 » : cette mention a été corrigée pour décrire le CNN-BiLSTM SMART BUG et le sens de sa classification binaire. La page du modèle et la documentation présentent désormais le dossier final.

## Vérifications du nettoyage, avant renommage

- **32 tests réussis**, dont les tests d’intégration HTTP, la concordance d’une prédiction enregistrée avec le modèle actif et deux tests de résolution des sources déplacées.
- Audit indépendant réussi sur **18 238 exemples**, avec encodages et labels concordants, et zéro recouvrement selon les six critères contrôlés.
- **297 fichiers conservés** du dataset, des modèles et des résultats SMART BUG/comparatifs : empreintes identiques avant et après nettoyage, dont `models/active_model.json`.
- Empreintes des trois sources brutes déplacées vérifiées.
- Syntaxe de **31 fichiers Python et 8 fichiers JavaScript** vérifiée.
- Aucun réentraînement du modèle actif ni changement de ses poids, de son vocabulaire, de sa calibration ou de ses métriques.

La validation des données ne prouve pas l’absence de tous les clones approximatifs. Les limites scientifiques des annotations et du test exploratoire restent celles des rapports SMART BUG et comparatifs.

## Appellation finale SMART BUG

Les fichiers du modèle n’utilisent plus de suffixe de version : `src/train.py`, `src/model.py`, `src/predictor.py`, `src/preprocessing.py`, `src/build_dataset.py`, `src/audit_dataset.py` et `src/plot_results.py`. La configuration est `config/model.json`. Les données préparées se trouvent dans `dataset/benchmark/`, les modèles dans `models/trained/` et les résultats dans `results/training/`.

L’interface, les rapports, les six figures (dont la comparaison) et la synthèse Word/PDF emploient SMART BUG. Les trois archives Keras ont reçu des noms et références de modules actualisés ; leurs charges utiles de poids sont identiques octet par octet. Les modèles PyTorch, les scalers, les entrées numériques et les prédictions sont inchangés. Aucun entraînement supplémentaire n’a été effectué.

Le contenu descriptif des manifestes ayant changé, leurs empreintes et identifiants ont été recalculés. `metadata_migration` identifie cette opération de nommage et le commit de référence `0874cd5d37c0fc7d0305364e7715613a2767186b`. Les empreintes des sources d’entraînement d’origine restent consultables dans l’historique et dans `original_source_sha256`. Les dates d’entraînement et d’évaluation ne sont pas réécrites. Les anciens commits constituent la provenance historique ; ils ne sont pas modifiés.

Après renommage : **32 tests réussis**, audit des **18 238 exemples** réussi, et concordance de la prédiction HTTP avec le résultat enregistré. Les empreintes des données et poids numériques ont été contrôlées, les imports Python et scripts JavaScript vérifiés, et les graphiques et les huit pages de synthèse inspectés.
