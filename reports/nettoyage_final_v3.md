# Dossier final V3 — nettoyage du 27 septembre 2026

Le dossier de travail contient désormais le modèle actif V3, son jeu de données, ses résultats et la comparaison expérimentale CNN/BiLSTM/CNN-BiLSTM/CodeT5/TF-IDF réalisée sur les partitions V3.

## Suppressions

79 fichiers historiques ont été retirés, soit environ 118,6 Mo dans le dossier de travail : anciens modèles Keras V1/V2, données préparées, métriques, prédictions, graphiques, audits spécifiques et 18 scripts obsolètes. Les dossiers devenus vides ont été retirés. Les quatre tests exclusivement consacrés aux scripts historiques ont été supprimés avec ces scripts.

L’historique Git et ses objets LFS sont conservés. Ce nettoyage ne réécrit pas les anciens commits et ne supprime pas leurs copies historiques. Les documents Word/PDF de synthèse restent dans `output/`.

## Sources brutes nécessaires à V3

Les trois fichiers JSON d’origine ont été déplacés, sans modification de contenu :

| Ancienne référence | Emplacement actuel |
|---|---|
| `dataset/v2/raw/train.json` | `dataset/sources/train.json` |
| `dataset/v2/raw/val.json` | `dataset/sources/val.json` |
| `dataset/v2/raw/test.json` | `dataset/sources/test.json` |

Ces fichiers servent à reconstruire les partitions V3 et à vérifier une prédiction à partir du code original. Ils ne représentent pas des résultats V2 à utiliser dans le mémoire. Leurs SHA-256 correspondent aux empreintes du manifeste V3 initial. `dataset/sources/manifest.json` conserve cette correspondance ; les règles Git LFS suivent les nouveaux chemins.

Les configurations et manifestes expérimentaux figés conservent leurs références d’origine. Le constructeur V3 utilise désormais `src/source_paths.py` pour résoudre les chemins déplacés et vérifier leur intégrité. Les commandes habituelles continuent de fonctionner :

```powershell
python -m src.audit_dataset_v3
python -m src.build_dataset_v3 --output dataset/v3_rebuilt
```

L’adaptation du constructeur modifie son empreinte par rapport à celle enregistrée en septembre 2026. Les autres fichiers sources enregistrés dans l’expérience V3 restent identiques. Une reconstruction produit un nouveau manifeste ; le snapshot distribué n’a pas été reconstruit ni modifié pendant ce nettoyage.

Le sous-module SmartBugs Curated, sa révision et sa licence sont conservés comme provenance tierce. Les fichiers de comparaison contenant `v1` dans leur nom appartiennent à la première version du protocole comparatif **sur V3** ; ils ne sont pas les anciens résultats du modèle V1. Les versions des moteurs heuristiques et des réglages de l’application sont également indépendantes de la version du modèle neuronal.

## Graphiques et application

`src/plot_results_v3.py` exporte cinq figures en PNG et SVG dans `results/v3/v3-403bb881-20260922T101754Z/plots/` :

- exactitude d’entraînement et de validation de la graine sélectionnée ;
- perte d’entraînement et de validation, avec indication du checkpoint retenu ;
- matrice de confusion sur le test, au seuil actif ;
- métriques globales du réseau brut, du réseau calibré et de la référence TF-IDF ;
- précision, rappel et F1 par classe du modèle actif.

Un manifeste enregistre les sources et empreintes des graphiques. Les figures ont été inspectées visuellement. Le graphique comparatif des nouveaux modèles PyTorch reste dans `results/comparison/comparison-v1-20260922/`.

Le rapport heuristique de SMART BUG indiquait encore « modèle IA V2 » : cette mention a été corrigée pour décrire le CNN-BiLSTM V3 et le sens de sa classification binaire. La page du modèle et la documentation présentent désormais le dossier final.

## Vérifications effectuées

- **32 tests réussis**, dont les tests d’intégration HTTP, la concordance d’une prédiction enregistrée avec le modèle actif et deux tests de résolution des sources déplacées.
- Audit indépendant réussi sur **18 238 exemples**, avec encodages et labels concordants, et zéro recouvrement selon les six critères contrôlés.
- **297 fichiers conservés** du dataset, des modèles et des résultats V3/comparatifs : empreintes identiques avant et après nettoyage, dont `models/active_model.json`.
- Empreintes des trois sources brutes déplacées vérifiées.
- Syntaxe de **31 fichiers Python et 8 fichiers JavaScript** vérifiée.
- Aucun réentraînement du modèle actif ni changement de ses poids, de son vocabulaire, de sa calibration ou de ses métriques.

La validation des données ne prouve pas l’absence de tous les clones approximatifs. Les limites scientifiques des annotations et du test exploratoire restent celles des rapports V3 et comparatifs.
