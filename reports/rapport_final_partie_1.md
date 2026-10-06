# Rapport final — partie 1

État sauvegardé le **6 octobre 2026**. Sujet : détection automatisée des vulnérabilités des smart contracts Solidity par apprentissage profond.

Cette partie sauvegarde le travail disponible et la reprise de l'entraînement. La comparaison principale comprend **CNN–BiLSTM, TF-IDF + XGBoost et TCN**. Le TCN est en pause à la demande de l'utilisateur ; ses résultats de test restent à produire.

## Travail enregistré

- Code d'entraînement, de validation, de calibration et d'évaluation : `src/run_model_benchmark.py` et `src/benchmark_tcn.py`.
- Configuration et hyperparamètres : `config/benchmark_models.json` ; [protocole détaillé](protocole_cnn_xgboost_tcn.md).
- Résultats CNN–BiLSTM et XGBoost, provenance des calculs réutilisés et sources figées.
- Historiques et poids des deux entraînements TCN terminés, ainsi que la sauvegarde du troisième avec optimiseur et états aléatoires.
- Générateur de graphiques, schéma TCN en couleurs et [graphiques des deux modèles évalués](../results/benchmark/cnn-xgboost-tcn-20261005/plots/modeles_evalues/README.md) : matrices de confusion, métriques, ROC et précision–rappel. Les graphiques comparatifs finaux à trois modèles attendent l'évaluation du TCN.
- [Mémoire Word](../output/documents/Memoire_SMART_BUG_Exauce_Kompani.docx) et [PDF](../output/pdf/Memoire_SMART_BUG_Exauce_Kompani.pdf), conservés dans leur édition précédente. La révision intégrant les résultats TCN n'est pas encore publiée.
- [Sources et préparation du mémoire](../docs/memoire/README.md), avec les scripts, figures et contrôles nécessaires à la reprise de la rédaction.
- Tests du protocole et du moteur TCN. Les environnements Python, dépendances locales et journaux temporaires restent hors du dépôt.

Le modèle actif SMART BUG et son dataset sont conservés. Cette comparaison n'active pas automatiquement un nouveau modèle dans l'application.

## Résultats disponibles

Les deux modèles évalués utilisent les **mêmes 1 709 contrats de test**, après sélection sur validation et ajustement des seuils sur calibration. Ces scores proviennent des fichiers enregistrés de l'expérience, sans comparaison avec des scores publiés sur d'autres corpus.

| Modèle | F1 macro | Précision positive | Rappel positif | Taux de faux positifs | État |
|---|---:|---:|---:|---:|---|
| CNN–BiLSTM actif | 82,10 % | 77,94 % | 88,95 % | 24,45 % | Évalué |
| TF-IDF + XGBoost | 89,41 % | 89,11 % | 89,43 % | 10,61 % | Évalué |
| TCN | — | — | — | — | Entraînement en pause |

XGBoost présente le meilleur F1 parmi les **deux modèles déjà évalués**. Aucun classement des trois modèles n'est encore possible. Le [rapport des calculs](../results/benchmark/cnn-xgboost-tcn-20261005/report.md) et le [suivi de la comparaison](comparaison_cnn_xgboost_tcn.md) décrivent l'état courant.

## Point de reprise TCN

Les graines **42 et 73** ont terminé leurs entraînements. La graine **101** a terminé deux époques et **800 lots sur 1 200 de la troisième époque**, soit **3 200 mises à jour** de l'optimiseur. Ce point a été relu et vérifié après l'arrêt du processus.

Fichier publié via Git LFS :

`models/benchmark/cnn-xgboost-tcn-20261005/tcn/seed-101/resume.pt`

SHA-256 : `78d0ab4c3d7728119b3ddd23723a3ff4b4d785a00b7c379e1eb8457619248660`.

Cette sauvegarde contient les poids courants, les meilleurs poids de validation, l'optimiseur, les états aléatoires, l'historique et la position dans l'époque. Les données, hyperparamètres et sources figées doivent rester identiques pour la reprise. Les éventuels lots exécutés après la dernière sauvegarde seront rejoués.

**Ne lancer la commande suivante qu'au moment de reprendre l'entraînement**, depuis la racine du projet et dans l'environnement scientifique configuré :

```powershell
git lfs pull
.venv-comparison/Scripts/python.exe -u -m src.run_model_benchmark --stage tcn
```

La commande conserve les deux graines terminées et reprend la troisième. Les interventions CPU et la pause sont consignées dans [runtime_notes.json](../results/benchmark/cnn-xgboost-tcn-20261005/runtime_notes.json).

## Vérifications et limites

Les 19 tests ciblés du moteur TCN et du protocole de comparaison ont réussi avant la mise en pause. L'audit indépendant a confirmé les empreintes des 15 fichiers du snapshot, des sept sources figées, de la configuration et du modèle CNN actif. La sauvegarde de reprise conserve des poids finis, l'optimiseur et les états aléatoires.

Le test a déjà été consulté : les conclusions sont exploratoires et demandent une confirmation indépendante. Les annotations sont en partie issues d'analyseurs statiques ; la classe négative ne certifie pas la sécurité. Les durées incluent des interruptions et la concurrence d'un autre entraînement et ne permettent pas un classement équitable de vitesse.

## Suite du travail — partie 2

1. Reprendre et terminer la graine TCN 101.
2. Sélectionner le checkpoint sur validation, puis calibrer et évaluer le TCN.
3. Calculer les différences appariées entre les trois modèles et exporter les graphiques finaux.
4. Intégrer les mesures dans le mémoire, vérifier les tableaux, figures, références et remerciements, puis publier la partie 2.
