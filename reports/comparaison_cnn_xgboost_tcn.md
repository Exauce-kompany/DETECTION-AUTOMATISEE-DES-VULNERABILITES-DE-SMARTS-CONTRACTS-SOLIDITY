# Comparaison expérimentale : CNN–BiLSTM, XGBoost et TCN

Le **TCN** a été choisi le 5 octobre 2026 comme troisième modèle. Il remplace CodeBERT dans la comparaison principale et est entraîné depuis zéro sur les tokens Solidity normalisés du même dataset.

Le [protocole](protocole_cnn_xgboost_tcn.md) décrit les données, hyperparamètres, règles de sélection et limites. Le [rapport calculé](../results/benchmark/cnn-xgboost-tcn-20261005/report.md) fournit les valeurs enregistrées. Les résultats CNN–BiLSTM et XGBoost déjà obtenus sont importés avec leur provenance, sans recopier leurs poids.

Les scores TCN seront ajoutés après les trois entraînements, la sélection sur validation et la calibration. Pendant le pilote ou un calcul en cours, une absence de score ne représente pas une performance nulle. Le test déjà observé conserve son rôle exploratoire.

L'entraînement a été mis **en pause à la demande de l'utilisateur le 6 octobre 2026**. Les graines 42 et 73 sont terminées. La graine 101 dispose d'une sauvegarde vérifiée après deux époques complètes et 800 lots de la troisième époque, soit 3 200 mises à jour. Les poids, l'optimiseur et les états aléatoires sont conservés. Aucun score de test TCN ni classement final n'a encore été produit ; le mémoire final attend ces mesures.

Pour reprendre dans ce dossier, sans recommencer les deux graines terminées :

```powershell
.venv-comparison/Scripts/python.exe -u -m src.run_model_benchmark --stage tcn
```

La configuration et les sources figées doivent rester identiques jusqu'à la fin de cette expérience.
