# Sources du mémoire — sauvegarde partie 1

Les fichiers de ce dossier sauvegardent les sources et les scripts de l'édition Word/PDF conservée au 6 octobre 2026, ainsi que la préparation de sa révision TCN. Ils proviennent du dossier de travail local `.cache-comparison/memoire` ; leurs sources scientifiques et les résultats correspondants restent dans le dépôt.

L'édition publiée dans `output/` présente CNN–BiLSTM et XGBoost et conserve CodeBERT comme candidat non entraîné. **Elle n'est pas encore la révision finale avec les résultats TCN.** Le brouillon `drafts/tcn_content_pending.md` prépare cette révision, sans scores TCN. Ses rapports QA décrivent la préparation inspectée avant la mise en pause et ne valident pas un nouveau mémoire final.

## Fichiers conservés

- `content.md`, `build_memoire.py`, `make_figures.py` et `figures/` : texte, construction Word et figures de l'édition publiée.
- `comparison_snapshot.json`, `document_index.json`, `qa.json` et `artifact.md` : valeurs importées, pagination et vérifications de cette édition.
- `pages.json` : uniquement les deux textes de remerciements nécessaires au contrôle d'identité, aux positions 3 et 4 attendues par le vérificateur. Le reste du mémoire PDF de référence n'est pas reproduit ici.
- `update_tcn_models.py`, `check_tcn_memoire.py`, `render_tcn_revision.ps1` et `integration_readiness.json` : préparation, rendu et contrôles de la révision TCN.
- `drafts/` : brouillon TCN et rapports de vérification de sa préparation.

## Reprise ultérieure

L'entraînement TCN reste en pause. **Ne lancer aucun rendu final avant que les trois modèles soient évalués** et que `summary.complete` vaille `true`. Le script de rendu vérifie cette condition, ainsi que les métriques et les trois comparaisons appariées enregistrées.

Après la fin de l'expérience et l'export des graphiques, la commande prévue depuis la racine du projet est :

```powershell
pwsh -NoProfile -File docs/memoire/render_tcn_revision.ps1
```

Les chemins du Python documentaire, du renderer, de LibreOffice et de Poppler sont ceux de la machine d'Exauce et devront être adaptés sur une autre machine. Les fichiers générés, sauvegardes locales, environnements et dépendances installées ne sont pas publiés. Le préparateur reconstruira sa référence locale `before_tcn_models` à partir des sources de cette édition et des fichiers Word/PDF de partie 1 ; conserver ces références lors de la première intégration.

Le rendu et les contrôles produisent un staging. L'inspection de toutes les pages, tableaux, figures et remerciements précède la copie vers `output/`. Le script ne publie pas automatiquement un nouveau mémoire.
