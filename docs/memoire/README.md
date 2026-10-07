# Sources du mémoire — édition finale CNN–BiLSTM / XGBoost / TCN

Ce dossier contient les sources de l'édition finale Word/PDF du 6 octobre 2026. La comparaison principale présente les trois checkpoints réellement évalués : CNN–BiLSTM 82,10 %, XGBoost 89,41 % et TCN 84,77 % de F1 macro sur le même test. Le TCN retenu est la graine 42, époque 7, choisie sur la log-loss de validation. La comparaison demeure exploratoire ; les expériences CNN et XGBoost sont réutilisées avec leur provenance.

Les résultats proviennent de `results/benchmark/cnn-xgboost-tcn-20261005/summary.json`, complet. Les trois IC bootstrap utilisent les 1 000 tirages par groupes enregistrés, graine 42. `PR-AUC` désigne `average precision` (AP). Les durées murales peuvent comprendre attentes, suspensions et concurrence CPU ; la pause avec processus arrêté est exclue. Elles ne fondent aucun classement de vitesse.

## Sources et contrôles

- `content.md`, `build_memoire.py`, `make_figures.py`, `figures/` : texte, constructeur Word et 15 figures de l'édition finale.
- `comparison_snapshot.json`, `document_index.json`, `artifact.md` : métriques importées avec SHA-256, pagination et contrat de mise en page.
- `qa.json`, `visual_inspection_final.json` : contrôle de 44 pages, 8 tableaux, 14 références et 44 abréviations ; inspection complète des pages ; hashes des livrables et images inspectées.
- `pages.json` : les deux textes de remerciements de référence, aux positions 3 et 4 attendues par le contrôleur. Les autres pages du PDF modèle ne sont pas reproduites ici.
- `check_tcn_memoire.py` : contrôles numériques, pagination, références et identité textuelle/pixel des remerciements.
- `../../archives/2026-10-07/historique/docs/memoire/drafts/` : archives datées de partie 1 et de préparation TCN du 5–6 octobre 2026. Les anciens statuts de pause ou d'incomplétude concernent uniquement ces archives.

## Régénérer directement l'édition finale

Depuis la racine du dépôt, utiliser les sources finales sans réexécuter le préparateur de migration. Les constructeurs résolvent la racine du dépôt depuis leur emplacement et ne relancent aucun modèle. Ils vérifient les empreintes des résultats importés.

Exemple avec le runtime documentaire de cette machine :

```powershell
$artifactPython = 'C:/Users/Exauce/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$artifactRenderer = 'C:/Users/Exauce/.codex/plugins/cache/openai-primary-runtime/documents/26.813.12317/skills/documents/render_docx.py'
$env:PATH = 'C:/Program Files/LibreOffice/program;C:/Users/Exauce/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin;' + $env:PATH
& $artifactPython docs/memoire/make_figures.py
& $artifactPython docs/memoire/build_memoire.py
& $artifactPython $artifactRenderer output/documents/Memoire_SMART_BUG_Exauce_Kompani.docx --output_dir .cache-comparison/memoire/final_render --emit_pdf
```

Adapter ces chemins sur une autre machine. Dépendances documentaires : `python-docx`, `matplotlib`, `Pillow`, `pdf2image`, `pdfplumber`, `pypdf`, LibreOffice et Poppler. Elles sont installées dans le runtime documentaire, distinct de l'environnement ML. Les figures d'IC et de pertes sont reprises des exports complets du benchmark, après contrôle de l'empreinte de son résumé.

Le PDF rendu se trouve dans `.cache-comparison/memoire/final_render/`. Inspecter les 44 PNG courants avant de recopier ce PDF dans `output/pdf/`. Une reproduction doit refaire ses propres contrôles ; le QA publié décrit uniquement les hashes des fichiers de l'édition vérifiée.

Pour vérifier aussi l'identité des remerciements en pixels, conserver la référence locale originale, puis exécuter :

```powershell
& $artifactPython docs/memoire/check_tcn_memoire.py --stage docs/memoire --docx output/documents/Memoire_SMART_BUG_Exauce_Kompani.docx --pdf .cache-comparison/memoire/final_render/Memoire_SMART_BUG_Exauce_Kompani.pdf --baseline .cache-comparison/memoire/before_tcn_models --report .cache-comparison/memoire/reproduction_qa.json --require-complete
```

`--poppler` permet d'adapter le chemin du binaire. La référence contient le `content.md` et le PDF de partie 1, antérieurs à l'intégration TCN. Elle reste locale et ignorée ; sur un nouveau clone, restaurer cette référence originale à partir de la sauvegarde partie 1 (commit `008981d`) avant ce contrôle. Ne pas prendre les livrables finaux comme leur propre référence.

## Préparateur historique

Les scripts archivés `archives/2026-10-07/historique/docs/memoire/update_tcn_models.py` et `render_tcn_revision.ps1` documentent la migration à partir des anciens repères CodeBERT et de la référence locale `before_tcn_models`. Ils ne sont pas idempotents sur les sources finales. Le préparateur refuse de recréer sa référence depuis une édition déjà marquée finale. Leur staging reste une étape locale de migration ; la régénération normale de cette édition utilise directement `content.md`, `make_figures.py` et `build_memoire.py` ci-dessus.

Archivage du 7 octobre 2026 : les scripts de comparaison ancienne et de migration documentaire sont conservés dans `archives/2026-10-07/historique/`, sous leurs chemins d'origine. Les commandes historiques nécessitent une restauration dans une copie séparée ; voir le [guide de restauration](../../archives/2026-10-07/README.md).
