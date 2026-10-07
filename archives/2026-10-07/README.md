# Archive du 7 octobre 2026

1 256 fichiers de l'inventaire ont été déplacés sans modification de contenu. Douze fichiers supplémentaires sont des copies de sauvegarde du code et des guides avant adaptation des dépendances. Volume total : 1 844 939 983 octets (environ 1,84 Go). Aucun fichier archivé n'a été supprimé définitivement.

## Organisation

- `historique/` : anciennes comparaisons, pilote CodeBERT, code et tests associés, rapports partiels, brouillons et ancienne synthèse Word/PDF.
- `local/` : caches, journaux et rendus intermédiaires. Ce dossier est ignoré par Git ; une publication du dépôt ne constitue pas une sauvegarde de ces fichiers locaux.
- `compatibilite/` : douze copies antérieures aux adaptations du projet actif.
- `manifest.json` : chemin d'origine, chemin archivé, catégorie, taille, SHA-256 et opération (`move` ou `copy`) pour chaque fichier.
- `completed.txt` : journal des 1 256 déplacements terminés.
- `operations/` : scripts ponctuels ayant effectué l'archivage ; conservés comme trace, à ne pas réexécuter.

Les noms et chemins mentionnés à l'intérieur des documents historiques décrivent leur état d'origine. Ils ne sont pas réécrits, afin de préserver leurs empreintes.

## Éléments conservés dans le projet actif

Les données, les poids CNN–BiLSTM/XGBoost/TCN, les résultats finaux, leurs instantanés de sources, les onze figures finales et le mémoire Word/PDF sont conservés à leurs emplacements. Le dossier XGBoost comportant `codebert` dans son nom reste indispensable : il contient les poids réellement utilisés. La provenance des baselines importées et la référence locale des remerciements restent également en place.

Le code partagé `pack_sequences` a été transféré vers `src/data/batching.py`. Les commandes et modules des anciennes comparaisons sont archivés ; le lanceur actif propose CNN–BiLSTM, XGBoost et TCN. Les résultats historiques restent figés : toute nouvelle campagne doit utiliser un nouvel identifiant.

L'archive porte sur les fichiers effectivement énumérés dans l'inventaire. Les dossiers alors inaccessibles, les environnements installés et les outils nécessaires aux vérifications ne sont pas assimilés à des fichiers supprimables. Des caches temporaires peuvent être recréés lors de nouvelles exécutions.

## Restauration

1. Créer une copie séparée du projet complet, avec son environnement et ses données. Ne pas restaurer par-dessus le projet final en cours.
2. Consulter `manifest.json`. Pour chaque fichier choisi, vérifier son SHA-256 puis copier `archived` depuis ce dossier vers `original` dans la copie du projet, en recréant les dossiers parents.
3. Pour retrouver les fonctionnalités historiques, restaurer ensemble `historique/` et les fichiers de `compatibilite/`. Les copies de compatibilité remplacent alors les fichiers correspondants uniquement dans cette copie séparée.
4. Restaurer `local/` seulement si les anciens caches ou rendus sont nécessaires. Les expériences scientifiques peuvent exiger leurs dépendances et configurations d'origine.

Le manifeste permet de restaurer un fichier isolé sans restaurer toute l'archive. Une nouvelle exécution ne doit pas écraser les résultats scientifiques figés.

## Vérifications après archivage

Les **77 tests actifs réussissent**. Les 15 tests spécifiques aux fonctionnalités historiques sont archivés avec leur code. Les **194 contrôles d'intégrité** des résultats finaux réussissent. Ruff valide le code actif et le formatage de 52 fichiers. Les différences de sources par rapport aux entraînements historiques restent explicitement signalées ; les instantanés originaux sont conservés.

Voir le [rapport d'intégrité](../../reports/integrite_apres_archivage_20261007.json) et le [compte rendu d'archivage](../../reports/archivage_20261007.json).
