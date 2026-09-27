# Sources brutes de reconstruction SMART BUG

Les fichiers `train.json`, `val.json` et `test.json` sont les entrées brutes historiques utilisées par le constructeur SMART BUG. Leurs noms décrivent les partitions d’entrée, et non les partitions finales : le constructeur les réunit, applique les exclusions et la déduplication, puis forme les cinq partitions SMART BUG par groupes.

Ils ont été déplacés depuis `dataset/v2/raw/` lors du nettoyage final, sans modifier leurs octets. `manifest.json` associe leurs anciens chemins à leurs emplacements actuels et aux SHA-256 du manifeste SMART BUG initial. `src/source_paths.py` assure la résolution des références figées.

Les partitions à utiliser pour l’apprentissage et l’évaluation sont dans `dataset/benchmark/`. Les droits sur le code source des contrats restent ceux de leurs auteurs respectifs.
