# Sources brutes de reconstruction V3

Les fichiers `train.json`, `val.json` et `test.json` sont les entrées brutes historiques utilisées par le constructeur V3. Leurs noms décrivent les partitions d’entrée, et non les partitions finales : le constructeur les réunit, applique les exclusions et la déduplication, puis forme les cinq partitions V3 par groupes.

Ils ont été déplacés depuis `dataset/v2/raw/` lors du nettoyage final, sans modifier leurs octets. `manifest.json` associe leurs anciens chemins à leurs emplacements actuels et aux SHA-256 du manifeste V3 initial. `src/source_paths.py` assure la résolution des références figées.

Les partitions à utiliser pour l’apprentissage et l’évaluation sont dans `dataset/v3/`. Les droits sur le code source des contrats restent ceux de leurs auteurs respectifs.
