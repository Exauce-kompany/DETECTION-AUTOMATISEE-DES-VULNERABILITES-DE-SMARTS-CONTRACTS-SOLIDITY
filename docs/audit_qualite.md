# Audit de qualité et réorganisation

Revue effectuée le 6 octobre 2026, complétée par les corrections du 7 octobre 2026. Son objectif est de corriger des défauts observables et de rendre les responsabilités explicites. L'apparence d'un fichier ne permet pas d'attribuer son écriture à une IA : aucune origine n'est déduite des constats ci-dessous.

## Travaux consultés

Liu et al. étudient 4 066 programmes Java et Python produits avec ChatGPT pour des exercices de programmation. Ils distinguent erreurs d'exécution, résultats incorrects, problèmes de style et de maintenabilité, et problèmes d'efficacité. Dans ce corpus, un programme qui passe les tests peut encore avoir des défauts de maintenabilité. Ces catégories donnent une grille de revue utile ; leurs taux ne sont pas des estimations applicables à SMART BUG. [Liu et al., *Refining ChatGPT-Generated Code: Characterizing and Mitigating Code Quality Issues*, version 2023](https://arxiv.org/html/2307.12596v2).

Molison et al. comparent des solutions Python humaines et générées avec des mesures SonarQube. Leurs résultats sont nuancés : les solutions générées présentent moins de défauts dans certaines mesures, mais des problèmes structurels apparaissent pour des tâches complexes. Il serait donc injustifié d'assimiler systématiquement « code généré » et « mauvais code ». L'évaluation dépend des tâches, des modèles et des mesures retenues. [Molison et al., *Is LLM-Generated Code More Maintainable & Reliable than Human-Written Code?*, ESEM 2025](https://arxiv.org/abs/2508.00700).

L'application de cette grille au dépôt est notre analyse, pas une conclusion de ces publications sur ce projet. Les corrections suivent des exemples reproductibles, les dépendances du code et les contrats des fonctions. L'organisation s'appuie séparément sur les principes et documentations cités dans [l'architecture](architecture.md).

## Constats locaux et corrections

| Catégorie de revue | Constat dans l'état initial | Correction et emplacement |
|---|---|---|
| Organisation et maintenabilité | `webapp/app.py` réunissait 1 222 lignes de routes, interprétation des résultats, rendu et accès à l'historique | Point d'assemblage ramené à 43 lignes ; routeurs dans `webapp/routes/`, analyse dans `webapp/services/analysis.py`, SQLite dans `webapp/persistence/database.py` |
| Organisation et maintenabilité | Scripts de données, entraînement, comparaisons et graphiques mélangés directement sous `src/` | Sous-packages par responsabilité, imports et commandes actualisés ; [table de migration](architecture.md#migration-des-commandes) |
| Couplage inutile | `experiment.py` associait chargement des artefacts, lots NumPy, TensorFlow et métriques | Répartition dans `data/artifacts.py`, `data/batching.py`, `training/datasets.py`, `evaluation/metrics.py` |
| Résultats incorrects | Une conversion de score tentait plusieurs formats, masquait des erreurs puis renvoyait zéro | Validation du contrat réellement fourni par le prédicteur, sans transformer une sortie invalide en résultat rassurant |
| Résultats incorrects aux limites | Un score déjà exprimé en pourcentage pouvait être multiplié par 100 lorsqu'il était inférieur ou égal à 1 : `0,5 %` devenait `50 %` | Unité définie par le nom du champ et conservée lors de la construction du résultat |
| Efficacité et limites de ressources | Le fichier envoyé était lu intégralement avant le contrôle de la limite de 2 Mio | Lecture bornée avant décodage et analyse dans la route d'analyse |
| Exécution et gestion des ressources | Les connexions SQLite étaient fermées seulement après une exécution réussie | Gestion de la durée de vie des connexions avec fermeture sur succès ou exception |
| Efficacité | Des gestionnaires `async` appelaient directement des calculs TensorFlow et des opérations SQLite synchrones | Calcul synchrone délégué hors de la boucle d'événements ; routes adaptées aux opérations synchrones |
| Résultats incorrects | Les traitements de commentaires pouvaient confondre leurs délimiteurs avec ceux présents dans des chaînes Solidity | Traitement lexical partagé pour les heuristiques ; conservation du prétraitement associé aux empreintes du dataset |
| Résultats incorrects | Les comparaisons `==` étaient comptées comme des écritures potentielles de stockage | Correction du motif de reconnaissance et régression dans `tests/test_analysis.py` |
| États d'erreur trompeurs | Un analyseur indisponible pouvait être affiché avec un score nul et aucun motif détecté | Scores absents représentés par `None`, affichage « Indisponible » ou « — » ; les vrais scores nuls restent affichés |
| Code superflu et lisibilité | Imports inutilisés, lecture inutile des enregistrements de validation, réutilisation du nom du module `model` pour une instance | Nettoyage vérifié par Ruff, noms distincts pour l'enregistrement des couches et le modèle chargé, formatage uniforme |

Ces défauts peuvent également apparaître dans du code écrit manuellement. Les commentaires décoratifs, les noms ou la longueur d'un fichier sont des signaux de lisibilité à examiner, pas des preuves de génération automatique.

## Validation

Les contrôles utiles à cette modification sont les régressions du pipeline scientifique, les imports après déplacement, les contrats HTTP et les cas limites corrigés. Depuis la racine :

```powershell
python -m unittest discover -s tests -v
python -m src.data.audit_dataset
python -m src.inference.predict --help
python -m src.experiments.comparison.run_comparison --help
python -m src.experiments.benchmark.run_model_benchmark --help
```

La suite complète a passé **92 tests en 28,050 secondes** après les corrections de revue, avec Python 3.12 dans `.venv-comparison`. Elle vérifie notamment une prédiction via HTTP contre la référence enregistrée, la sérialisation TensorFlow, les checkpoints TCN et l'isolation de la calibration. Cette exécution a eu lieu hors de l'environnement restreint : une première tentative dans celui-ci rencontrait des refus d'accès Windows aux dossiers temporaires (`WinError 5`). Le journal local est `.tmp-tests/review-fixes-tests.log` ; ce dossier est ignoré par Git.

Le contrôle Ruff et la vérification du formatage passent sur `src/`, `webapp/` et `tests/`. La version de l'outil est fixée dans `requirements-dev.txt`, avec sa configuration dans `pyproject.toml`. Le fichier `src/preprocessing.py` est explicitement exclu pour préserver son empreinte. Cette convention rend les contrôles reproductibles ; elle ne remplace pas les tests fonctionnels. [Documentation officielle de Ruff](https://docs.astral.sh/ruff/configuration/).

Les deux scripts de vues modifiés ont passé une vérification de syntaxe V8 et sept groupes de contrôles de rendu avec un DOM simulé : indisponibilité, données partielles, scores absents et vrais scores nuls. Il ne s'agit pas d'un test visuel dans un navigateur.

L'audit du dataset a également réussi sur **18 238 échantillons** : les encodages correspondent aux enregistrements et aucun recouvrement entre partitions n'a été détecté selon les critères de l'auditeur (`cross_split_overlaps = 0`). Ce contrôle ne démontre pas l'absence de tous les clones approximatifs.

La conservation des artefacts scientifiques se vérifie aussi avec `git diff` sur `config/`, `dataset/`, `models/` et `results/`, ainsi que sur `src/preprocessing.py`. Les manifestes ne doivent pas être mis à jour pour masquer un changement de calcul.

## Corrections issues de la contre-revue du 7 octobre

- Le rapport imprimable lit désormais les alertes dans `risk_analysis.static_analysis.findings`, conformément à la réponse de l'API. Les textes des alertes restent échappés pour l'affichage HTML.
- Un moteur en échec, absent ou sans score exploitable s'affiche comme indisponible. Les vrais scores nuls restent visibles ; les anciens scores d'une réponse en échec ne sont pas réutilisés.
- La recherche d'un appel externe dans une boucle parcourt le corps délimité par ses accolades. Les longs commentaires ne consomment plus une fenêtre arbitraire de 1 200 caractères, et un appel placé après la boucle n'est plus classé à l'intérieur. Les parenthèses et blocs imbriqués sont pris en compte pour les boucles `for` et `while` à corps entre accolades ; cela reste une heuristique textuelle.
- Les preuves des alertes de risque sont prises dans les lignes originales, tandis que la détection utilise le texte masqué. Les signatures et les espaces internes aux chaînes Solidity sont conservés, sous réserve de la longueur maximale des extraits affichés.
- Le rapport utilise `confidence_percent` et l'intitulé « SCORE IA » : il n'affiche plus une probabilité sur `[0, 1]` comme un pourcentage ni une combinaison de scores qui n'existe pas.

Les régressions Python correspondantes sont dans `tests/test_analysis.py`. Les tests du rapport sont conservés dans `tests/frontend/report_view.test.js`, exécutables avec `node tests/frontend/report_view.test.js`. Les **45 assertions** de ce script ont été exécutées dans V8 avec un DOM simulé, Node.js n'étant pas installé dans l'environnement local. Il ne s'agit pas d'un contrôle visuel dans un navigateur. Les nouvelles régressions ont d'abord reproduit les défauts avant les corrections.

## Portée

Cette revue ne constitue ni une mesure automatique de l'origine du code, ni un audit de sécurité exhaustif du produit. Les heuristiques restent des indications sur le texte Solidity, sans analyse complète du graphe d'exécution. Les résultats scientifiques conservés n'ont pas été recalculés par la réorganisation ; leurs limites méthodologiques restent celles des [rapports d'expérience](experiments.md).
