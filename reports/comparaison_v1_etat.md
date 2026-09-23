# Comparaison V1 : résultats finaux de l'expérience

Expérience `comparison-v1-20260922`, terminée le 23 septembre 2026 à 14 h 16 UTC
(15 h 16, heure locale). Les **26 entraînements et évaluations** sont achevés :
18 réseaux depuis zéro, six têtes CodeT5 et deux références TF-IDF.

Le [rapport complet](../results/comparison/comparison-v1-20260922/report.md)
présente les dix variantes, leurs performances, la calibration et les comparaisons
appariées. Les valeurs détaillées figurent dans
[summary.csv](../results/comparison/comparison-v1-20260922/summary.csv).

## Résultats sur le test exploratoire

Moyennes sur trois graines pour les réseaux, une estimation pour TF-IDF.
Les seuils du tableau sont ajustés exclusivement sur calibration avec un objectif
de rappel de 90 %, qui n'est pas garanti sur le test. Les 1 709 contrats du test V3
avaient déjà été consultés lors de travaux antérieurs.

| Modèle, code complet | F1 macro | Rappel | Taux de faux positifs | Latence médiane |
|---|---:|---:|---:|---:|
| CNN-BiLSTM | 84,54 % | 89,07 % | 19,84 % | 3,00 ms |
| CodeT5 figé + tête dense | 83,40 % | 86,18 % | 19,30 % | 558,48 ms |
| CNN | 83,36 % | 87,97 % | 21,07 % | 1,74 ms |
| BiLSTM | 83,10 % | 88,16 % | 21,76 % | 6,97 ms |
| TF-IDF + régression logistique | 81,39 % | 84,20 % | 21,34 % | 2,42 ms |

Les latences incluent l'encodage propre à chaque modèle et l'encodeur CodeT5,
mais excluent la normalisation Solidity commune, les accès disque et le chargement
des poids. Elles sont mesurées sur les mêmes 50 contrats de validation.

## Sélection et incertitude

**CodeT5 complet a été sélectionné sur validation**, selon le critère fixé avant
l'accès aux nouveaux résultats du test : rappel moyen à FPR ≤ 10 %, puis F1 macro
et log-loss. Cette sélection reste enregistrée dans `selection.json`.

Le CNN-BiLSTM complet présente ensuite le meilleur F1 moyen observé sur le test,
avec une latence médiane environ 186 fois inférieure à celle de CodeT5. Cette
observation ne doit pas devenir une nouvelle sélection faite sur le test.
Les seuils calibrés pour un FPR cible de 10 % sont évalués séparément dans le
rapport, qui affiche les FPR réellement obtenus.

L'écart de F1 moyen entre CNN-BiLSTM et CodeT5 est de **+1,14 point**. L'intervalle
bootstrap apparié à 95 % pour cet écart est d'environ **[−0,26 ; +2,69] points** :
il inclut zéro. Une supériorité générale du CNN-BiLSTM n'est donc pas établie.
Ce bootstrap rééchantillonne les groupes de contrats, conditionne sur les graines
et la calibration exécutées, et ne corrige pas les comparaisons multiples.

## Effet du code complet

Le F1 du CNN-BiLSTM passe de **74,85 %** avec les 512 premiers tokens lexicaux
à **84,54 %** avec le code complet : **+9,69 points**, avec un intervalle bootstrap
exploratoire de **[+7,20 ; +13,56] points**. Les cinq familles de modèles obtiennent
un meilleur F1 moyen avec le code complet dans cette expérience.

La troncature crée des collisions de préfixes pour 386 exemples du test. Sur le
sous-ensemble commun de 1 323 exemples sans ces collisions, le F1 des variantes
complètes est de 84,32 % pour CNN-BiLSTM et 83,32 % pour CodeT5. Ce diagnostic ne
constitue pas un nouveau test indépendant et n'exclut pas tous les clones approximatifs.

## Conclusion pour le projet

Les résultats soutiennent la conservation du contexte complet et ne justifient
pas le remplacement automatique du CNN-BiLSTM actif par CodeT5 figé. Le compromis
entre performances observées et coût d'inférence favorise le maintien de
l'architecture actuelle pendant la préparation d'une confirmation indépendante.

CodeT5 utilise un encodeur préentraîné **figé**, suivi d'une tête entraînée :
cette étude ne mesure pas un fine-tuning de CodeT5 sur Solidity et ne permet pas
de conclure que le préentraînement serait inutile en général.

Les réseaux de comparaison ont été réentraînés sous PyTorch. Leurs scores moyens
ne sont pas ceux du checkpoint TensorFlow actuellement actif dans SmartBug.
**Le modèle V3 actif n'a pas été remplacé.**

## Fichiers et reproduction

- `completed.json`, `protocol.json`, `selection.json` : fin d'exécution, protocole et sélection.
- `summary.csv`, `summary.json`, `paired_comparisons.json` : résultats et incertitude.
- `comparison.png`, `comparison.svg` : graphiques partageables.
- Chaque dossier de modèle : historique, calibration, évaluation et prédictions.
- `models/comparison/comparison-v1-20260922/` : checkpoints et scalers via Git LFS.
- `pretrained_manifest.json`, `pretrained_extraction.json` : révision CodeT5 et extraction.

L'encodeur se télécharge depuis `Salesforce/codet5-small`, à la révision
`b1ee9570c289f21b5922b9c768a1ce12957bf968`. Les six fichiers CodeT5 `.pt` versionnés
contiennent uniquement les têtes entraînées ; les deux `scaler.pkl` contiennent
la standardisation apprise sur train.

Pour reconstruire le cache CodeT5 sur un clone, fixer d'abord la révision publiée
depuis l'environnement de comparaison :

```python
import json
from pathlib import Path
from src.comparison_data import save_json

run = Path("results/comparison/comparison-v1-20260922")
manifest = json.loads((run / "pretrained_manifest.json").read_text())
save_json(Path(".cache-comparison/codet5/revision.json"),
          {key: manifest[key] for key in ("model_id", "revision")})
```

Puis `python -m src.comparison_pretrained` reconstruit les caractéristiques depuis
les données V3. Le [protocole](protocole_comparaison_v1.md) décrit la chaîne complète.
Les 34 tests du projet avaient passé avant publication ; les empreintes des 26
checkpoints et évaluations ont aussi été vérifiées après calcul.

Les labels historiques, les six seuls négatifs du holdout de source et le test
déjà consulté limitent les conclusions. Les durées murales d'entraînement incluent
des pauses et ne constituent pas des temps de calcul purs. Une confirmation sur
des contrats indépendants, avec labels vérifiés, reste nécessaire.
