# Comparaison V1 : état publié et conclusion provisoire

Snapshot publié le 23 septembre 2026, pour l'expérience
`comparison-v1-20260922`. Cette publication sauvegarde les modifications et les
résultats disponibles ; elle ne signifie pas que la comparaison est terminée.

## Ce qui est terminé

- 18 entraînements depuis zéro : CNN, BiLSTM et CNN-BiLSTM, chacun avec code
  complet ou 512 premiers tokens lexicaux, sur trois graines.
- Deux références TF-IDF + régression logistique, code complet ou tronqué.
- Conservation des 20 modèles terminés, des historiques et des prédictions de validation.
- Scripts de sélection, calibration, test, bootstrap apparié et génération du rapport.
- Neuf tests de comparaison et 25 tests du projet V3 réussis.

## Résultats disponibles

Résultats sur les **1 709 contrats de validation**, pas sur le test final.
Le F1 macro utilise le seuil brut 0,5 ; le rappel est mesuré séparément sur la
courbe ROC à un taux de faux positifs inférieur ou égal à 10 %.
Les réseaux sont résumés par la moyenne de trois graines, TF-IDF par un seul ajustement.

| Modèle | F1 macro, code complet | F1 macro, 512 tokens | Rappel à FPR ≤ 10 %, code complet |
|---|---:|---:|---:|
| CNN | 80,33 % | 75,85 % | 71,89 % |
| BiLSTM | 80,54 % | 75,94 % | 68,69 % |
| CNN-BiLSTM | 81,73 % | 75,48 % | 72,09 % |
| TF-IDF + régression logistique | 78,98 % | 70,27 % | 73,63 % |

Les écarts-types et AUC figurent dans le
[tableau détaillé](../results/comparison/comparison-v1-20260922/validation_progress.md).

## Conclusion provisoire

Conserver le CNN-BiLSTM avec le code complet est une décision raisonnable à ce
stade. Le gain le plus net observé concerne la conservation du contexte :
**+6,25 points de F1 macro** pour le CNN-BiLSTM complet par rapport à sa variante
tronquée. Les autres architectures bénéficient aussi du code complet.

L'avantage moyen de l'hybride sur les réseaux seuls reste modeste : +1,40 point
par rapport au CNN et +1,19 point par rapport au BiLSTM. Trois graines ne suffisent
pas à établir une supériorité générale ; les intervalles des comparaisons finales
restent à calculer.

La référence classique obtient le meilleur rappel au point de fonctionnement
FPR ≤ 10 % parmi les variantes terminées. Le deep learning n'est donc pas
supérieur sur tous les critères. Le choix définitif suivra le critère fixé sur
validation dans le protocole, après l'entraînement de tous les candidats.

## Ce qui reste en cours

CodeT5-small utilise un encodeur préentraîné figé, à la révision
`b1ee9570c289f21b5922b9c768a1ce12957bf968`, puis une tête dense entraînée :
ce n'est pas un fine-tuning de l'encodeur. Son extraction sur CPU est en cours
au moment de ce snapshot. Le runner enchaînera ensuite les six entraînements
de têtes, la sélection sur validation, la calibration et l'évaluation commune.

Les scores CodeT5, les résultats du test, les latences comparatives et le
classement final ne sont pas encore disponibles dans cette publication.
Le modèle V3 actif de SmartBug n'a pas été remplacé.

## Limites à conserver dans le mémoire

Le test V3 a déjà été consulté : la comparaison sera exploratoire. Une nouvelle
réserve indépendante sera nécessaire pour une confirmation. La troncature crée
des collisions de préfixes pour 386 exemples du test ; un diagnostic commun sur
les exemples sans ces collisions est prévu. Les labels historiques et les six
seuls négatifs du holdout de source limitent aussi les conclusions.

Les durées murales d'entraînement incluent de longues pauses de la machine.
Elles ne doivent pas être assimilées à des temps de calcul purs. Les latences
seront mesurées séparément.

Pour les hyperparamètres, les partitions et les commandes de reprise, consulter
le [protocole](protocole_comparaison_v1.md). Les caches locaux CodeT5 sont exclus
de Git ; les poids des modèles terminés sont distribués via Git LFS.
