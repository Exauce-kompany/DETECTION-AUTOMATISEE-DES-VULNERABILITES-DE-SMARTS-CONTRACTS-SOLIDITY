# Progression expérimentale après V3

Cette étude répond aux quatre étapes : architectures seules et combinées, code
complet contre entrée tronquée, concurrent préentraîné, comparaison selon plusieurs
critères. Le modèle actif et les partitions V3 sont conservés. Les résultats de
cette étude sont exploratoires : le test V3 a déjà servi à des évaluations.

## Matrice des expériences

| Architecture | Apprentissage | Entrées | Répétitions |
|---|---|---|---:|
| CNN | De zéro | Complètes / 512 premiers tokens lexicaux | 3 par entrée |
| BiLSTM | De zéro | Complètes / 512 premiers tokens lexicaux | 3 par entrée |
| CNN-BiLSTM hiérarchique | De zéro | Complètes / 512 premiers tokens lexicaux | 3 par entrée |
| CodeT5-small + tête dense | Encodeur préentraîné figé, tête entraînée | Complètes / 512 premiers tokens lexicaux | 3 par entrée |
| TF-IDF + régression logistique | Référence classique | Complètes / 512 premiers tokens lexicaux | 1 par entrée |

Il y a 24 entraînements neuronaux et deux références classiques. Pour CodeT5,
les graines font varier uniquement l'entraînement de la tête. Cela ne constitue
pas un fine-tuning de l'encodeur. Le processeur local est utilisé ; aucun GPU
NVIDIA n'a été détecté. L'encodeur CodeT5 est téléchargé depuis son dépôt officiel,
à une révision figée et enregistrée localement. Les contrats ne sont pas envoyés
à un service d'IA.

[CodeT5-small officiel](https://huggingface.co/Salesforce/codet5-small) est un
modèle de code généraliste. Cette expérience teste son transfert sur les entrées
normalisées du projet, pas un modèle préentraîné spécifiquement sur Solidity.

## Données et budget

Les 11 959 contrats d'entraînement, 1 709 de validation, 1 709 de calibration,
1 709 de test et 1 152 du holdout de source proviennent du snapshot V3 identifié
dans `config/comparison_v1.json`. Chaque époque parcourt tous les contrats de train.
Les poids d'échantillonnage par source et classe sont communs. Les variantes sont
entraînées sous PyTorch pour éviter de confondre architecture et framework. Les
poids de V3 ne sont pas repris : la variante CNN-BiLSTM est une nouvelle expérience.

Les réseaux partagent embeddings 32, dense 32, dropout 0,3, Adam à 0,001,
gradient limité à une norme de 1, maximum 12 époques et patience 3 sur la log-loss
de validation. Le learning rate est divisé par deux après deux époques sans
amélioration, minimum 1e-6. Graines : 42, 73 et 101. Les CNN ont 24 filtres de
taille 5 ; les LSTM ont 24 unités par direction. Les fenêtres CNN contiennent
256 positions. Le batch contient au maximum 32 contrats et 65 536 positions
après remplissage, sauf contrat individuel plus long. Aucune fin de contrat
n'est supprimée dans le mode complet. Ce budget égalise les passages sur les
données, pas le temps CPU ni le nombre de paramètres.

CodeT5 produit une moyenne pondérée par le nombre de sous-tokens et un maximum
global des états de l'encodeur sur toutes les fenêtres. Les tokens spéciaux sont
exclus de cette agrégation. Les caractéristiques sont standardisées sur train
uniquement, puis la même taille de tête dense est entraînée. Les embeddings
figés sont calculés une fois et réutilisés pour les trois graines.

## Sélection, calibration et test

Le checkpoint de chaque entraînement minimise la log-loss validation. La variante
est ensuite choisie par rappel moyen sur la ROC validation à FPR ≤ 10 %, puis F1
macro et log-loss. Cette sélection est enregistrée avant l'évaluation du test.
Une température et deux seuils sont ajustés uniquement sur calibration : un seuil
visant 90 % de rappel, l'autre visant au plus 10 % de faux positifs. Le test peut
s'écarter de ces cibles ; le rapport affiche les taux effectivement obtenus.

Le rapport compare F1 macro, rappel, précision, faux positifs, AUC, Brier, ECE,
variabilité entre graines, paramètres et latence. Le bootstrap apparié rééchantillonne
les groupes de contrats, pas les lignes indépendamment. Ses intervalles sont
exploratoires et ne corrigent pas les comparaisons multiples.

Les latences sont mesurées séquentiellement sur les mêmes 50 contrats de validation,
après échauffement. Elles incluent l'encodage propre à chaque modèle et l'encodeur
CodeT5. Le chargement des poids, le disque et la normalisation Solidity commune
sont exclus. Les temps d'extraction CodeT5 doivent être ajoutés au temps de tête
pour apprécier le coût complet de cette méthode.

## Incohérence découverte pendant l'ablation

Après troncature à 512 tokens lexicaux, 386 des 1 709 exemples de test ont un
préfixe identique à celui d'au moins un contrat de train, validation ou calibration.
Cela n'invalide pas l'audit de séparation des contrats complets, mais peut rendre
la comparaison tronquée trop optimiste. Le rapport fournit aussi un diagnostic
sur les 1 323 exemples communs sans cette collision, avec les seuils inchangés.
Cette restriction ne constitue pas un nouveau test indépendant et ne retire pas
les clones approximatifs inconnus. Une future étude confirmatoire devra regrouper
les familles/projets et les équivalences induites par toutes les représentations
avant de constituer une réserve indépendante.

Le holdout de source contient seulement six négatifs. Ses faux positifs ne permettent
pas une conclusion robuste. Les labels historiques restent une limite : une
confirmation nécessitera des annotations vérifiées, des négatifs comparables et
un jeu indépendant qui n'a pas guidé le développement.

## Exécution et reprise

Dans l'environnement dédié `.venv-comparison`, après installation de
`requirements-comparison.txt` :

```powershell
.\.venv-comparison\Scripts\python.exe -m unittest tests.test_comparison -v
.\.venv-comparison\Scripts\python.exe -u -m src.run_comparison
```

Le runner enchaîne les entraînements, les représentations préentraînées, la sélection,
la calibration et l'évaluation. Il écrit son état dans
`results/comparison/comparison-v1-20260922/execution_status.json` et les sorties
dans `execution.log`. Une reprise conserve les entraînements terminés et les
blocs de caractéristiques extraits. Un entraînement interrompu avant la fin
repart de zéro pour sa graine ; aucun checkpoint partiel n'est présenté comme fini.

La configuration et les empreintes des scripts sont figées dans `protocol.json`.
La reprise refuse une modification du code d'entraînement. À la fin, `report.md`,
`summary.csv`, `paired_comparisons.json` et `completed.json` décrivent les résultats
mesurés. L'existence des scripts ou des checkpoints seuls ne signifie pas que
l'étude est achevée. Le runner n'active et ne publie aucun modèle automatiquement.
