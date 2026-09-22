# Corrections de l'audit — 22 septembre 2026

Le protocole actif est V3. Le modèle a été réellement réentraîné, calibré, évalué et raccordé à l'application. Les artefacts V2 sont conservés comme preuves historiques ; leurs métriques ne sont plus présentées comme celles du modèle actif.

## Les dix chantiers

Le rapport initial détaillait douze constats ; les deux derniers (robustesse de construction et reproductibilité) sont inclus dans les chantiers 5 et 10 ci-dessous.

| Nº | Problème | Correction effectuée | Limite restante |
|---|---|---|---|
| 1 | Fuites entre partitions | Déduplication globale puis partitions par composantes de familles, variantes structurelles et provenance ; audit de six empreintes sur les dix paires de partitions | Des fragments de bibliothèques et des clones approximatifs peuvent encore être partagés |
| 2 | Labels contradictoires | 796 exemples exclus en quarantaine quand le code canonique ou les tokens d'entrée portent des labels opposés | Une expertise des sources est nécessaire pour les réintégrer ; aucun label n'a été inventé |
| 3 | Code tronqué à 512 tokens | Toutes les fenêtres de 256 tokens sont encodées par un CNN puis agrégées par un BiLSTM ; une cible par contrat | Le réseau reste une représentation apprise, sans analyse sémantique complète ni garantie de localiser la faille |
| 4 | Commentaires révélant les annotations | Suppression lexicale des commentaires en respectant les chaînes ; normalisation des identifiants ; vocabulaire appris sur le train uniquement | Les indices de style/provenance ne peuvent pas tous être éliminés |
| 5 | Audits et construction fragiles | Audit bloquant des représentations complètes, labels, groupes et empreintes ; manifestes ; audit V2 sérialise ses conflits internes et corrige l'union des labels ; exceptions pour CSV/sources invalides et dimensions dynamiques | Les anciens scripts V2 restent des archives et ne doivent pas servir à valider le protocole V3 |
| 6 | Hashes de sens différents | SHA-256 du texte brut, du code canonique, des tokens, de la structure et de l'encodage calculés séparément ; références d'origine conservées | Une empreinte commune ne prouve pas l'identité sémantique de deux contrats |
| 7 | Classe « non vulnérable » trop absolue | Exclusion des 503 négatifs CGT dont la couverture des sept familles est incomplète ; définition et affichage d'une absence de signal/annotation | Les autres labels hérités restent partiellement non vérifiés ; leur provenance est indiquée |
| 8 | Biais de source et fonctions sans contexte | Exclusion de 1 141 fonctions sans contrat parent ; groupes, pondérations source/label sur le train, métriques par source/granularité/confiance d'annotation ; CGT réservé avec toutes ses familles liées | Manque de négatifs comparables, surtout dans la source réservée ; le biais n'est pas prétendu supprimé |
| 9 | Confiance non calibrée et combinaison 70/30 | Température et seuil ajustés sur une partition de calibration dédiée ; Brier, ECE et classes de fiabilité ; suppression du score combiné ; scores IA/statique séparés | L'ECE augmente fortement hors source ; la calibration n'est pas une garantie individuelle |
| 10 | Prétraitements et versions incohérents | Même prédicteur CLI/web, correction du tokenizer V1, configuration versionnée, graines 42/73/101, déterminisme TF, hashes et identifiant d'expérience ; UI lit les artefacts actifs ; baseline TF-IDF | Résultats bit à bit dépendants de l'environnement ; contrôle visuel de l'interface encore à effectuer |

## Architecture et hyperparamètres

Le modèle est **entraîné depuis zéro**, sans poids pré-entraînés ni fine-tuning. Il comporte **277 017 paramètres**.

1. Tokenisation du code sans commentaires. Les identifiants utilisateur sont renommés de façon cohérente dans chaque contrat. Les chaînes et adresses longues sont remplacées par des catégories ; les autres nombres sont conservés pour l'entrée du réseau. La normalisation structurelle plus agressive ne sert qu'au regroupement.
2. Vocabulaire de 8 000 entrées appris uniquement sur le train. PAD, UNK, 256 octets et un séparateur sont réservés. Un token hors vocabulaire est représenté par sa séquence UTF-8 encadrée, ce qui évite de fusionner tous les inconnus en une même entrée effective.
3. Fenêtres de 256 tokens encodés, sans abandon de la fin du contrat. Embedding de dimension 32 ; convolution de 24 filtres, noyau 5, activation ReLU ; moyennes et maxima masqués par fenêtre.
4. BiLSTM de 24 unités par direction sur la séquence de fenêtres. Moyennes et maxima masqués sur les fenêtres ; dropout 0,30 ; Dense 32 ReLU ; Dense 1 produisant un logit. Le label est appliqué au contrat complet, jamais à chaque fenêtre positive par défaut.
5. Adam, taux initial 0,001 ; perte binaire calculée depuis les logits. Lots de référence de 32 contrats, réduits selon leur longueur (32/32/16/8/4/2). Pondérations par couple source/label calculées sur le train, plafonnées avant renormalisation.
6. Maximum 12 époques ; arrêt anticipé après 3 époques sans amélioration de `val_loss`, réduction du taux par 0,5 après 2 époques, minimum 0,000001. Checkpoint au meilleur `val_loss`, rechargé avant évaluation.
7. Trois graines 42, 73, 101. Respectivement 6, 6 et 8 époques exécutées. Choix par log-loss de validation : graine 73, époque 3. F1 macro sur la validation : moyenne 79,36 %, écart-type 1,39 point sur les trois graines.

La configuration exacte est dans `config/v3.json`. `src/train_v3.py` orchestre entraînement, validation, calibration et test ; `src/model_v3.py` construit le réseau, et `src/experiment_v3.py` calcule les données par lot et les métriques. Le réseau ne prédit ni types de vulnérabilités ni lignes ; ces informations dans l'application viennent du moteur statique.

## Données et indépendance

26 134 exemples V2 initiaux ; 2 440 exclusions en quarantaine et 5 456 doublons retirés, avec conservation des références. Restent **18 238 contrats/représentations de contrats**.

| Partition | Total | Groupes | Label 0 | Label 1 | Usage |
|---|---:|---:|---:|---:|---|
| train | 11 959 | 9 196 | 6 072 | 5 887 | Poids et vocabulaire |
| validation | 1 709 | 1 310 | 867 | 842 | Checkpoints et sélection de graine |
| calibration | 1 709 | 1 235 | 868 | 841 | Température et seuil |
| test | 1 709 | 1 295 | 867 | 842 | Évaluation finale après gel |
| source_holdout | 1 152 | 1 054 | 6 | 1 146 | Transfert à une source réservée |

Les composants contenant CGT vont dans `source_holdout`, y compris leurs éventuelles variantes d'autres sources. Le reste est réparti avec `StratifiedGroupKFold`, 10 plis, graine 42 : pli 0 test, 1 validation, 2 calibration, autres train. Les strates source/label rares sont regroupées avec une strate de même label ; les proportions exactes sont celles du tableau.

Les noms de granularité manquants sont explicitement `contract_inferred`, pas une certitude reconstituée. Les annotations CGT ne couvrent que les familles déclarées : arithmetic, unchecked_low_calls, reentrancy, denial_service, front_running, time_manipulation, bad_randomness. Les annotations des autres sources conservent leur périmètre d'origine ; le modèle binaire ne constitue pas un détecteur exhaustif de ces familles.

Manifest : `dataset/v3/manifest.json`. Identifiant : `403bb88190a299381be57635f48aba7053ee48257ed7df1a6312b883f7428ae7`. L'audit indépendant a relu les 18 238 enregistrements, recalculé leur encodage et comparé les tableaux/labellisations : **aucune différence et aucun recouvrement selon les six clés contrôlées**.

## Résultats effectivement mesurés

Expérience : `v3-403bb881-20260922T101754Z`. Température ajustée sur calibration : 0,9368407446 ; seuil : 0,2694744341. Le seuil maximise la précision sous contrainte de rappel d'au moins 90 % sur la calibration. Le test n'intervient pas dans ce choix.

| Méthode | Exactitude test | F1 macro | Rappel positif | ROC AUC | Brier | ECE, 10 classes |
|---|---:|---:|---:|---:|---:|---:|
| Réseau brut, seuil 0,5 | 82,45 % | 82,41 % | 79,10 % | 0,90695 | 0,125856 | 5,65 % |
| Réseau calibré, seuil actif | 82,15 % | 82,10 % | 88,95 % | 0,90695 | 0,125753 | 5,12 % |
| TF-IDF + régression logistique calibrée | 81,39 % | 81,39 % | 84,20 % | 0,90500 | 0,132345 | 8,00 % |

La référence utilise les tokens complets, TF-IDF unigrammes/bigrammes (20 000 features, `min_df=2`, TF sous-linéaire), régression logistique `C=1`, solveur liblinear, maximum 1 000 itérations, mêmes pondérations du train, puis sa propre calibration indépendante. Le faible écart de performances ne justifie pas une affirmation de supériorité statistiquement démontrée du réseau.

Matrice de confusion active : lignes = labels réels [0,1], colonnes = prédictions [0,1] : `[[655,212],[93,749]]`. Réduire le seuil diminue les faux négatifs (176 → 93) en augmentant les faux positifs (124 → 212). Le rappel cible n'est pas atteint exactement sur le test (88,95 %).

Sur `source_holdout`, rappel positif 89,01 %, F1 macro 47,73 %, ECE 29,84 %. Cet ensemble comporte seulement six négatifs ; sa précision globale de 88,63 % est moins élevée que celle d'une règle toujours positive (99,48 %). La forte asymétrie des classes rend l'exactitude et l'aire précision-rappel peu informatives pour juger ce transfert. **Le biais de source et la qualité des annotations restent des limites de recherche, pas des points déclarés résolus par le seul code.**

Les prédictions individuelles, résultats par source/granularité/confiance d'annotation et classes du diagramme de fiabilité sont dans `results/v3/<run>/`. Le modèle ne peut pas être comparé directement au 85,87 % historique V2 : les partitions, exclusions et représentations ont changé.

## Vérifications et reproduction

- `python -m unittest discover -s tests -v` : 25 tests réussis, dont tests HTTP avec base SQLite temporaire, concordance d'une prédiction du test avec le modèle actif, invariance aux annotations dans les commentaires, conservation des tokens finaux, masquage, sauvegarde/rechargement et refus des données invalides.
- `python -m src.audit_dataset_v3` : intégrité et encodages complets vérifiés indépendamment, zéro recouvrement.
- Syntaxe de 35 fichiers Python et des 8 scripts JavaScript vérifiée. Les écrans web n'ont pas pu être inspectés visuellement : aucun navigateur connecté n'était disponible.
- Les avertissements Keras sur la cardinalité inconnue du générateur ne signifient pas ici une perte de contrats : un test vérifie le passage de tous les contrats à chaque itération, et les prédictions sauvegardées couvrent les partitions attendues. Les historiques des trois graines sont conservés.
- Les fichiers Python de l'expérience correspondent aux hashes de `experiment.json` ; les artefacts sont protégés par des manifestes SHA-256. Git conserve les sources en LF et les artefacts générés octet pour octet afin que les contrôles restent valides après clonage.

Pour une nouvelle expérience : `python -m src.train_v3`. Pour reconstruire sans écraser l'archive : `python -m src.build_dataset_v3 --output dataset/v3_rebuilt`. Ne pas réutiliser le test comme critère d'ajustement après avoir lu ces résultats.

Restent à faire avec des preuves externes : revoir les annotations mises en quarantaine, recueillir des négatifs vérifiés dans les sources principalement positives, et obtenir un corpus indépendant représentatif des futurs contrats. Ces tâches nécessitent de nouvelles données ou une expertise Solidity, et ne peuvent pas être remplacées par un changement automatique de labels.
