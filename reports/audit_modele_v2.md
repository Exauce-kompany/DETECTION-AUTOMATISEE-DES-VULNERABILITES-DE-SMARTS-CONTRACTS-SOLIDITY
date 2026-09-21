Audit du modèle de détection de vulnérabilités — 21 septembre 2026

Cet audit repose sur la lecture des scripts, l'inspection des deux archives Keras, le recalcul des entrées numériques de tous les exemples V2 et une nouvelle inférence du modèle V2 sur les 2 363 exemples de test. Aucun entraînement ni changement des scripts applicatifs n'a été effectué. Les mesures sont conservées dans les trois fichiers JSON `audit_modele_v2_*` de ce dossier.

**Le modèle utilisé est un BiLSTM supervisé, entraîné à partir de zéro.** L'application charge `models/smart_contract_vulnerability_model_v2.keras` depuis `webapp/predictor.py:28`. Sa construction est définie dans `src/train_v2.py:391`. Il comporte 346 978 paramètres entraînables. L'archive indique Keras 3.15.1 et une sauvegarde le 28 août 2026. La vérification d'inférence a utilisé l'environnement Conda `smartcontract`, Python 3.12.13 et TensorFlow 2.21.0.

Un modèle préentraîné réutiliserait des poids appris auparavant sur une autre tâche ou un autre corpus. Un fine-tuning chargerait ces poids puis les ajusterait. Ici, `main()` appelle `build_model(len(vocabulary))`, puis `model.fit()`. Aucun modèle V1, CodeBERT, GPT, Transformer ni embedding externe n'est chargé dans cet entraînement. Les embeddings sont initialisés aléatoirement et appris avec le reste du réseau. V2 signifie ici nouvel entraînement sur un dataset enrichi. Le chargement du fichier `.keras` par l'application est une utilisation du modèle déjà entraîné, pas une opération de fine-tuning.

Le réseau reçoit uniquement les identifiants numériques issus du champ `context`. Sa cible est `has_vulnerability`, avec 0 = `non_vulnerable` et 1 = `vulnerable`. Il n'apprend pas directement les types de vulnérabilités, les lignes concernées ou le coût en gas. Les types et lignes affichés par l'application proviennent de `webapp/risk_analyzer.py`, un moteur heuristique distinct. Les estimations de performance viennent de `webapp/performance_analyzer.py`. `webapp/app.py:869` distingue explicitement ces capacités.

**La chaîne de traitement comprend plusieurs fichiers.** Le fichier ouvert dans l'IDE ne réalise pas l'entraînement.

| Fichier | Fonction effective |
|---|---|
| `src/audit_cgt_overlap.py` | Prépare les candidats CGT, contrôle leurs recouvrements et conflits avec V1, produit notamment `results/cgt_overlap_audit/cgt_unique_final.csv`. |
| `src/build_dataset_v2.py` | Construit les trois JSON bruts V2 à partir de V1 et des candidats CGT retenus. |
| `src/solidity_tokenizer.py` | Découpe le texte Solidity par expressions régulières. |
| `src/prepare_data_v2.py` | Construit le vocabulaire sur le train, encode les trois splits, écrit les tableaux NumPy et les métadonnées. |
| `src/train_v2.py` | Construit le réseau, entraîne ses poids, suit la validation, sauvegarde le meilleur modèle et réalise un test final rapide. |
| `src/evaluate_v2.py` | Recharge le modèle sauvegardé et calcule le rapport de classification, les métriques et les prédictions de test. |
| `src/predict_v2.py` et `webapp/predictor.py` | Utilisent le modèle pour de nouveaux codes. |

Les fichiers V1 `train.py` et `train_final.py` définissent la même architecture générale et utilisent `dataset/processed/prepared`. `train_old.py` emploie d'autres paramètres, notamment un batch de 16 et 50 époques maximales. Plusieurs scripts V1 écrivent aux mêmes emplacements : les fichiers seuls ne permettent pas d'attribuer avec certitude un entraînement historique V1 à une commande précise. Le modèle chargé par l'application actuelle est toutefois clairement V2.

**Le fichier `build_dataset_v2.py` assemble des données, sans optimiser de poids.** Ses chemins sont définis aux lignes 14 à 40. Il dépend du dossier voisin `../smart-contract-vuln-dataset/data/processed/balanced_stage1_resplit_721/has_vul_721_stratified_v1` et du CSV des candidats CGT. Ces dépendances étaient présentes lors de l'audit.

Son déroulement est le suivant :

1. `load_v1()` lit les trois partitions V1 déjà constituées. Ce script n'effectue pas lui-même de tirage train/validation/test.
2. `load_cgt_candidates()` lit le CSV et convertit `label`, `n_assessed_types` et `n_assessments` en valeurs numériques.
3. `select_cgt_candidates()` conserve tous les candidats positifs et seulement les candidats négatifs dont au moins quatre types de vulnérabilités ont été évalués. `MIN_NEGATIVE_COVERAGE = 4` est un paramètre de sélection des données, pas un hyperparamètre neuronal.
4. `convert_cgt_to_v1_format()` lit chaque fichier Solidity indiqué dans le CSV, découpe son code en lignes et crée un enregistrement avec son contexte, sa cible, sa provenance et ses métadonnées. Un fichier absent ou vide est sauté.
5. `build_v2()` concatène les exemples CGT au train V1. La validation et le test restent ceux de V1.
6. `verify_unchanged_split()` vérifie les tailles et listes de `sample_id`, puis `save_v2()` écrit les JSON et le résumé.

L'enrichissement effectivement enregistré comprend 1 693 exemples CGT : 1 190 positifs et 503 négatifs. Le train V1 contenait donc 17 411 exemples. Validation et test V2 sont identiques octet par octet aux fichiers V1 voisins, ce que cet audit a vérifié indépendamment.

| Partition V2 | Classe 0 | Classe 1 | Total | Part du total V2 |
|---|---:|---:|---:|---:|
| Entraînement | 9 182 | 9 922 | 19 104 | 73,10 % |
| Validation | 2 298 | 2 369 | 4 667 | 17,86 % |
| Test | 1 153 | 1 210 | 2 363 | 9,04 % |
| Total | 12 633 | 13 501 | 26 134 | 100 % |

Les noms de dossiers contenant `721` décrivent le protocole d'origine : les proportions finales V2 ne sont plus exactement 70/20/10 après l'enrichissement du train. Ce n'est pas une erreur d'entraînement, mais il faut documenter les proportions réelles.

**La préparation transforme du texte en tableaux numériques.** `prepare_data_v2.py:101` reconstruit le code à partir de `context`. `tokenize_solidity()` reconnaît mots-clés, identifiants, nombres, chaînes, symboles et commentaires. Il ne construit pas d'arbre syntaxique ni de graphe de contrôle. Certains opérateurs composés non prévus par l'expression régulière sont séparés en plusieurs tokens.

Le vocabulaire contient les 4 998 tokens les plus fréquents du train, plus `<PAD>` = 0 et `<UNK>` = 1. La validation et le test ne servent pas à choisir ce vocabulaire, ce qui est correct. Chaque exemple est limité aux 512 premiers tokens, puis complété à droite avec des zéros s'il est plus court. Les fichiers `X_train.npy`, `X_val.npy`, `X_test.npy` ont respectivement les dimensions `(19104, 512)`, `(4667, 512)` et `(2363, 512)`. Les fichiers `y_*.npy` contiennent les cibles entières correspondantes. Ils se trouvent dans `dataset/v2/prepared`, avec `vocabulary.pkl`, `labels.json` et `preparation_metadata.json`.

Le recalcul de la tokenisation et de l'encodage a donné **zéro différence** avec les tableaux sauvegardés, sur les trois partitions. Les cibles numériques correspondent également aux JSON. Le taux de tokens inconnus parmi les tokens effectivement conservés vaut 4,28 % sur le train, 4,78 % sur la validation et 4,35 % sur le test.

**L'architecture apprend une représentation du texte puis une décision binaire.** Une dimension `B` représente le nombre d'exemples d'un lot.

| Couche | Sortie | Paramètres | Rôle |
|---|---|---:|---|
| Entrée entière | `(B, 512)` | 0 | Identifiants de tokens. |
| Embedding 5 000 × 64 | `(B, 512, 64)` | 320 000 | Apprend un vecteur de 64 nombres par token ; masque les zéros de padding. |
| BiLSTM, 32 unités par direction | `(B, 512, 64)` | 24 832 | Analyse la séquence dans les deux sens et concatène leurs représentations. |
| GlobalAveragePooling1D | `(B, 64)` | 0 | Moyenne les représentations des positions non masquées. |
| Dropout 0,40 | `(B, 64)` | 0 | Désactive aléatoirement 40 % des activations pendant l'entraînement. |
| Dense 32, ReLU | `(B, 32)` | 2 080 | Combine les caractéristiques pour la décision. |
| Dropout 0,30 | `(B, 32)` | 0 | Seconde régularisation pendant l'entraînement. |
| Dense 2, Softmax | `(B, 2)` | 66 | Produit deux scores positifs dont la somme vaut 1. |
| Total | | **346 978** | |

Les LSTM utilisent `tanh` pour l'état transformé, `sigmoid` pour les portes, et conservent une sortie à chaque position (`return_sequences=True`). Leurs dropout interne et récurrent sont à 0 ; les deux dropout explicites sont situés après le pooling et la couche dense. La bidirectionnalité ne donne pas accès aux tokens supprimés au-delà de la position 512. Le [fonctionnement de Bidirectional est décrit par Keras](https://keras.io/api/layers/recurrent_layers/bidirectional/).

**Le fichier d'entraînement central est `src/train_v2.py`.** Les constantes sont aux lignes 40 à 47, la construction à partir de la ligne 391, les callbacks à partir de 492 et l'appel à `fit()` à la ligne 715.

| Réglage | Valeur et signification |
|---|---|
| Graine aléatoire | 42 pour NumPy et TensorFlow. |
| Batch size | 32 exemples par mise à jour des poids, soit 597 lots par époque avec ce train. |
| Époques maximales | 30 passages complets sur le train. |
| Longueur maximale | 512 tokens. |
| Vocabulaire | 5 000 entrées. |
| Dimension d'embedding | 64. |
| LSTM | 32 unités dans chaque direction. |
| Dense intermédiaire | 32 unités, activation ReLU. |
| Dropout | 0,40 puis 0,30. |
| Optimiseur | Adam, taux initial 0,001. |
| Paramètres Adam sérialisés | `beta_1=0.9`, `beta_2=0.999`, `epsilon=1e-7`, `amsgrad=False`. |
| Fonction de perte | `sparse_categorical_crossentropy`, adaptée aux labels entiers 0/1 et à la sortie Softmax à deux classes. |
| Métrique suivie | Accuracy ; les métriques par classe sont calculées dans l'évaluation détaillée. |
| Poids de classes | `balanced` : `N / (2 × effectif_classe)`, soit environ 1,0403 pour 0 et 0,9627 pour 1. |
| Mélange | `shuffle=True` pendant l'entraînement. |
| EarlyStopping | Suit `val_loss`, patience 6, restaure les meilleurs poids. |
| ReduceLROnPlateau | Suit `val_loss`, multiplie le taux par 0,5 après 3 époques sans amélioration suffisante, plancher 0,000001. |
| ModelCheckpoint | Écrit seulement les nouveaux meilleurs modèles selon `val_loss`. |

Le train sert à calculer les gradients et modifier les poids. La validation est évaluée à la fin de chaque époque sans mise à jour des poids ; elle sert au choix de l'époque et à la réduction du taux d'apprentissage. Le test intervient ensuite dans `evaluate_test()`, puis peut être recalculé par `evaluate_v2.py`. Il s'agit d'une validation sur partition fixe, pas d'une validation croisée à plusieurs plis.

L'historique montre **10 époques réellement exécutées**, avec la meilleure `val_loss` à l'époque 4 : 0,341624, pour une validation accuracy de 84,96 %. La perte de validation se dégrade ensuite tandis que la perte de train diminue : c'est un signe de surapprentissage. L'arrêt à l'époque 10 est cohérent avec une patience de 6. Le taux passe à 0,0005 pour l'époque 8. L'accuracy de validation maximale survient à l'époque 8, mais le critère de sélection est la perte : retenir l'époque 4 est donc cohérent. Voir la [définition officielle d'EarlyStopping](https://keras.io/api/callbacks/early_stopping/).

**Les résultats enregistrés sont reproductibles avec le fichier de modèle présent.** La nouvelle inférence a reproduit exactement les 2 363 prédictions et toutes leurs probabilités sauvegardées.

| Mesure V2 | Valeur |
|---|---:|
| Accuracy | 85,8654 % |
| Précision de la classe vulnérable | 88,7611 % |
| Rappel de la classe vulnérable | 82,8926 % |
| F1 de la classe vulnérable | 85,7265 % |
| F1 macro | 85,8641 % |
| Perte de test | environ 0,344680 |

| Réalité / Prédiction | Non vulnérable | Vulnérable |
|---|---:|---:|
| Non vulnérable | 1 026 | 127 |
| Vulnérable | 207 | 1 003 |

Le modèle manque donc 17,11 % des exemples vulnérables de ce test. V1 obtenait 84,6805 % d'accuracy : V2 gagne 1,1849 point, soit 28 décisions correctes supplémentaires au total, avec 19 faux négatifs et 9 faux positifs de moins. La correspondance des cibles V1/V2 a été vérifiée. Cette comparaison porte sur un entraînement enregistré de chaque version ; elle ne suffit pas à isoler l'effet causal du seul enrichissement CGT, car le vocabulaire est aussi reconstruit et les entraînements sont stochastiques.

**Les principales incohérences se situent dans les données et le protocole d'évaluation.** Les points ci-dessous distinguent les faits mesurés des risques qu'ils créent.

1. **Recouvrement des entrées effectivement vues par le réseau — critique.** En comparant directement les lignes des matrices NumPy, 841 exemples de test sur 2 363, soit 35,59 %, ont une entrée de 512 entiers strictement identique à une entrée de train. Ils correspondent à 523 groupes distincts. Pour la validation, 1 572 exemples partagent leur entrée avec le train. L'absence de code brut identique entre partitions ne garantit donc pas l'indépendance des entrées du modèle. Les tokens inconnus et la troncature peuvent effacer les différences entre codes.

   L'accuracy atteint 93,34 % sur les 841 exemples concernés contre 81,73 % sur les 1 522 autres. Ces sous-groupes ont aussi des distributions différentes : 81,73 % est un diagnostic, pas un score de généralisation « corrigé ». Remède : regrouper les familles de contrats/projets et les clones avant le découpage, puis contrôler aussi les entrées encodées. Traiter les collisions produites par la représentation et reconstruire un benchmark versionné indépendant. Garder le vocabulaire appris sur le train uniquement. Les méthodes de [validation par groupes de scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data) décrivent le principe de séparation des observations dépendantes.

2. **Labels contradictoires — critique.** Les exemples de test aux indices Python 257 et 258 contiennent exactement le même code et le même nom de contrat `0xe47014f16c55ddd3add5b68b951e10fcea7da686.sol`, mais portent respectivement les classes 1 (`zeus_vulnerable`) et 0 (`zeus_safe`). Aucune décision déterministe sur ce code ne peut satisfaire les deux étiquettes. Le test contient aussi 11 groupes de code brut dupliqué. Dans le train, 140 groupes d'entrées numériques identiques ont des labels contradictoires. Ceux-ci peuvent résulter de la troncature de codes différents et ne sont pas tous une erreur d'annotation. Remède : résoudre les conflits à partir des preuves sources, mettre les cas indécidables à part et améliorer la représentation quand l'information discriminante a été perdue.

3. **Troncature massive — majeure.** 14 914 exemples du train (78,07 %), 3 765 de validation (80,67 %) et 1 882 du test (79,64 %) dépassent 512 tokens. La médiane du test est de 960 tokens. Le label décrit souvent un contrat ou fichier complet, alors que le réseau ne voit que son début. Une faille située après la coupure peut être totalement invisible. Remède : traitement par fonctions avec contexte, fenêtres couvrant le code complet, ou représentation hiérarchique. Conserver toutes les fenêtres d'un contrat dans une même partition. Ne pas attribuer automatiquement le label positif du contrat à toutes ses fenêtres : employer des annotations locales ou une agrégation au niveau du contrat.

4. **Annotations de vulnérabilité conservées dans les commentaires — majeure.** Le tokenizer inclut les commentaires dans ses tokens. Le test contient notamment `// SWC-107-Reentrancy: L95` et des annotations `@vulnerable_at_lines`. Ces chaînes peuvent révéler une annotation ou la provenance du benchmark. La présence de ces indices est confirmée ; leur contribution exacte au score n'a pas été isolée par une étude d'ablation. Remède : retirer les annotations et métadonnées d'audit avant l'apprentissage, ou supprimer les commentaires avec une analyse lexicale respectant les chaînes de caractères. Refaire le vocabulaire et réentraîner, en utilisant exactement le même prétraitement à l'inférence.

5. **Audit existant incomplet — majeure.** `audit_dataset_v2_official.py:111` utilise des hashes fournis dans les données et un hash du contexte brut, sans comparer les matrices encodées. Les exemples hérités sans `dedup_hash_normalized` ne participent pas à cet index ; ils participent seulement à l'index du contexte. La fonction `audit_internal()` affiche les doublons/conflits internes, mais ces compteurs ne sont pas inclus dans le JSON final. La conclusion à partir de la ligne 798 se fonde sur les seuls recouvrements inter-partitions du hash dit officiel. Elle ne suffit pas à valider le dataset. De plus, le test `labels_a != labels_b` manque un conflit lorsque les deux ensembles valent `{0, 1}`. Remède : recalculer des empreintes homogènes, enregistrer les conflits internes, contrôler l'union des labels et bloquer les étapes suivantes selon des critères explicites.

6. **Sens hétérogène des champs de hash.** `build_dataset_v2.py:340` stocke `fp_sol` dans `dedup_hash_raw`. Selon le README CGT local, `fp_sol` est déjà une empreinte du source sans espaces ni commentaires ; ce n'est pas un hash brut comparable automatiquement à ceux d'autres sources. Remède : conserver cette valeur dans `source_fingerprint`, et calculer séparément un SHA-256 du code brut et une empreinte normalisée avec un algorithme commun versionné.

7. **« Non vulnérable » est un label trop absolu.** Les 503 négatifs CGT n'ont pas de résultat positif parmi les types sélectionnés, avec au moins quatre types évalués. Cela n'établit pas l'absence de toute vulnérabilité. Le filtre CGT cible sept familles correspondant à huit identifiants SWC et ne couvre pas toutes les failles Solidity. Le train comprend aussi 13 351 exemples issus de résultats Slither, 858 exemples de fonctions avec vulnérabilités injectées, ainsi que des annotations d'audit et des données héritées. Remède : définir précisément le périmètre des vulnérabilités, conserver les types évalués et non évalués et présenter la classe 0 comme une absence de vulnérabilité annotée dans ce périmètre. Prévoir des labels inconnus quand la couverture est insuffisante.

8. **Biais de source et granularités mélangées.** Dans le train, DAppSCAN, ScrawlD et SolidiFI ne fournissent ici que des positifs. Le réseau peut associer leur style au label. Un classifieur de contrôle utilisant seulement la source et le label majoritaire de chaque source atteindrait déjà environ 67,41 % sur ce test. Les données mélangent contrats, fichiers et fonctions ; le test contient 104 fonctions SolidiFI, toutes positives, toutes correctement classées. Ce résultat ne mesure pas la capacité à reconnaître des fonctions sûres de cette source. Remède : évaluer par source, granularité et famille de vulnérabilités, ajouter des négatifs comparables, séparer les familles de contrats et prévoir un test sur une source ou période distincte. L'absence de CGT dans le test permet une comparaison sur V1, mais ne valide pas directement la généralisation à CGT.

9. **Le score Softmax n'est pas une confiance calibrée démontrée.** `webapp/predictor.py:259` traduit des seuils 0,60/0,75/0,90 en niveaux de confiance. `risk_analyzer.py:721` combine arbitrairement 70 % du score IA et 30 % du score heuristique. Aucun protocole de calibration ou de validation du score combiné n'apparaît dans la chaîne examinée. Remède : calibration sur données indépendantes de l'entraînement, diagramme de fiabilité et métrique de calibration ; choisir le seuil de décision selon le compromis faux négatifs/faux positifs sur validation seulement. Afficher séparément le score neuronal et les alertes heuristiques. Voir la [documentation de calibration de scikit-learn](https://scikit-learn.org/stable/modules/calibration.html).

10. **Ancienne inférence V1 incohérente.** `src/predict.py:126` utilise `code.split()`, alors que `src/prepare_data.py:160` utilise `tokenize_solidity()`. Des symboles collés deviennent donc des tokens différents. Ce défaut concerne cette ancienne commande ; l'application V2 utilise bien le tokenizer partagé. Remède : centraliser préparation et inférence dans une seule fonction versionnée, et archiver explicitement les commandes obsolètes.

11. **Construction et vérifications fragiles.** `verify_unchanged_split()` compare les IDs, pas tout le contenu ; plusieurs exemples hérités n'ont pas de `sample_id`. Des fichiers CGT absents sont sautés. `prepare_data_v2.py:581` impose les tailles 19 104/4 667/2 363 en dur et utilise des `assert`, désactivables avec Python optimisé. Remède : identifiants obligatoires, manifestes de sources, refus ou journal explicite des exclusions, comparaison des contenus, contrôles d'intégrité dynamiques et exceptions explicites. Les vérifications indépendantes confirment néanmoins la cohérence des fichiers actuellement présents.

12. **Reproductibilité et présentation.** Les graines NumPy/TensorFlow existent, mais la graine Python et le déterminisme des opérations ne sont pas explicitement configurés. Plusieurs scripts, chemins externes et valeurs d'interface codées en dur compliquent le suivi des versions. Remède : `tf.keras.utils.set_random_seed(42)`, déterminisme quand disponible, configuration unique, identifiant de run, hashes du modèle/vocabulaire/dataset et métriques chargées depuis les artefacts du run. Répéter les expériences sur plusieurs graines et comparer aussi à des modèles simples de référence. Le [guide TensorFlow sur le déterminisme](https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism) détaille les conditions et limites.

**L'ordre de correction recommandé est d'abord de fiabiliser l'expérience.** Résoudre les annotations contradictoires et les commentaires révélateurs, traiter la perte de contexte, construire des partitions par groupes, puis figer un nouveau test. Ensuite seulement réentraîner le BiLSTM, comparer des représentations et modèles alternatifs sur validation, mesurer le rappel des vulnérabilités et calibrer les scores. Un modèle plus volumineux ou préentraîné ne corrigerait pas à lui seul les problèmes observés dans les données.

Les données et le modèle actuels constituent une expérience de classification binaire dont le score est reproductible. Les recouvrements d'entrées, les contradictions et la visibilité partielle du code limitent toutefois la portée de ce score pour des contrats nouveaux.
