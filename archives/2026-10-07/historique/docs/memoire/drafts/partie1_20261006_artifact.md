# Mémoire SMART BUG — contrat de réalisation

Référence de conception : C:/Users/Exauce/Downloads/mémoire EXAUCÉ KOMPANI.pdf (44 pages). Le PDF original reste intact. Adaptation mesurée vers DOCX, faute de source Word.

Plan : couverture ; résumé ; remerciements (texte et paragraphes inchangés, deux pages) ; dédicace ; sommaire ; figures ; tableaux ; abréviations ; introduction ; trois chapitres (Concepts théoriques de base / Architecture et modélisation / Investigation expérimentale) ; conclusion ; bibliographie.

Design : A4 ; marge gauche 98,9 pt, droite 98,9 pt ; haut 79 pt, bas 95 pt ; corps Palatino Linotype 12 pt (substitut Windows du Palladio du PDF), interligne 17,45 pt ; alinéa 12 pt ; espacement après 12 pt. Noir sur blanc. Titres de chapitre 24,8 pt ; premiers titres à 174 pt du haut environ. En-tête courant à droite, capitales italiques 10 pt, filet fin ; pied centré. Pagination romaine des liminaires, arabe du corps. Couverture centrée, sans logo inventé ; date septembre 2026 adaptée à l'expérience réelle. Trois chapitres et longueur cible 44 pages.

Exceptions justifiées : tableaux 9,5–10 pt et légendes 10 pt pour lisibilité ; titres de sections 16/13 pt ; acronymes en définitions compactes ; bibliographie 10,5 pt ; sommaire détaillé pour éviter la page quasi vide du modèle. Les diagrammes scientifiques sont créés d'après le code, les courbes d'après les JSON archivés ; aucun résultat simulé. Diagramme de classes explicitement conceptuel pour les modules procéduraux. Aucune capture d'écran inventée.

Contenu immuable : les remerciements, y compris la mention « mémoire de licence en droit », sont reproduits comme demandé. Dédicace conservée. Sujet scientifique remplacé par celui de l'utilisateur. Bibliographie adaptée et vérifiée sur sources primaires. Ne pas réintroduire l'ancienne appellation du modèle.

Preuves : config/model.json ; models/active_model.json ; dataset/benchmark/manifest.json ; results/training/smartbug-403bb881-20260922T101754Z ; results/comparison/comparison-v1-20260922 ; sources Python. Comparaison exploratoire distincte du modèle actif, encodeur CodeT5 figé, limites des annotations et du transfert explicitement documentées.

Livrables : output/documents/Memoire_SMART_BUG_Exauce_Kompani.docx et output/pdf/Memoire_SMART_BUG_Exauce_Kompani.pdf. Vérifier chaque page rendue, tableaux, figures, renvois et bibliographie ; vérifier textuellement les remerciements et les valeurs numériques.

## Vérification finale

44 pages confirmées dans Microsoft Word et dans le PDF rendu. Champs PAGE actualisés avec Word par automatisation, puis DOCX rendu à nouveau avec render_docx.py. Les 44 pages ont été inspectées ; les pages modifiées après actualisation ont été réinspectées. 15 figures, 8 tableaux, 12 références bibliographiques et 36 entrées d'abréviations. Table des matières et listes des figures/tableaux avec liens internes ; pagination contrôlée après export. Les données des graphiques viennent directement des JSON archivés. La légende d'ablation a été placée hors des barres.

Comparaison textuelle des deux pages de remerciements : identité exacte après normalisation des espaces et césures typographiques. Dédicace conservée. Aucun changement des données, poids, résultats ou fichiers source du logiciel. Nouveau mémoire enregistré localement en Word et PDF, sans publication distante demandée pour cette tâche. Détails des contrôles dans qa.json.

## Révision couleur demandée par l’utilisateur

Schémas avec fonds pastel et contours bleus, verts, violets, orange, roses et cyan ; états d’erreur rouges et états validés verts. Courbes entraînement/validation bleues et orange ; comparaison des architectures avec couleurs distinctes et identiques dans les graphiques de performance et de latence. Les couleurs complètent les libellés et les symboles existants.

Les images intégrées ont été remplacées sans modifier aucun fichier XML du DOCX : textes, champs, dimensions et mise en page sont identiques. Nouveau rendu dans render_color ; les 13 pages visuellement modifiées ont été inspectées entièrement. Les 31 autres pages sont pixel pour pixel identiques au rendu déjà inspecté. Les 44 pages conservent exactement leur texte extrait ; remerciements, pagination et renvois vérifiés. PDF final mis à jour dans output/pdf. Contrôles dans color_qa.json et qa.json.

## Révision expérimentale du 5 octobre 2026

La comparaison principale se fonde désormais sur les checkpoints CNN–BiLSTM et XGBoost effectivement évalués avec les mêmes partitions. CodeBERT demeure un candidat sans entraînement complet ni score ; le troisième modèle définitif reste à choisir. La revue de littérature reste qualitative, sans comparaison de ses scores avec ceux du projet. Les anciennes expériences CodeT5, CNN seul, BiLSTM seul, référence logistique et ablation à 512 tokens sont retirées de la discussion principale. Trois schémas en couleur décrivent les modèles, les partitions et l’agrégation/calibration plutôt que des barres simulées.

Preuves actuelles : results/benchmark/cnn-xgboost-codebert-20261005-final, config/benchmark_models.json, reports/protocole_cnn_xgboost_codebert.md. Le tableau 3.3 importe les métriques de summary.json, avec contrôle d’égalité à evaluation.json et des empreintes du checkpoint XGBoost. La graine 101, choisie sur la log-loss de validation, donne 89,41 % de F1 macro, 89,43 % de rappel et 10,61 % de faux positifs. Le CNN actif conserve 82,10 / 88,95 / 24,45 %. Les trois graines XGBoost décrivent une variabilité de validation ; le score test porte seulement sur le checkpoint choisi, sans moyenne de test inventée. L’intervalle de l’écart de F1 provient de paired_cnn_xgboost.json, bootstrap par groupes conditionné aux checkpoints sélectionnés. Le test, déjà observé auparavant, reste exploratoire. Le pilote CPU de CodeBERT justifie l’absence de score ; sa durée extrapolée n’est pas une mesure de vitesse comparative.

Rendu LibreOffice produit par le renderer du skill documents, avec rasterisation PyMuPDF faute de Poppler. Tous les passages modifiés ont été inspectés visuellement, puis chaque nouvelle modification réinspectée ; les autres pages sont identiques pixel pour pixel à un rendu déjà vérifié. Export final dans render_three_models_verified, recopié dans output/pdf. 44 pages, 15 figures, 8 tableaux, 13 références et 44 abréviations. Pagination, renvois, tableaux numériques et remerciements contrôlés ; texte des remerciements inchangé exactement dans les sources et identique après normalisation typographique dans le PDF. Vérifications dans qa.json. DOCX et PDF enregistrés localement ; aucune publication Git pour cette révision.
