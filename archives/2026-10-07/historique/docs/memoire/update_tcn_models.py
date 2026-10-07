"""Prepare a faithful 44-page TCN revision without touching final outputs.

The staged builder writes only draft_documents. Publication is a separate step,
after the benchmark status has been confirmed and the rendered pages reviewed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN_ID = 'cnn-xgboost-tcn-20261005'
parser = argparse.ArgumentParser()
parser.add_argument('--stage', type=Path, default=HERE / 'tcn_staging')
parser.add_argument('--run', type=Path, default=ROOT / 'results/benchmark' / RUN_ID)
parser.add_argument('--status-note', default='préparation et pilote ; entraînement complet non terminé')
args = parser.parse_args()
stage = args.stage.resolve()
assert stage.is_relative_to(HERE.resolve()), 'Staging must stay inside memoire cache'
stage.mkdir(parents=True, exist_ok=True)

baseline = HERE / 'before_tcn_models'
if not baseline.exists():
    current_snapshot = json.loads((HERE / 'comparison_snapshot.json').read_text(encoding='utf-8'))
    if current_snapshot.get('status') == 'cnn_xgboost_tcn_evaluated':
        raise RuntimeError('This migration requires the original local before_tcn_models reference. Regenerate the published final edition directly with make_figures.py and build_memoire.py; do not migrate final sources again.')
    baseline.mkdir()
    for name in ('content.md', 'build_memoire.py', 'make_figures.py', 'comparison_snapshot.json', 'document_index.json', 'qa.json'):
        shutil.copy2(HERE / name, baseline / name)
    for extension, folder in (('docx', 'documents'), ('pdf', 'pdf')):
        name = 'Memoire_SMART_BUG_Exauce_Kompani.' + extension
        shutil.copy2(ROOT / 'output' / folder / name, baseline / name)

source = (baseline / 'content.md').read_text(encoding='utf-8')
text = source
summary_path = args.run / 'summary.json'
summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else None
tcn = (summary or {}).get('models', {}).get('tcn', {})
metrics = tcn.get('metrics')
complete = bool((summary or {}).get('complete') and metrics and tcn.get('selected'))
paired = (summary or {}).get('paired_comparisons')
def pct(value):
    return f'{100 * value:.2f}'.replace('.', ',')
status = 'trois modèles évalués' if complete else args.status_note
xgb_cnn_pair = next((pair for pair in (paired or {}).get('pairs', []) if pair['left'] == 'xgboost' and pair['right'] == 'cnn_bilstm'), None)
if xgb_cnn_pair:
    delta = xgb_cnn_pair['f1_macro']
    gain_text = f"Le gain de F1 est de {100 * delta['difference']:.2f} points ; le bootstrap par groupes donne un IC à 95 % de [{100 * delta['ci95'][0]:.2f} ; {100 * delta['ci95'][1]:.2f}] points, conditionné aux checkpoints choisis. ".replace('.', ',')
else:
    gain_text = 'Les intervalles de différence attendent les rééchantillonnages communs de cette campagne. '
performance = (f"Le TCN atteint {pct(metrics['f1_macro'])} % de F1 macro, "
               f"{pct(metrics['recall_vulnerable'])} % de rappel et "
               f"{pct(metrics['false_positive_rate'])} % de faux positifs.") if complete else (
               'Le TCN constitue le troisième modèle retenu ; son entraînement complet et son évaluation restent à terminer.')

def paragraph(prefix, replacement):
    global text
    pattern = r'^' + re.escape(prefix) + r'[^\n]*(?:\n(?!\n|@)[^\n]*)*'
    text, count = re.subn(pattern, lambda _: replacement, text, flags=re.M)
    assert count == 1, (prefix, count)

paragraph('Ce mémoire étudie', 'Ce mémoire étudie la détection automatisée des vulnérabilités des smart contracts Solidity par apprentissage profond. SMART BUG associe un réseau convolutif et un réseau récurrent bidirectionnel pour analyser le code complet. Le corpus nettoyé comprend 18 238 contrats ou représentations de contrats, répartis par groupes entre apprentissage, validation, calibration, test et source réservée. Le CNN–BiLSTM actif obtient un F1 macro de 82,10 % et un rappel positif de 88,95 % sur le test. Sur les mêmes contrats, XGBoost atteint 89,41 % de F1 macro. ' + performance + ' La comparaison reste exploratoire, car le test a déjà été observé. Ces résultats soutiennent une aide au tri des contrats à auditer, sous réserve des annotations et d’une validation indépendante.')
paragraph('L’objectif général consiste', 'L’objectif général consiste à concevoir et évaluer SMART BUG, un système de classification binaire associé à une application d’analyse. Les objectifs spécifiques sont de constituer un corpus traçable, de limiter les recouvrements entre partitions, de traiter les contrats sans supprimer leur fin et de calibrer les scores sur des données dédiées. La comparaison principale oppose le CNN–BiLSTM actif à XGBoost et à un TCN sur les mêmes contrats.')
paragraph('Trois hypothèses guident', 'Trois hypothèses guident la comparaison. Le CNN–BiLSTM pourrait apprendre des régularités locales et séquentielles utiles à la détection. XGBoost pourrait fournir une référence classique compétitive à partir des fréquences lexicales. Le TCN pourrait exploiter des dépendances de portée croissante grâce aux convolutions dilatées. Ces hypothèses demandent des mesures communes ; la supériorité d’une méthode n’est pas présumée.')
paragraph('La démarche associe', 'La démarche associe une revue de littérature ciblée, la préparation des données et une évaluation quantitative. L’entraînement apprend les poids ; la validation sélectionne les états ; la calibration règle les probabilités et les seuils. Le test et une source réservée servent à l’évaluation. ' + ('Les trois modèles ont été évalués.' if complete else 'Le CNN–BiLSTM et XGBoost sont évalués ; les résultats du TCN restent à compléter.') + ' La comparaison reste exploratoire, car le test avait déjà été consulté pendant le développement.')
paragraph('Les poids du modèle actif sont initialisés', 'Les poids du modèle actif et du TCN sont appris depuis zéro. Un TCN combine des convolutions causales dilatées et des connexions résiduelles (BAI, KOLTER et KOLTUN, 2018). Les dilations élargissent les voisinages accessibles à chaque position. Dans notre adaptation, le pooling global agrège toutes les positions valides du contrat ; conserver toute l’entrée ne signifie pas que chaque caractéristique locale couvre tout le contrat.')
paragraph('CodeBERT est un encodeur', 'BAI, KOLTER et KOLTUN (2018) étudient le TCN pour la modélisation de séquences. GOPALI et al. (2022) l’appliquent à des séquences d’opcodes EVM pour détecter les contrats vulnérables. Notre adaptation utilise des tokens source Solidity normalisés et un protocole propre ; les scores de ces publications ne sont pas repris. XGBoost apprend des arbres par boosting (CHEN et GUESTRIN, 2016) et fournit une référence classique.')
paragraph('Trois choix définissent', 'Trois choix définissent le positionnement de SMART BUG : conserver le contexte complet, séparer calibration et validation, et comparer le CNN–BiLSTM actif avec XGBoost et le TCN. La revue de littérature présente les démarches existantes. Les conclusions expérimentales reposent exclusivement sur les prédictions effectivement mesurées sur les mêmes contrats ; les architectures, représentations et budgets ne sont pas identiques.')
paragraph('Les poids sont appris depuis zéro', 'Les poids sont appris depuis zéro : aucun embedding ni encodeur préentraîné n’intervient dans le CNN–BiLSTM actif. L’ordre des fenêtres est disponible pour la récurrence, mais les agrégations compressent leur contenu. Le modèle estime une propriété globale du contrat et ne fournit pas directement une localisation. Son protocole de comparaison avec XGBoost et le TCN est présenté au troisième chapitre.')
paragraph('Les résultats actifs sont archivés', 'Les résultats actifs sont archivés dans results/training ; la comparaison cnn-xgboost-tcn-20261005, dans results/benchmark. Les résultats CNN et XGBoost sont réutilisés avec leur provenance et leurs empreintes. Les historiques, prédictions et manifestes permettent de recalculer les indicateurs. Les poids actifs sont désignés par models/active_model.json ; leur empreinte SHA-256 et celle du vocabulaire sont vérifiées.')
paragraph('La reproductibilité repose', 'La reproductibilité repose sur les graines 42, 73 et 101, les partitions, les configurations et les versions archivées. Des bibliothèques ou matériels différents peuvent modifier les calculs. Le commit 2df2580 décrit le CNN historique ; la comparaison TCN conserve ses propres manifestes et ne réécrit pas ces preuves.')
paragraph('L’expérimentation conserve', 'L’expérimentation conserve le CNN–BiLSTM actif sous TensorFlow, dont le checkpoint a été sélectionné sur validation. Elle réutilise XGBoost, appris depuis zéro sur TF-IDF avec les mêmes partitions. Le troisième modèle retenu est un TCN entraîné depuis zéro sur les tokens normalisés. ' + ('Son checkpoint a été sélectionné avant calibration et test.' if complete else 'Sa préparation et son pilote précèdent l’entraînement complet ; aucun score final n’est encore disponible.'))
paragraph('Les observations du modèle actif', 'Les observations du modèle actif proviennent des expériences de septembre 2026 ; XGBoost a été entraîné et évalué en octobre 2026. Ces résultats sont réutilisés, avec provenance, dans la comparaison TCN. Les schémas décrivent le protocole commun, les partitions et l’agrégation du TCN ; ils ne représentent aucune mesure simulée.')
if complete:
    text = text.replace('Les schémas décrivent le protocole commun, les partitions et l’agrégation du TCN ; ils ne représentent aucune mesure simulée.', 'Les graphiques comparatifs proviennent des sorties archivées : indicateurs du test, différences bootstrap et pertes du TCN sélectionné. Les schémas décrivent le système et les architectures.')
paragraph('Python porte la préparation', 'Python porte la préparation des données. TensorFlow/Keras implémente le CNN–BiLSTM actif ; XGBoost entraîne les arbres et PyTorch implémente le TCN. Scikit-learn fournit TF-IDF et les métriques. FastAPI expose les services ; HTML, CSS et JavaScript constituent l’interface, et SQLite conserve les analyses. Le protocole documente le matériel effectivement utilisé pour chaque calcul.')
paragraph('TF-IDF emploie', 'TF-IDF emploie unigrammes et bigrammes, au plus 20 000 caractéristiques, une fréquence minimale de deux documents et une fréquence de terme sous-linéaire. Son vocabulaire est appris sur l’entraînement uniquement. XGBoost a été entraîné avec les graines 42, 73 et 101 ; le TCN suit ces mêmes graines. Le CNN comparé reste le checkpoint actif de graine 73, sélectionné auparavant sur validation.')
text = text.replace('### 3.2.3 Comparaison du CNN–BiLSTM, de XGBoost et de CodeBERT', '### 3.2.3 Comparaison du CNN–BiLSTM, de XGBoost et du TCN')
paragraph('Le tableau présente les résultats', 'Le tableau présente les résultats du CNN–BiLSTM actif et du checkpoint XGBoost sélectionné sur validation. ' + ('La troisième ligne rapporte le checkpoint TCN retenu.' if complete else 'Le TCN reste en cours de préparation ; les cases à compléter indiquent une absence de résultat.') + ' Chaque ligne mesurée représente un checkpoint choisi sur validation, avec son seuil ajusté sur calibration.')
text = text.replace('@FIG comparison|3.5|Deux modèles évalués et un candidat supplémentaire.', '@FIG comparison|3.5|Comparaison commune du CNN–BiLSTM, de XGBoost et du TCN.')
paragraph('Les modèles classent chacun', 'Les modèles classent chacun le contrat complet. XGBoost exploite TF-IDF ; le TCN traite toutes les positions encodées. La log-loss de validation de XGBoost vaut 42 : 0,3244 ; 73 : 0,3271 ; 101 : 0,3197. Cette variabilité concerne la validation avant calibration ; le tableau n’est pas une moyenne des trois graines sur le test.')
paragraph('La validation choisit', 'La validation choisit les arbres ou le checkpoint TCN selon la log-loss non pondérée. La calibration ajuste la température et les seuils visant 90 % de rappel ou 10 % de faux positifs. Les taux observés sur test doivent aussi être rapportés. Un objectif fixé sur calibration n’est pas une garantie sur de nouveaux contrats.')
paragraph('Le protocole est décrit', 'Le protocole est décrit dans reports/protocole_cnn_xgboost_tcn.md et config/benchmark_models.json. Le script src/run_model_benchmark.py pilote les calculs. Le corpus, les identifiants et la perte au contrat restent communs. Les résultats CNN et XGBoost sont réutilisés ; les sorties TCN sont ajoutées après vérification de leur statut et de leur provenance.')
text = text.replace('### 3.2.5 Fine-tuning, calibration et coût de calcul', '### 3.2.5 Architecture TCN, calibration et coût de calcul')
paragraph('Le protocole CodeBERT-base', 'Le TCN comporte un embedding 32, neuf blocs résiduels de deux convolutions causales, 24 canaux, noyau 3 et dilations 1, 2, 4, 8, 16, 32, 64, 128 et 256. Une projection adapte 32 à 24 dimensions. Son champ réceptif est de 2 045 positions encodées : 1 + 2 × (3 − 1) × 511. Cette adaptation s’inspire de BAI, KOLTER et KOLTUN (2018).')
paragraph('Le lot effectif prévu', 'Le pooling moyen et maximum masqué agrège toutes les positions valides, sans tronquer la fin du contrat. Une couche dense 32 puis un logit scalaire produisent une seule perte binaire par contrat. Le dropout vaut 0,20. AdamW utilise un taux de 0,001 et une décroissance des poids de 0,0001 ; douze époques au plus et une patience de trois sont prévues pour chacune des graines 42, 73 et 101.')
text = text.replace('Le dropout vaut 0,20. AdamW utilise', 'Le dropout élémentaire vaut 0,20, après les convolutions et avant la tête ; la weight normalization n’est pas utilisée. AdamW utilise')
text = text.replace('pour chacune des graines 42, 73 et 101.', 'pour les graines 42, 73 et 101. Les lots regroupent au plus 32 contrats : budget 16 384 positions, sauf contrat plus long ; clipping 1,0.')
paragraph('Le pilote CPU de CodeBERT', ('Les durées TCN doivent être interprétées avec les budgets et le matériel documentés. Le checkpoint choisi sur validation est calibré séparément. Les coûts différents et la réutilisation de deux expériences historiques empêchent d’isoler un effet architectural à budget égal. Aucune hiérarchie générale des vitesses n’est établie.' if complete else 'Le TCN est en préparation et en cours de pilote ; ses durées complètes et ses scores ne sont pas encore disponibles. Le pilote sert à vérifier la faisabilité et ne constitue pas une comparaison de vitesse. Les budgets diffèrent et deux expériences historiques sont réutilisées : l’étude n’isole pas un effet architectural à budget égal.'))
paragraph('Le CNN–BiLSTM actif obtient', 'Le CNN–BiLSTM actif obtient 82,10 % de F1 macro, 88,95 % de rappel et 24,45 % de faux positifs. XGBoost atteint 89,41 %, 89,43 % et 10,61 % sur les mêmes contrats. ' + gain_text + (performance if complete else 'Le TCN n’est pas encore évalué.') + ' Ce test déjà observé reste exploratoire ; les scores ne sont pas des moyennes de graines sur test.')
paragraph('La validité externe demande', 'La validité externe demande un corpus indépendant représentatif des contrats futurs. Le test actuel a déjà été observé ; la comparaison demeure exploratoire. La normalisation des identifiants et le champ réceptif fini du TCN peuvent affecter le transfert. Les représentations, optimiseurs et budgets diffèrent ; un écart de score ne mesurerait donc pas uniquement l’effet de l’architecture.')
text = text.replace('La ROC AUC mesure la capacité de classement sur l’ensemble des seuils ; elle ne décrit pas à elle seule la qualité du seuil utilisé dans l’application.', 'La ROC AUC mesure le classement sur l’ensemble des seuils. La PR-AUC est calculée ici par average precision ; elle résume la précision en fonction du rappel. Ces indicateurs ne décrivent pas à eux seuls le seuil utilisé.')
text = text.replace('La suite du projet comptait 32 tests réussis lors de la vérification finale du code.', 'La vérification historique du modèle actif comptait 32 tests réussis. Les contrôles propres au TCN portent notamment sur les convolutions causales, le masquage et la reprise des poids.')
paragraph('Sur le plan expérimental, la priorité', 'Sur le plan expérimental, la priorité est ' + ('de reproduire la comparaison avec de nouveaux entraînements communs et un test indépendant.' if complete else 'd’achever les trois entraînements TCN, de sélectionner le checkpoint sur validation puis de le calibrer et de l’évaluer avec le même protocole.') + ' Les critères, seuils, coûts et variabilités doivent rester documentés. Une étude ultérieure pourrait examiner les graphes ou le préentraînement, accompagnés d’une évaluation indépendante.')
paragraph('Le modèle actif est entraîné depuis zéro. Sur 1 709', 'Le modèle actif est entraîné depuis zéro. Sur 1 709 exemples de test, il obtient un F1 macro de 82,10 % et un rappel positif de 88,95 %, avec 93 faux négatifs et 212 faux positifs. XGBoost atteint 89,41 % de F1 macro sur le même test. La comparaison principale confronte le CNN–BiLSTM, XGBoost et le TCN. ' + (performance if complete else 'Le TCN est retenu, mais son entraînement et son évaluation restent à terminer.') + ' La généralisation à d’autres corpus demande une validation indépendante.')
if complete:
    assert paired and len(paired['pairs']) == 3, 'Completed revision requires all paired comparisons'
    selected = tcn['selected']
    seed = selected['seed']
    selection = json.loads((args.run / 'tcn/selection.json').read_text(encoding='utf-8'))
    runs = selection['runs']
    assert len(runs) == 3 and {r['seed'] for r in runs} == {42, 73, 101}
    losses = ' ; '.join(f"{r['seed']} : {r['validation']['log_loss']:.4f}".replace('.', ',') for r in runs)
    paragraph('Les modèles classent chacun', f'Le TCN est retenu avec la graine {seed}, selon la log-loss de validation : {losses}. XGBoost conserve la graine 101 et le CNN la graine 73. Ces sélections portent sur la validation avant calibration ; les scores du tableau ne sont pas des moyennes des trois graines sur le test.')
    text = text.replace('@FIG comparison|3.5|Comparaison commune du CNN–BiLSTM, de XGBoost et du TCN.', '@FIG comparison|3.5|Indicateurs mesurés des trois checkpoints sur le test.')
    text = text.replace('@FIG ablation|3.6|Rôle des cinq partitions dans la comparaison.', '@FIG ablation|3.6|Différences de F1 macro et intervalles bootstrap par groupes.')
    text = text.replace('@FIG latency|3.7|Agrégation par contrat et calibration des décisions.', f'@FIG latency|3.7|Pertes du TCN retenu, graine {seed}.')
    paragraph('Le protocole est décrit', 'La figure 3.6 montre les écarts de F1 macro entre checkpoints sélectionnés, avec 1 000 rééchantillonnages par groupe, graine 42, et un IC à 95 %. Les intervalles sont exploratoires, conditionnés aux poids retenus et non corrigés pour comparaisons multiples. Le protocole et les sorties détaillées figurent dans reports/protocole_cnn_xgboost_tcn.md et results/benchmark/cnn-xgboost-tcn-20261005.')
    names = {'cnn_bilstm': 'CNN', 'xgboost': 'XGBoost', 'tcn': 'TCN'}
    comparisons = []
    for pair in paired['pairs']:
        delta = pair['f1_macro']
        low, high = [100 * value for value in delta['ci95']]
        comparisons.append(f"{names[pair['left']]} − {names[pair['right']]} : {100 * delta['difference']:+.2f} [{low:+.2f} ; {high:+.2f}]".replace('.', ','))
    paragraph('Le CNN–BiLSTM actif obtient', 'Les écarts de F1 macro et leurs IC à 95 %, en points, sont : ' + ' ; '.join(comparisons) + '. Ils décrivent le test déjà observé et les checkpoints sélectionnés. Ces mesures ne prouvent pas une supériorité générale ni une stabilité à chaque nouvel entraînement ; les conclusions doivent être confirmées sur un corpus indépendant.')
    paragraph('Les durées TCN doivent', f"La figure 3.7 oppose la BCE pondérée d’entraînement à la log-loss non pondérée de validation. La graine {seed} a exécuté {selected['epochs_executed']} époques ; l’époque {selected['best_epoch']} est retenue. Les durées sont murales par segments actifs ; attentes, suspensions et concurrence CPU peuvent y contribuer. La pause utilisateur, processus arrêté, est exclue. Les budgets différents et la réutilisation de deux expériences historiques empêchent d’isoler un effet architectural ou de comparer les vitesses.")
assert 'CodeBERT' not in text
assert re.findall(r'^@PAGE (.+)$', source, re.M) == re.findall(r'^@PAGE (.+)$', text, re.M)
def acknowledgments(s):
    return s[s.index('@PAGE ii\n'):s.index('@PAGE iv\n')]
assert acknowledgments(source) == acknowledgments(text)
(stage / 'content.md').write_text(text, encoding='utf-8')

old_snapshot = json.loads((baseline / 'comparison_snapshot.json').read_text(encoding='utf-8'))
snapshot = dict(old_snapshot)
snapshot['run_id'] = RUN_ID
snapshot['status'] = 'cnn_xgboost_tcn_evaluated' if complete else 'cnn_and_xgboost_evaluated_tcn_in_progress'
snapshot['third_model_status'] = status
snapshot['tcn_metrics'] = metrics
snapshot['tcn_selected'] = tcn.get('selected')
snapshot['reused_xgboost_run_id'] = old_snapshot['run_id']
if summary:
    assert summary['models']['xgboost']['metrics'] == snapshot['metrics'], 'Reused XGBoost metrics differ'
    snapshot['summary_source'] = str(summary_path.relative_to(ROOT)).replace('\\', '/')
    snapshot['summary_sha256_at_import'] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    snapshot['tcn_artifacts'] = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.run / 'tcn').glob('*.json')}
runtime_notes_path = args.run / 'runtime_notes.json'
if runtime_notes_path.exists():
    snapshot['runtime_notes_source'] = runtime_notes_path.relative_to(ROOT).as_posix()
    snapshot['runtime_notes_sha256'] = hashlib.sha256(runtime_notes_path.read_bytes()).hexdigest()
    snapshot['runtime_notes'] = json.loads(runtime_notes_path.read_text(encoding='utf-8-sig'))
(stage / 'comparison_snapshot.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding='utf-8')

builder = (baseline / 'build_memoire.py').read_text(encoding='utf-8')
builder = builder.replace('HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]', f'HERE=Path(__file__).resolve().parent;ROOT=Path({str(ROOT)!r})')
builder = builder.replace("OUT=ROOT/'output/documents'", "OUT=HERE/'draft_documents'")
builder = builder.replace("['Feng et al. (2020)','CodeBERT, encodeur préentraîné']", "['Bai et al. (2018)','TCN causal, dilaté et résiduel']")
builder = builder.replace("['Ferreira et al. (2020)','Framework SmartBugs']", "['Gopali et al. (2022)','TCN sur séquences d’opcodes EVM']")
builder = builder.replace('Comparaison CNN–BiLSTM / XGBoost / CodeBERT', 'Comparaison CNN–BiLSTM / XGBoost / TCN')
builder = builder.replace("['src/run_model_benchmark.py','Comparaison CNN–BiLSTM / XGBoost / TCN']", "['src/benchmark_tcn.py','TCN, pilote, entraînement et reprise'],['src/run_model_benchmark.py','Comparaison CNN–BiLSTM / XGBoost / TCN']")
if complete:
    builder = builder.replace('État de la comparaison à trois modèles.', 'Résultats comparatifs des trois checkpoints.')
builder = builder.replace("['Méthode','F1 macro (%)','Rappel (%)','FPR (%)'],[],[34,26,20,20]", "['Méthode','F1 macro (%)','Rappel (%)','FPR (%)','PR-AUC (%)'],[],[29,21,17,14,19]")
builder = builder.replace('PR-AUC (%)', 'PR-AUC (AP, %)')
builder = builder.replace("pct(fp/(tn+fp))])", "pct(fp/(tn+fp)),pct(r['average_precision'])])")
builder = builder.replace("pct(r['false_positive_rate'])])", "pct(r['false_positive_rate']),pct(r['average_precision'])])")
old = "TABLES['comparison'][3].append(['CodeBERT (candidat)','Calcul à compléter','Calcul à compléter','Calcul à compléter'])"
new = "tcn_metrics=snapshot.get('tcn_metrics')\nif tcn_metrics:\n    TABLES['comparison'][3].append(['TCN / tokens',pct(tcn_metrics['f1_macro']),pct(tcn_metrics['recall_vulnerable']),pct(tcn_metrics['false_positive_rate']),pct(tcn_metrics['average_precision'])])\nelse:\n    TABLES['comparison'][3].append(['TCN (en préparation)']+['Calcul à compléter']*4)"
assert old in builder
builder = builder.replace(old, new)
builder = builder.replace("('CodeBERT','Encodeur préentraîné pour le code et le langage naturel'),", '')
for item in ("('BERT','Bidirectional Encoder Representations from Transformers'),", "('BPE','Byte Pair Encoding : encodage en sous-tokens'),", "('CLS','Classification token : marqueur de représentation de fenêtre'),"):
    builder = builder.replace(item, '')
builder = builder.replace("ABBREVS.sort(key=lambda pair:pair[0].casefold())", "ABBREVS += [('TCN','Temporal Convolutional Network : réseau convolutif temporel'),('PAD','Padding : remplissage masqué des séquences'),('AP','Average precision : précision moyenne'),('COMPSAC','Computer Software and Applications Conference'),('EVM','Ethereum Virtual Machine : machine virtuelle Ethereum')]\nABBREVS=list(dict(ABBREVS).items())\nABBREVS.sort(key=lambda pair:pair[0].casefold())")
builder = builder.replace('Precision–Recall Area Under the Curve : aire précision-rappel', 'Precision–Recall Area Under the Curve ; mesurée ici par AP')
feng = "('FENG, Z. et al. (2020). CodeBERT: A Pre-Trained Model for Programming and Natural Languages. Findings of EMNLP, 1536–1547. DOI : 10.18653/v1/2020.findings-emnlp.139.','https://aclanthology.org/2020.findings-emnlp.139/','aclanthology.org/2020.findings-emnlp.139'),"
bai = "('BAI, S., KOLTER, J. Z. et KOLTUN, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. arXiv:1803.01271.','https://arxiv.org/abs/1803.01271','arxiv.org/abs/1803.01271'),"
gopali = "('GOPALI, S., KHAN, Z. A., CHHETRI, B., KARKI, B. et NAMIN, A. S. (2022). Vulnerability Detection in Smart Contracts Using Deep Learning. COMPSAC, 1249–1255. DOI : 10.1109/COMPSAC54236.2022.00197.','https://doi.org/10.1109/COMPSAC54236.2022.00197','doi.org/10.1109/COMPSAC54236.2022.00197'),"
assert feng in builder
builder = builder.replace(feng, bai + '\n' + gopali)
builder = builder.replace('SMART BUG : code, configurations, corpus préparé et résultats expérimentaux. Dépôt du mémoire, commit 2df2580.', 'SMART BUG : code et corpus de base, commit 2df2580 ; comparaison TCN du 5 octobre 2026, archives locales results/benchmark.')
builder = builder.replace('CNN et XGBoost : checkpoints sélectionnés sur validation. CodeBERT : candidat non entraîné.', 'CNN et XGBoost : checkpoints réutilisés ; TCN : ' + ('checkpoint sélectionné sur validation.' if complete else 'entraînement et évaluation à compléter.'))
builder = builder.replace('protocole comparatif ; évaluation CodeBERT à compléter.', 'protocole comparatif CNN–BiLSTM / XGBoost / TCN.')
if complete:
    builder = builder.replace('protocole comparatif CNN–BiLSTM / XGBoost / TCN.', 'résultats expérimentaux archivés, comparaison CNN / XGBoost / TCN.')
builder = builder.replace("'comparison_status':'cnn_and_xgboost_evaluated_codebert_incomplete'", "'comparison_status':snapshot['status']")
(stage / 'build_memoire.py').write_text(builder, encoding='utf-8')
figures = (baseline / 'make_figures.py').read_text(encoding='utf-8')
figures = figures.replace("sys.path.insert(0,str(HERE/'python_packages'))", '# Use matplotlib from the authorised bundled runtime.')
figures = figures.replace('ROOT=HERE.parents[1]', f'ROOT=Path({str(ROOT)!r})')
figures = figures.replace('Réseaux séquentiels\\nCNN / BiLSTM', 'Réseaux séquentiels\\nCNN / BiLSTM / TCN')
figures = figures.replace('graphes / CodeBERT', 'graphes / préentraînement')
figures = figures.replace('Prédicteur commun\\nAPI et ligne de commande', 'Prédicteur commun\\nAPI et commande')
figures = figures.replace('CodeBERT-base (candidat)\\nfine-tuning encodeur + tête\\nnon entraîné', 'TCN causal et dilaté\\napprentissage depuis zéro\\n' + ('checkpoint validé' if complete else 'calcul à compléter'))
figures = figures.replace('Une décision par contrat\\nTroisième modèle à choisir', 'Une décision par contrat\\ncalibration séparée')
figures = figures.replace('Code complet\\ntoutes les fenêtres\\naucune fin supprimée', 'Séquence encodée complète\\ntoutes les positions\\naucune fin supprimée')
figures = figures.replace('CodeBERT + tête CLS\\nlogit de chaque fenêtre\\nencodeur adapté', 'TCN : 9 blocs résiduels\\n2 convolutions par bloc\\ndilatations 1 à 256')
figures = figures.replace('Moyenne des logits\\nun logit par contrat\\nune seule perte', 'Pooling moyen + maximum\\nPAD masqués\\nun logit et une perte')
if complete:
    figures += '''\n# Replace protocol placeholders only after all measured outputs exist.\nfrom matplotlib.ticker import PercentFormatter\nsnapshot=json.loads((HERE/'comparison_snapshot.json').read_text(encoding='utf-8'))\nrun=ROOT/'results/benchmark'/snapshot['run_id']\nsummary=json.loads((run/'summary.json').read_text(encoding='utf-8'))\nassert summary['complete']\nmodels=['cnn_bilstm','xgboost','tcn'];labels=['CNN–BiLSTM','XGBoost','TCN'];colors=[TEAL,PURPLE,ORANGE]\nkeys=['f1_macro','recall_vulnerable','false_positive_rate','average_precision']\nfig,ax=plt.subplots(figsize=(9,4.3))\nfor i,(model,label,color) in enumerate(zip(models,labels,colors)):\n    values=[100*summary['models'][model]['metrics'][key] for key in keys]\n    positions=[j+(i-1)*.24 for j in range(4)]\n    bars=ax.bar(positions,values,width=.23,color=color,label=label)\n    ax.bar_label(bars,labels=[f'{v:.1f}'.replace('.',',') for v in values],padding=3,fontsize=8)\nax.set(xticks=range(4),xticklabels=['F1 macro','Rappel positif','FPR','PR-AUC'],ylim=(0,110),ylabel='Pourcentage')\nax.yaxis.set_major_formatter(PercentFormatter(100));ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)\nax.legend(loc='upper center',bbox_to_anchor=(.5,1.16),ncol=3,frameon=False,fontsize=10)\nfig.tight_layout();save(fig,'comparison')\nnames={'cnn_bilstm':'CNN','xgboost':'XGBoost','tcn':'TCN'}\npairs=summary['paired_comparisons']['pairs']\nfig,ax=plt.subplots(figsize=(9,4.1))\nfor i,pair in enumerate(pairs):\n    delta=pair['f1_macro'];value=100*delta['difference'];low,high=[100*v for v in delta['ci95']]\n    ax.plot([low,high],[i,i],color=colors[i],lw=2.5);ax.plot(value,i,'o',color=colors[i],ms=8)\n    ax.annotate(f'{value:+.2f} [{low:+.2f} ; {high:+.2f}]'.replace('.',','),(value,i),xytext=(0,14),textcoords='offset points',ha='center',fontsize=10)\nax.axvline(0,color=GREY,ls=':');ax.set(yticks=range(len(pairs)),yticklabels=[names[p['left']]+' − '+names[p['right']] for p in pairs],ylim=(-.55,len(pairs)-.35),xlabel='Différence de F1 macro (points) et IC à 95 %')\nax.invert_yaxis();ax.grid(axis='x',alpha=.15);fig.tight_layout();save(fig,'ablation')\nselected=summary['models']['tcn']['selected'];seed=selected['seed']\nhistory=json.loads((run/'tcn'/f'seed-{seed}'/'history.json').read_text(encoding='utf-8'))\nfig,ax=plt.subplots(figsize=(9,4.1))\nepochs=[row['epoch'] for row in history]\nax.plot(epochs,[row['loss'] for row in history],'-o',color=BLUE,label='Entraînement : BCE pondérée')\nax.plot(epochs,[row['validation_log_loss'] for row in history],'--s',color=ORANGE,label='Validation : log-loss non pondérée')\nax.axvline(selected['best_epoch'],color=GREY,ls=':',label='Époque retenue')\nax.set(xlabel='Époque',ylabel='Perte',xticks=epochs);ax.grid(alpha=.15);ax.legend(fontsize=10);fig.tight_layout();save(fig,'latency')\n'''
if complete:
    figures = figures.replace('colors=[TEAL,PURPLE,ORANGE]', "colors=['#0072B2','#009E73','#E69F00']")
    figures = figures.replace("xticklabels=['F1 macro','Rappel positif','FPR','PR-AUC']", "xticklabels=['F1 macro','Rappel positif','FPR','PR-AUC (AP)']")
    figures += '''
# Reuse fixed-size benchmark exports when pinned to the same saved summary.
exports=run/'plots'
manifest_path=exports/'plot_manifest.json'
if manifest_path.exists():
    import shutil
    import hashlib
    plot_manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    summary_key=(run/'summary.json').relative_to(ROOT).as_posix()
    assert plot_manifest['sources_sha256'][summary_key]==hashlib.sha256((run/'summary.json').read_bytes()).hexdigest()
    for plot,target in [('ecarts_f1_ic95','ablation'),('pertes_tcn_selectionne','latency')]:
        exported=exports/(plot+'.png')
        assert exported.exists()
        shutil.copy2(exported,HERE/'figures'/(target+'.png'))
'''
(stage / 'make_figures.py').write_text(figures, encoding='utf-8')
shutil.copytree(HERE / 'figures', stage / 'figures', dirs_exist_ok=True)
manifest = {'benchmark_run_id': RUN_ID, 'complete': complete, 'status': status, 'original_docx_sha256': hashlib.sha256((baseline / 'Memoire_SMART_BUG_Exauce_Kompani.docx').read_bytes()).hexdigest(), 'original_pdf_sha256': hashlib.sha256((baseline / 'Memoire_SMART_BUG_Exauce_Kompani.pdf').read_bytes()).hexdigest(), 'acknowledgments_source_exact': True, 'planned_pages': len(re.findall(r'^@PAGE ', text, re.M)), 'stage': str(stage)}
(stage / 'preparation.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(manifest, ensure_ascii=False, indent=2))
