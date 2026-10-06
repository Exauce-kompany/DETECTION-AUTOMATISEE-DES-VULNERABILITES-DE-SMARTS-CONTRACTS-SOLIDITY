from pathlib import Path
import sys, json
HERE=Path(__file__).resolve().parent
# Use matplotlib from the authorised bundled runtime.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse
ROOT=HERE.parents[1]
OUT=HERE/'figures'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':200})
BLUE='#2563EB'; TEAL='#0D9488'; PURPLE='#7C3AED'; ORANGE='#D97706'; RED='#DC2626'; GREY='#64748B'
PALETTE=[(BLUE,'#D6E9FF'),(TEAL,'#CFF4ED'),(PURPLE,'#E9DCFF'),(ORANGE,'#FFE5C2'),(RED,'#FFDADB'),('#0891B2','#CFF4FF')]
plt.rcParams['text.color']='#172B4D'

def stage_color(name,key,index):
    if name=='states':
        return {'a':PALETTE[0],'b':PALETTE[1],'c':PALETTE[2],'d':PALETTE[1],'e':PALETTE[4]}[key]
    return PALETTE[index%len(PALETTE)]
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),bbox_inches='tight',pad_inches=.12);plt.close(fig)
def diagram(name,boxes,edges,size=(9,4)):
    fig,ax=plt.subplots(figsize=size);ax.set_xlim(0,10);ax.set_ylim(0,6);ax.axis('off')
    for index,(key,(x,y,w,h,label)) in enumerate(boxes.items()):
        edge,fill=stage_color(name,key,index)
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06,rounding_size=.06',facecolor=fill,edgecolor=edge,lw=1.2))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10)
    for a,b in edges:
        x,y,w,h,_=boxes[a];xx,yy,ww,hh,_=boxes[b]
        if abs(y-yy)<.1:
            start=(x+w,y+h/2) if xx>x else (x,y+h/2)
            end=(xx,yy+hh/2) if xx>x else (xx+ww,yy+hh/2)
        else: start=(x+w/2,y);end=(xx+ww/2,yy+hh)
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',color=GREY,lw=1.4,shrinkA=4,shrinkB=4))
    save(fig,name)
diagram('lifecycle',{'a':(.2,4,2.6,1.1,'Code Solidity\nétat et fonctions'),'b':(3.7,4,2.6,1.1,'Compilation\net déploiement'),'c':(7.2,4,2.6,1.1,'Exécution EVM\ntransactions'),'d':(3.7,1.2,2.6,1.1,'Audit en amont\nSMART BUG')},[('a','b'),('b','c'),('a','d')],(9,3.6))
diagram('approaches',{'a':(.2,4,2.7,1.2,'Règles statiques\nmotifs et alertes'),'b':(3.65,4,2.7,1.2,'Réseaux séquentiels\nCNN / BiLSTM / TCN'),'c':(7.1,4,2.7,1.2,'Représentations riches\ngraphes / préentraînement'),'d':(3.65,1,2.7,1.4,'Décision et validation\nannotations, seuils,\ncontrats indépendants')},[('a','d'),('b','d'),('c','d')],(9,3.5))
diagram('pipeline',{'a':(.2,4.5,2.5,1,'Sources de contrats\net annotations'),'b':(3.7,4.5,2.5,1,'Nettoyage, groupes\net cinq partitions'),'c':(7.2,4.5,2.5,1,'Apprentissage\nvalidation / calibration'),'d':(7.2,1.3,2.5,1.3,'Modèle actif\nvocabulaire et seuil'),'e':(3.7,1.3,2.5,1.3,'Prédicteur commun\nAPI et commande'),'f':(.2,1.3,2.5,1.3,'Interface SMART BUG\nrapport et historique')},[('a','b'),('b','c'),('c','d'),('d','e'),('e','f')],(9,4))
# Runtime arrows are drawn separately in a clear left-to-right conceptual flow.
diagram('network',{'a':(.1,4.3,2.7,1.2,'Tokens normalisés\nvocabulaire 8 000\nfenêtres de 256'),'b':(3.65,4.3,2.7,1.2,'Embedding 32\nCNN : 24 × noyau 5\nmoyenne + maximum'),'c':(7.15,4.3,2.7,1.2,'Vecteur de fenêtre\n48 dimensions\nmasquage du remplissage'),'d':(7.15,1,2.7,1.4,'BiLSTM : 24 + 24\nsur les fenêtres\npooling global : 96'),'e':(3.65,1,2.7,1.4,'Dropout 0,30\nDense 32 ReLU\nlogit scalaire'),'f':(.1,1,2.7,1.4,'Température 0,93684\nseuil 0,26947\nune décision par contrat')},[('a','b'),('b','c'),('c','d'),('d','e'),('e','f')],(9,4.1))
# Correct reverse-direction links on second row.
def class_diagram():
    fig,ax=plt.subplots(figsize=(9,4.8));ax.set_xlim(0,10);ax.set_ylim(0,6);ax.axis('off')
    items=[(.15,3.2,4.1,2.5,'Contrat','code : texte\nnom : texte\ntaille : entier'),(5.55,3.2,4.2,2.5,'SmartContractPredictor','manifest_path ; model ; vocabulary\nload_resources()\npredict(code)'),(.15,.2,4.1,2.2,'Analyse','label ; probabilité ; verdict\nseuil ; compteurs de tokens\ndate ; code'),(5.55,.2,4.2,2.2,'« module » database','save_analysis(...)\nget_all_analyses()\ndelete_analysis(id)')]
    for index,(x,y,w,h,title,body) in enumerate(items):
        edge,fill=PALETTE[index]
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='square,pad=0',fc=fill,ec=edge,lw=1.2));ax.plot([x,x+w],[y+h-.6,y+h-.6],color=edge,lw=1)
        ax.text(x+w/2,y+h-.3,title,ha='center',va='center',weight='bold',fontsize=10);ax.text(x+.17,y+h-.85,body,va='top',fontsize=10,linespacing=1.65)
    for start,end,label in [((4.25,4.6),(5.55,4.6),'entrée'),((5.55,3.3),(4.25,2.1),'produit'),((4.25,1.4),(5.55,1.4),'stockage')]:
        ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':GREY});ax.text((start[0]+end[0])/2,(start[1]+end[1])/2+.12,label,ha='center',fontsize=9)
    save(fig,'classes')
class_diagram()
fig,ax=plt.subplots(figsize=(9,4));ax.set_xlim(0,10);ax.set_ylim(0,6);ax.axis('off')
ax.add_patch(plt.Circle((1.1,4.5),.23,fill=False,ec=BLUE));ax.plot([1.1,1.1],[4.27,3.35],color=BLUE);ax.plot([.5,1.7],[3.9,3.9],color=BLUE);ax.plot([.5,1.1,1.7],[2.8,3.35,2.8],color=BLUE);ax.text(1.1,2.4,'Utilisateur',ha='center')
for index,(y,t) in enumerate([(5,'Soumettre un contrat Solidity'),(3.6,'Consulter le verdict et les alertes'),(2.2,'Consulter les métriques du modèle'),(.8,'Consulter / supprimer l’historique')]):
    edge,fill=PALETTE[index]
    ax.add_patch(Ellipse((6.7,y),5.3,1,fc=fill,ec=edge));ax.text(6.7,y,t,ha='center',va='center',fontsize=10);ax.plot([1.8,4.05],[3.8,y],color=edge,lw=.8)
save(fig,'usecases')
fig,ax=plt.subplots(figsize=(9,4.3));ax.set_xlim(-.4,4.4);ax.set_ylim(0,7);ax.axis('off')
for x,t in enumerate(['Utilisateur','API','Prédicteur','Analyse statique','SQLite']):
    ax.text(x,6.7,t,ha='center',fontsize=10,weight='bold',color=PALETTE[x][0]);ax.plot([x,x],[.2,6.4],ls='--',color=PALETTE[x][0],alpha=.45)
for a,b,y,t in [(0,1,6,'fichier .sol'),(1,2,5.1,'code validé'),(2,1,4.2,'score calibré et label'),(1,3,3.3,'code pour règles'),(3,1,2.4,'alertes détaillées'),(1,4,1.5,'enregistrer'),(1,0,.6,'rapport séparant les scores')]:
    ax.annotate('',xy=(b,y),xytext=(a,y),arrowprops={'arrowstyle':'->','color':PALETTE[a][0]});ax.text((a+b)/2,y+.12,t,ha='center',fontsize=9)
save(fig,'sequence')
diagram('states',{'a':(.2,4,2.6,1,'En attente'),'b':(3.7,4,2.6,1,'Fichier validé'),'c':(7.2,4,2.6,1,'Analyses en cours'),'d':(7.2,1.1,2.6,1,'Rapport disponible'),'e':(3.7,1.1,2.6,1,'Erreur signalée')},[('a','b'),('b','c'),('c','d'),('b','e')],(9,2.8))
TR=ROOT/'results/training/smartbug-403bb881-20260922T101754Z'
ev=json.loads((TR/'evaluation.json').read_text())
hist=json.loads((TR/'seed-73-history.json').read_text())
for name,train,val,ylabel in [('loss','loss','val_loss','Perte binaire'),('accuracy','accuracy','val_accuracy','Exactitude')]:
    fig,ax=plt.subplots(figsize=(8,4));ax.plot(range(1,7),hist[train],'-o',color=BLUE,label='Entraînement');ax.plot(range(1,7),hist[val],'--s',color=ORANGE,label='Validation');ax.axvline(3,color='#bbbbbb',ls=':',label='Poids retenus : époque 3');ax.set(xlabel='Époque',ylabel=ylabel,xticks=range(1,7));ax.grid(alpha=.15);ax.legend(fontsize=10);fig.tight_layout();save(fig,name)
fig,ax=plt.subplots(figsize=(6,4.4));cm=ev['test']['neural_calibrated']['confusion_matrix'];ax.imshow(cm,cmap='Blues',vmin=0,vmax=850)
for i in range(2):
    for j in range(2):ax.text(j,i,str(cm[i][j]),ha='center',va='center',fontsize=24,color='white' if cm[i][j]>400 else 'black')
ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Absence de signal','Signal positif'],yticklabels=['Label 0','Label 1'],xlabel='Prédiction',ylabel='Annotation réelle');fig.tight_layout();save(fig,'confusion')
fig,ax=plt.subplots(figsize=(7,4));ax.plot([0,1],[0,1],':',color=GREY,label='Calibration idéale')
for k,c,t in [('neural_raw',ORANGE,'Brut'),('neural_calibrated',TEAL,'Calibré')]:
    bins=ev['test'][k]['reliability_bins'];bins=[b for b in bins if b['count']];ax.plot([b['predicted_probability'] for b in bins],[b['observed_positive_rate'] for b in bins],'-o',color=c,label=t)
ax.set(xlabel='Probabilité moyenne prédite',ylabel='Fréquence positive observée',xlim=(0,1),ylim=(0,1));ax.legend();ax.grid(alpha=.15);fig.tight_layout();save(fig,'reliability')
# Protocol diagrams replace score charts until all three models are evaluated.
diagram('comparison',{
'a':(3.5,4.8,3,0.8,'Mêmes contrats Solidity\nannotations et groupes'),
'b':(.15,2.5,2.9,1.35,'CNN–BiLSTM actif\napprentissage depuis zéro\ncheckpoint 73 validé'),
'c':(3.55,2.5,2.9,1.35,'TF-IDF + XGBoost\narbres appris depuis zéro\ncheckpoint 101 retenu'),
'd':(6.95,2.5,2.9,1.35,'TCN causal et dilaté\napprentissage depuis zéro\ncheckpoint validé'),
'e':(3.5,.3,3,0.95,'Une décision par contrat\ncalibration séparée')},[('a','b'),('a','c'),('a','d'),('b','e'),('c','e'),('d','e')],(9,4.5))
diagram('ablation',{
'a':(.1,4.6,3,1.05,'Entraînement : 11 959\napprendre les poids / arbres'),
'b':(3.5,4.6,3,1.05,'Validation : 1 709\nsélectionner les états'),
'c':(6.9,4.6,3,1.05,'Calibration : 1 709\nprobabilités et seuils'),
'd':(1.25,1.3,3.2,1.2,'Test : 1 709\nmesures communes\ndéjà consulté : exploratoire'),
'e':(5.55,1.3,3.2,1.2,'Source réservée : 1 152\ntransfert hors source\nseulement six négatifs')},[('a','b'),('b','c'),('c','d'),('c','e')],(9,4.1))
diagram('latency',{
'a':(.1,4.6,3,1.05,'Séquence encodée complète\ntoutes les positions\naucune fin supprimée'),
'b':(3.5,4.6,3,1.05,'TCN : 9 blocs résiduels\n2 convolutions par bloc\ndilatations 1 à 256'),
'c':(6.9,4.6,3,1.05,'Pooling moyen + maximum\nPAD masqués\nun logit et une perte'),
'd':(6.9,1.3,3,1.2,'Température\net seuil sur calibration\nchaque modèle séparément'),
'e':(3.5,1.3,3,1.2,'Test commun\nF1, rappel, précision, FPR\ncalibration et coût')},[('a','b'),('b','c'),('c','d'),('d','e')],(9,4.1))
print('15 figures created in',OUT)

# Replace protocol placeholders only after all measured outputs exist.
from matplotlib.ticker import PercentFormatter
snapshot=json.loads((HERE/'comparison_snapshot.json').read_text(encoding='utf-8'))
run=ROOT/'results/benchmark'/snapshot['run_id']
summary=json.loads((run/'summary.json').read_text(encoding='utf-8'))
import hashlib
assert summary['complete']
assert hashlib.sha256((run/'summary.json').read_bytes()).hexdigest()==snapshot['summary_sha256_at_import']
models=['cnn_bilstm','xgboost','tcn'];labels=['CNN–BiLSTM','XGBoost','TCN'];colors=['#0072B2','#009E73','#E69F00']
keys=['f1_macro','recall_vulnerable','false_positive_rate','average_precision']
fig,ax=plt.subplots(figsize=(9,4.3))
for i,(model,label,color) in enumerate(zip(models,labels,colors)):
    values=[100*summary['models'][model]['metrics'][key] for key in keys]
    positions=[j+(i-1)*.24 for j in range(4)]
    bars=ax.bar(positions,values,width=.23,color=color,label=label)
    ax.bar_label(bars,labels=[f'{v:.1f}'.replace('.',',') for v in values],padding=3,fontsize=8)
ax.set(xticks=range(4),xticklabels=['F1 macro','Rappel positif','FPR','PR-AUC (AP)'],ylim=(0,110),ylabel='Pourcentage')
ax.yaxis.set_major_formatter(PercentFormatter(100));ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
ax.legend(loc='upper center',bbox_to_anchor=(.5,1.16),ncol=3,frameon=False,fontsize=10)
fig.tight_layout();save(fig,'comparison')
names={'cnn_bilstm':'CNN','xgboost':'XGBoost','tcn':'TCN'}
pairs=summary['paired_comparisons']['pairs']
fig,ax=plt.subplots(figsize=(9,4.1))
for i,pair in enumerate(pairs):
    delta=pair['f1_macro'];value=100*delta['difference'];low,high=[100*v for v in delta['ci95']]
    ax.plot([low,high],[i,i],color=colors[i],lw=2.5);ax.plot(value,i,'o',color=colors[i],ms=8)
    ax.annotate(f'{value:+.2f} [{low:+.2f} ; {high:+.2f}]'.replace('.',','),(value,i),xytext=(0,14),textcoords='offset points',ha='center',fontsize=10)
ax.axvline(0,color=GREY,ls=':');ax.set(yticks=range(len(pairs)),yticklabels=[names[p['left']]+' − '+names[p['right']] for p in pairs],ylim=(-.55,len(pairs)-.35),xlabel='Différence de F1 macro (points) et IC à 95 %')
ax.invert_yaxis();ax.grid(axis='x',alpha=.15);fig.tight_layout();save(fig,'ablation')
selected=summary['models']['tcn']['selected'];seed=selected['seed']
history=json.loads((run/'tcn'/f'seed-{seed}'/'history.json').read_text(encoding='utf-8'))
fig,ax=plt.subplots(figsize=(9,4.1))
epochs=[row['epoch'] for row in history]
ax.plot(epochs,[row['loss'] for row in history],'-o',color=BLUE,label='Entraînement : BCE pondérée')
ax.plot(epochs,[row['validation_log_loss'] for row in history],'--s',color=ORANGE,label='Validation : log-loss non pondérée')
ax.axvline(selected['best_epoch'],color=GREY,ls=':',label='Époque retenue')
ax.set(xlabel='Époque',ylabel='Perte',xticks=epochs);ax.grid(alpha=.15);ax.legend(fontsize=10);fig.tight_layout();save(fig,'latency')

# Reuse fixed-size benchmark exports when pinned to the same saved summary.
exports=run/'plots'
manifest_path=exports/'plot_manifest.json'
if manifest_path.exists():
    import shutil
    import hashlib
    plot_manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    assert plot_manifest['status']=='complete'
    summary_key=(run/'summary.json').relative_to(ROOT).as_posix()
    assert plot_manifest['sources_sha256'][summary_key]==hashlib.sha256((run/'summary.json').read_bytes()).hexdigest()
    for plot,target in [('ecarts_f1_ic95','ablation'),('pertes_tcn_selectionne','latency')]:
        exported=exports/(plot+'.png')
        assert exported.exists()
        shutil.copy2(exported,HERE/'figures'/(target+'.png'))
