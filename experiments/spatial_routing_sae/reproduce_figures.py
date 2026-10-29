"""CPU-only validation, tables and figures from the archived confusion matrices."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
R=ROOT/'results'; F=ROOT/'figures'; F.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
COLORS={'entropy':'#727782','confidence':'#2262ad','confidence_N':'#16856b','confidence_NZ':'#8c45b4'}
LABELS={'entropy':'Entropy','confidence':'Confidence','confidence_N':'Confidence + N','confidence_NZ':'Confidence + N + Z'}
def miou(c):
 c=c.astype(float);d=np.diagonal(c,axis1=-2,axis2=-1);u=c.sum(-2)+c.sum(-1)-d
 return 100*np.divide(d,u,out=np.full_like(d,np.nan),where=u>0).mean() if (u>0).all() else 100*np.nanmean(np.divide(d,u,out=np.full_like(d,np.nan),where=u>0))
def save(fig,name):
 fig.tight_layout()
 for ext in ['svg','png','pdf']:fig.savefig(F/f'{name}.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig)
# All saved ACDC rows are recomputed from pixel counts, not averaged frame mIoU.
with np.load(R/'acdc_confusions.npz',allow_pickle=False) as z:
 c=z['confs']; names=z['names'];groups=z['groups'];methods=z['methods'].tolist();ks=z['ks'].tolist()
 domains=np.array([n.split('_')[0] for n in names])
 assert c.shape==(406,3,4,19,19) and len(set(names))==406 and len(set(groups))==21 and (c>=0).all()
 q=pd.read_csv(R/'acdc_quality.csv')
 for row in q.itertuples():
  ix=np.ones(len(names),bool) if row.domain=='all' else domains==row.domain
  assert abs(miou(c[ix,ks.index(row.k),methods.index(row.method)].sum(0))-row.miou)<1e-9
 q.pivot(index=['domain','k'],columns='method',values='miou').to_csv(R/'acdc_metrics_recomputed.csv')
 perclass=[]
 for j,k in enumerate(ks):
  for v,m in enumerate(methods):
   cm=c[:,j,v].sum(0).astype(float);d=np.diag(cm);u=cm.sum(0)+cm.sum(1)-d
   for cls,iou in enumerate(100*np.divide(d,u,out=np.full_like(d,np.nan),where=u>0)):
    perclass.append({'k':k,'method':m,'train_id':cls,'iou':iou})
 pd.DataFrame(perclass).to_csv(R/'acdc_class_iou.csv',index=False)
with np.load(R/'oof_confusions.npz',allow_pickle=False) as z:
 c=z['confs'];ms=z['methods'].tolist();ks=z['ks'].tolist();cv=pd.read_csv(R/'cv_summary.csv')
 for row in cv.itertuples():assert abs(np.mean([miou(c[:,ks.index(row.k),s,ms.index(row.method)].sum(0)) for s in range(3)])-row.mean_miou)<1e-9
# Quality vs selected windows: no time is inferred from window count.
fig,axes=plt.subplots(1,2,figsize=(11,4))
for m in methods:
 a=cv[cv.method==m];axes[0].plot(a.k,a.mean_miou,'o-',color=COLORS[m],label=LABELS[m])
 a=q[(q.domain=='all')&(q.method==m)];axes[1].plot(a.k,a.miou,'o-',color=COLORS[m],label=LABELS[m])
for ax,title in zip(axes,['Cityscapes P0: grouped OOF, 3 seeds','ACDC: 406 frames, frozen seed 0']):
 ax.set(title=title,xlabel='Selected windows per frame',ylabel='mIoU (%)',xticks=[1,2,3]);ax.grid(alpha=.2)
axes[1].legend(fontsize=8);save(fig,'quality_vs_windows')
fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
for ax,k in zip(axes,[1,2,3]):
 a=q[q.k==k].pivot(index='domain',columns='method',values='miou').loc[['fog','night','rain','snow']]
 for m in methods:ax.plot(range(4),a[m],'o-',color=COLORS[m],label=LABELS[m])
 ax.set(title=f'ACDC, k={k}',xticks=range(4),xticklabels=a.index,ylabel='mIoU (%)');ax.grid(alpha=.2)
axes[-1].legend(fontsize=8);save(fig,'quality_by_weather')
fig,ax=plt.subplots(figsize=(9,4))
a=q[q.domain=='all'].pivot(index='k',columns='method',values='miou')
for j,m in enumerate(methods[:-1]):
 d=a[m]-a.entropy;ax.bar(np.arange(3)+(j-1)*.24,d,width=.24,color=COLORS[m],label=LABELS[m])
ax.axhline(0,color='black',lw=.8);ax.set(xticks=range(3),xticklabels=['1','2','3'],xlabel='Windows per frame',ylabel='Delta vs entropy (mIoU pp)',title='ACDC: measured differences at the same number of windows');ax.legend();save(fig,'delta_vs_entropy')
bs=pd.read_csv(R/'acdc_bootstrap.csv');a=bs[(bs.domain=='all')&(bs.k==1)].reset_index(drop=True)
fig,ax=plt.subplots(figsize=(10,4))
for i,r in a.iterrows():ax.plot([r.ci_low,r.ci_high],[i,i],color='#2262ad',lw=2);ax.scatter(r.delta_pp,i,color='#2262ad')
ax.axvline(0,color='black',lw=.8);ax.set(yticks=range(len(a)),yticklabels=a.comparison.str.replace(' minus ',' − '),xlabel='Delta mIoU (pp), saved 95% sequence bootstrap',title='ACDC, k=1: 21 sequences');save(fig,'primary_comparisons')
fig,ax=plt.subplots(figsize=(8,4))
for m in methods[:-1]:
 h=json.loads((R/f'router_{m}.json').read_text());ax.plot([r['epoch'] for r in h],[r['train_loss'] for r in h],label=LABELS[m],color=COLORS[m])
ax.set(xlabel='Epoch',ylabel='Smooth L1 training loss',title='Final routers: training on all 300 Cityscapes P0 frames');ax.legend();save(fig,'training_loss')
# Historical timing is explicitly a separate experiment.
a=pd.DataFrame([['entropy',53.505973,475.509205],['base',53.717237,487.214145],['norm',54.120435,488.903998],['norm_residual',53.811583,491.274089],['b5_full',57.486238,398.888320]],columns=['method','miou','mean_ms'])
a.to_csv(R/'historical_spatial_timing_k3.csv',index=False)
fig,ax=plt.subplots(figsize=(8,4))
offsets=[(-75,-22),(-70,25),(-50,35),(8,-16),(8,-14)]
for r,offset in zip(a.itertuples(),offsets):
 ax.scatter(r.mean_ms,r.miou,s=65);ax.annotate(r.method,(r.mean_ms,r.miou),xytext=offset,textcoords='offset points',arrowprops={'arrowstyle':'-','color':'#666'})
ax.set(xlabel='Measured whole-pipeline mean time (ms)',ylabel='mIoU (%)',title='Earlier 2-block experiment only: k=3, T4; timing on 40 frames');ax.margins(.18);save(fig,'historical_quality_time')
checks={'archive_acdc_frames':406,'acdc_sequences':21,'cityscapes_frames':300,'acdc_csv_matches_confusions':True,'cv_csv_matches_confusions':True,'source_sha_matches_protocol':hashlib.sha256((ROOT/'source/allblock_router.py').read_bytes()).hexdigest()==json.loads((R/'protocol.json').read_text())['script_sha'],'gpu_run_performed_for_report':False}
assert checks['source_sha_matches_protocol']
(R/'report_validation.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
# Context ablation: historical, different experiment, no cross-protocol comparisons.
ctx=pd.DataFrame({'budget':[.1,.2],'B0':[73.647362]*2,'true_mid':[75.383037,76.431929],'true_four':[75.916776,76.929944],'full_tiles':[76.481198,77.556035]})
ctx.to_csv(R/'historical_context_quality.csv',index=False)
fig,ax=plt.subplots(figsize=(8,4))
for m in ['B0','true_mid','true_four','full_tiles']:ax.plot(ctx.budget*100,ctx[m],'o-',label=m)
ax.set(xlabel='Selected tile budget (%)',ylabel='mIoU (%)',xticks=[10,20],title='E8c: true B5 context states, cs_test 233 frames');ax.legend();save(fig,'historical_context')
# Overview in an editable SVG, without rasterized text.
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
fig,ax=plt.subplots(figsize=(11,7));ax.set(xlim=(0,11),ylim=(0,7));ax.axis('off')
def node(x,y,w,h,title,body,color):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.12',facecolor=color,edgecolor='#b5c3d2'))
 ax.text(x+w/2,y+h*.69,title,ha='center',va='center',weight='bold',fontsize=11)
 ax.text(x+w/2,y+h*.28,body,ha='center',va='center',fontsize=9)
def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color='#566780',lw=1.4))
node(.3,5.3,2.3,1,'RGB frame','CS 1024x512 / ACDC 1024x576','#eef3f9')
node(3.3,5.3,2.6,1,'Frozen B0','Probabilities + 8 Mix-FFN taps','#dceaff')
node(6.8,5.3,3.6,1,'Feature variants','Confidence / +12 N / +512 Z','#eee4fa')
node(6.8,3.1,3.6,1.2,'Gain SpatialRouter','548x16x16 + 20 global -> score/window','#eee4fa')
node(3.3,3.1,2.6,1.2,'Top-k windows','k = 1 / 2 / 3; stable ranking','#fff0d4')
node(.3,3.1,2.3,1.2,'Frozen B5 crops','Context: 512x512 working grid','#dff1e7')
node(3.3,.7,4.4,1.2,'Merged segmentation','Replace selected window; keep B0 elsewhere','#dff1e7')
for a,b in [((2.7,5.8),(3.2,5.8)),((6,5.8),(6.7,5.8)),((8.6,5.2),(8.6,4.4)),((6.7,3.7),(6,3.7)),((3.2,3.7),(2.7,3.7)),((1.45,3),(3.8,2))]:arrow(a,b)
from matplotlib.path import Path as PlotPath
route=PlotPath([(4.6,5.2),(4.6,4.75),(6.4,4.75),(6.4,2.0)],[PlotPath.MOVETO,PlotPath.LINETO,PlotPath.LINETO,PlotPath.LINETO])
ax.add_patch(FancyArrowPatch(path=route,arrowstyle='-|>',mutation_scale=13,color='#566780',lw=1.4))
ax.text(6.5,2.5,'B0 mask',fontsize=8,ha='left')
ax.text(5.5,6.8,'SPATIAL ROUTING: FROZEN SEGMENTATORS, LEARNED WINDOW GAIN',ha='center',fontsize=13,weight='bold')
save(fig,'pipeline')
print('Figures complete.')
