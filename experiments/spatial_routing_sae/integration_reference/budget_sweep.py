"""Read cached levels; sweep unchanged router budgets; time live paths separately."""
import os,sys,argparse,json,copy,time
from pathlib import Path
sys.path.insert(0,os.environ['ZR_EXPERIMENTS'])
import experiment as m
import numpy as np
import pandas as pd
import torch
from router import Router,LEVELS

BUDGETS=[.1,.25,.5,.75,1.]
OUT=m.OUT/'budget_sweep_v1'
def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((m.OUT/'manifest.json').read_text())['test']
 state=torch.load(m.OUT/'state.pt',map_location='cpu',weights_only=True)
 assert state['protocol_sha']==m.sha(m.OUT/'protocol.json')
 m.require_acceptance()
 selected=[]
 for d in m.DOMAINS:
  ix=[i for i,r in enumerate(manifest) if r['domain']==d]
  selected += [ix[j] for j in np.linspace(0,len(ix)-1,min(8,len(ix))).astype(int)]
 m.fixed(OUT/'protocol.json',dict(source_sha=m.sha(__file__),parent_protocol_sha=m.sha(m.OUT/'protocol.json'),state_sha=m.sha(m.OUT/'state.pt'),budgets=BUDGETS,methods=m.METHODS,order='blocked',timing_names=[manifest[i]['name'] for i in selected],repeats=2,quality='all498 common frames; thresholds fixed on cs_fit',timing='fixed40 frames, blocked stream history; CPU decode to CPU mask; not population mean',status='exploratory budget sweep; do not select/refit on evaluation'))
 scores=[];confs=[]
 for r in manifest:
  with np.load(m.CACHE/(r['name']+'.npz'),allow_pickle=False) as z:scores.append(z['scores']);confs.append(z['confs'])
 return manifest,state,selected,np.array(scores),np.array(confs)
def routers(state,budget):
 result={}
 for method in m.METHODS:
  r=Router(budget,m.QUANTILE,m.WINDOW)
  if method!='entropy':r.threshold=state['thresholds'][method]
  result[method]=r
 return result

def gap(value,light,heavy):
 den=heavy-light
 return float(100*(value-light)/den) if abs(den)>1e-6 else np.nan

def quality():
 manifest,state,selected,scores,all_conf=prepare();domains=np.array([r['domain'] for r in manifest]);names=np.array([r['name'] for r in manifest]);summary=[]
 for budget in BUDGETS:
  rs=routers(state,budget);cs=np.zeros((len(manifest),len(m.METHODS),19,19),np.int64);levels=np.zeros((len(manifest),len(m.METHODS)),int);cost=levels.astype(float)
  for i,r in enumerate(manifest):
   for j,method in enumerate(m.METHODS):
    level=rs[method].decide(m.flag(scores[i],method),float(scores[i,0]));index=m.MODES.index(level.name);levels[i,j]=index;cost[i,j]=level.cost;cs[i,j]=all_conf[i,index]
  m.save(OUT/f'budget_{budget:g}_confusions.npz',confs=cs,cost=cost,levels=levels,names=names,methods=np.array(m.METHODS))
  for domain in m.DOMAINS+['acdc_all','pool']:
   ix=np.ones(len(manifest),bool) if domain=='pool' else domains!='cs_test' if domain=='acdc_all' else domains==domain
   light=m.miou(all_conf[ix,0].sum(0));heavy=m.miou(all_conf[ix,-1].sum(0))
   for j,method in enumerate(m.METHODS):
    value=m.miou(cs[ix,j].sum(0));summary.append(dict(domain=domain,method=method,budget=budget,miou=value,miou_b0=light,miou_b5=heavy,gain_vs_b0_pp=value-light,gap_closed_percent=gap(value,light,heavy),mean_b5_area_cost=float(cost[ix,j].mean()),frames=int(ix.sum())))
  print('Quality budget',budget,'done',flush=True)
 for domain in m.DOMAINS+['acdc_all','pool']:
  ix=np.ones(len(manifest),bool) if domain=='pool' else domains!='cs_test' if domain=='acdc_all' else domains==domain
  light=m.miou(all_conf[ix,0].sum(0));heavy=m.miou(all_conf[ix,-1].sum(0))
  for method,value,budget in [('b0',light,0.),('b5_full',heavy,1.)]:summary.append(dict(domain=domain,method=method,budget=budget,miou=value,miou_b0=light,miou_b5=heavy,gain_vs_b0_pp=value-light,gap_closed_percent=gap(value,light,heavy),mean_b5_area_cost=budget,frames=int(ix.sum())))
 pd.DataFrame(summary).to_csv(OUT/'quality.csv',index=False)
 # Preserve reference confusions for later gap/bootstrap without the big local cache.
 m.save(OUT/'reference_confusions.npz',light=all_conf[:,0],heavy=all_conf[:,-1],names=names,groups=np.array([r['sequence'] for r in manifest]))
 print(pd.DataFrame(summary).query("domain=='pool'").to_string(index=False))

@torch.inference_mode()
def timing():
 assert torch.cuda.is_available(),'GPU required for real timing'
 m.setup();manifest,state,selected,scores,all_conf=prepare();engine=m.Engine();m.load_state(engine)
 keys=[(b,method) for b in BUDGETS for method in m.METHODS]+[(0.,'b0'),(1.,'b5_full')]
 snapshots={};decisions={}
 for b in BUDGETS:
  rs=routers(state,b)
  for i in range(len(manifest)):
   for method in m.METHODS:
    if i in selected:snapshots[b,i,method]=copy.deepcopy(rs[method])
    level=rs[method].decide(m.flag(scores[i],method),float(scores[i,0]))
    if i in selected:decisions[b,i,method]=m.MODES.index(level.name)
 def call(r,method,router):
  torch.cuda.synchronize();start=time.perf_counter();img=m.small(r)
  if method=='b5_full':answer=engine.heavy_probs(img,1.).argmax(0).byte().cpu().numpy();index=3
  elif method=='b0':answer=engine.b0(img[None].cuda(),tuple(r['size'])).softmax(1).argmax(1)[0].byte().cpu().numpy();index=0
  else:
   kind=method.split('_')[0] if method.split('_')[0] in m.KINDS else None
   probs,out,ent=engine.forward(img,kind)
   fl=float(out['score'].mean()) if kind else float(engine.knn(engine.native())) if method=='knn_entropy' else None
   decision=router.decide(fl,ent);answer=engine.predict(img,probs,decision);index=m.MODES.index(decision.name)
  torch.cuda.synchronize();return answer,index,(time.perf_counter()-start)*1000
 for b,method in keys:call(manifest[selected[0]],method,copy.deepcopy(snapshots.get((b,selected[0],method))))
 for pos,i in enumerate(selected):
  dest=OUT/'timing_records'/(manifest[i]['name']+'.json')
  rec=json.loads(dest.read_text()) if dest.exists() else {}
  tasks=[(b,method,rep) for b,method in keys for rep in range(2)];np.random.default_rng(100000+i).shuffle(tasks)
  gt=None
  for b,method,rep in tasks:
   key=f'{b:g}|{method}|{rep}'
   if key in rec:continue
   answer,index,ms=call(manifest[i],method,copy.deepcopy(snapshots.get((b,i,method))))
   expected=0 if method=='b0' else 3 if method=='b5_full' else decisions[b,i,method]
   assert index==expected,f'Live routing mismatch: {manifest[i]["name"]} {b} {method}'
   if gt is None:gt=m.ground(manifest[i])
   assert np.array_equal(m.conf(gt,answer),all_conf[i,index]),f'Live prediction mismatch: {manifest[i]["name"]} {method}'
   rec[key]=float(ms);m.write(dest,rec)
  print('Timing sweep',pos+1,'/',len(selected),flush=True)
 summary=[]
 for b,method in keys:
  times=[]
  for i in selected:
   rec=json.loads((OUT/'timing_records'/(manifest[i]['name']+'.json')).read_text());times += [rec[f'{b:g}|{method}|{rep}'] for rep in range(2)]
  summary.append(dict(method=method,budget=b,mean_ms=float(np.mean(times)),p95_ms=float(np.percentile(times,95)),timing_frames=len(selected),repeats=2))
 pd.DataFrame(summary).to_csv(OUT/'timing.csv',index=False)
 q=pd.read_csv(OUT/'quality.csv');combined=q.merge(pd.DataFrame(summary),on=['method','budget'],validate='many_to_one');combined.to_csv(OUT/'curve_metrics.csv',index=False);print(combined.query("domain=='pool'").to_string(index=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['quality','timing']);a=p.parse_args();torch.set_num_threads(4);globals()[a.stage]()
