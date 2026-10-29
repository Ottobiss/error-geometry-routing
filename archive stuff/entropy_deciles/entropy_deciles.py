"""Conditional mIoU of fixed baseline/Z/R/ZR pipelines on clean entropy bins."""
import os,sys,argparse,json,time
from pathlib import Path
sys.path.insert(0,os.environ['ZR_EXPERIMENTS'])
import experiment as m
import numpy as np
import pandas as pd
import torch

OUT=m.OUT/'entropy_deciles_v1';CACHE=m.ROOT/'entropy_deciles_cache_v1'
METHODS=m.METHODS+['b0','b5_full']
def patch_mean(value):
 h,w=value.shape;assert h%16==0 and w%16==0
 return value.reshape(h//16,16,w//16,16).mean((1,3))
def binned_conf(gt,pred,bins):
 valid=gt!=255;idx=bins[valid].astype(np.int64)*361+gt[valid].astype(np.int64)*19+pred[valid].astype(np.int64)
 return np.bincount(idx,minlength=10*361).reshape(10,19,19)
def prepare():
 OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
 m.require_acceptance();manifest=json.loads((m.OUT/'manifest.json').read_text());state=torch.load(m.OUT/'state.pt',map_location='cpu',weights_only=True)
 assert state['protocol_sha']==m.sha(m.OUT/'protocol.json')
 m.fixed(OUT/'protocol.json',dict(source_sha=m.sha(__file__),parent_protocol_sha=m.sha(m.OUT/'protocol.json'),state_sha=m.sha(m.OUT/'state.pt'),methods=METHODS,patch=16,bins='mean B0 full-working-size entropy per16patch; q10..q90 on cs_fit267, no GT',test='same498 frames; same blocked budget0.5 pipeline levels',macro='equal-weight mean of nonempty decile mIoUs, report bin counts; not global mIoU',status='exploratory conditional diagnostic; no tuning on test; no latency claim'))
 return manifest['fit'],manifest['test']

@torch.inference_mode()
def thresholds():
 fit,test=prepare();m.setup();engine=m.Engine();values=[]
 for i,r in enumerate(fit):
  path=CACHE/'clean'/(r['name']+'.npy');path.parent.mkdir(parents=True,exist_ok=True)
  if path.exists():values.append(np.load(path,allow_pickle=False));continue
  probs,_,_=engine.forward(m.small(r));ent=-(probs*probs.clamp_min(1e-12).log()).sum(0).cpu().numpy();a=patch_mean(ent).astype(np.float32);tmp=path.with_suffix('.npy.part')
  with tmp.open('wb') as f:np.save(f,a)
  tmp.replace(path);values.append(a)
  if (i+1)%20==0:print('Clean entropy',i+1,'/',len(fit),flush=True)
 cuts=np.quantile(np.concatenate([a.ravel() for a in values]),np.arange(1,10)/10)
 m.write(OUT/'thresholds.json',dict(cuts=cuts.tolist(),fit_frames=len(fit),ties='searchsorted side=right; repeated cuts can produce empty bins'))
 print('Entropy cuts:',cuts,flush=True)

@torch.inference_mode()
def evaluate():
 fit,test=prepare();m.setup();engine=m.Engine();m.load_state(engine);cuts=np.array(json.loads((OUT/'thresholds.json').read_text())['cuts'])
 with np.load(m.OUT/'blocked_confusions.npz',allow_pickle=False) as z:
  assert list(z['names'])==[r['name'] for r in test];levels=z['levels'];assert list(z['methods'])==m.METHODS
 start=time.monotonic()
 for i,r in enumerate(test):
  dest=CACHE/'test'/(r['name']+'.npz')
  if dest.exists():continue
  img=m.small(r);probs,maps,ent,knn=engine.all_scores(img)
  actual=np.array([ent,knn]+[float(maps[k].mean()) for k in m.KINDS])
  with np.load(m.CACHE/(r['name']+'.npz'),allow_pickle=False) as z:
   assert np.allclose(actual,z['scores'],rtol=0,atol=1e-6),f'Scores changed: {r["name"]}'
   expected=z['confs'].copy()
  entropy=-(probs*probs.clamp_min(1e-12).log()).sum(0).cpu().numpy();pe=patch_mean(entropy);pb=np.searchsorted(cuts,pe,side='right');bins=np.repeat(np.repeat(pb,16,0),16,1);gt=m.ground(r)
  predictions={0:probs.argmax(0).byte().cpu().numpy()}
  for index in sorted(set(levels[i].tolist()+[3])):
   if index not in predictions:predictions[index]=engine.predict(img,probs,m.LEVELS[m.MODES[index]])
  method_indices=levels[i].tolist()+[0,3];confs=[]
  for method,index in zip(METHODS,method_indices):
   c=binned_conf(gt,predictions[index],bins);assert np.array_equal(c.sum(0),expected[index]),f'Conditional/global mismatch: {r["name"]} {method}';confs.append(c)
  valid=(gt!=255).reshape(gt.shape[0]//16,16,gt.shape[1]//16,16).sum((1,3))>0
  m.save(dest,confs=np.stack(confs),patch_count=np.bincount(pb.ravel(),minlength=10),valid_patch_count=np.bincount(pb[valid],minlength=10),patch_scores=np.stack([pe]+[maps[k] for k in m.KINDS]))
  if (i+1)%5==0 or i+1==len(test):print('Decile evaluation',i+1,'/',len(test),'elapsed',round((time.monotonic()-start)/60,1),'min',flush=True)
 analyze()

def analyze():
 fit,test=prepare();cs=[];patches=[];validpatches=[]
 for r in test:
  with np.load(CACHE/'test'/(r['name']+'.npz'),allow_pickle=False) as z:cs.append(z['confs']);patches.append(z['patch_count']);validpatches.append(z['valid_patch_count'])
 cs=np.array(cs);patches=np.array(patches);validpatches=np.array(validpatches);names=np.array([r['name'] for r in test]);groups=np.array([r['sequence'] for r in test]);domains=np.array([r['domain'] for r in test]);cuts=np.array(json.loads((OUT/'thresholds.json').read_text())['cuts'])
 m.save(OUT/'decile_confusions.npz',confs=cs,names=names,groups=groups,domains=domains,methods=np.array(METHODS),cuts=cuts,patch_count=patches,valid_patch_count=validpatches)
 rows=[];macro=[]
 for domain in m.DOMAINS+['acdc_all','pool']:
  ix=np.ones(len(test),bool) if domain=='pool' else domains!='cs_test' if domain=='acdc_all' else domains==domain
  total=cs[ix].sum(0);valid=total[-2].sum((1,2));error=valid-np.trace(total[-2],axis1=1,axis2=2);nonempty=valid>0
  for j,method in enumerate(METHODS):
   values=[]
   for d in range(10):
    value=m.miou(total[j,d]) if nonempty[d] else np.nan;values.append(value)
    rows.append(dict(domain=domain,method=method,decile=d+1,miou=value,patches=int(patches[ix,d].sum()),valid_patches=int(validpatches[ix,d].sum()),valid_pixels=int(valid[d]),b0_error_pixels=int(error[d]),gt_classes=int((total[-2,d].sum(1)>0).sum()),frames_with_valid_pixels=int((cs[ix,-2,d].sum((1,2))>0).sum()),scope='fixed clean entropy bins; conditional, not a new primary metric'))
   macro.append(dict(domain=domain,method=method,mean_decile_miou=float(np.mean(np.array(values)[nonempty])),nonempty_deciles=int(nonempty.sum()),global_miou=m.miou(total[j].sum(0)),frames=int(ix.sum())))
 pd.DataFrame(rows).to_csv(OUT/'decile_miou.csv',index=False);pd.DataFrame(macro).to_csv(OUT/'macro_decile_miou.csv',index=False);print(pd.DataFrame(macro).query("domain in ['acdc_all','pool']").to_string(index=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['thresholds','evaluate','analyze']);a=p.parse_args();torch.set_num_threads(4)
 if a.stage!='analyze':assert torch.cuda.is_available(),'GPU required for entropy/conditional predictions'
 globals()[a.stage]()
