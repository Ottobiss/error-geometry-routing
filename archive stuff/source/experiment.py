"""Z/R/ZR scorer integration: exact SAE scorer, unchanged Ottobiss Router rule.
Controlled FP32 rerun; working sizes follow SAE, not published main numbers.
"""
import os
os.environ['USE_TF']='0'
import argparse,json,hashlib,time,copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scorer import ErrorScorer,F0
from sae_lib import load_model,output_features
from router import Router,LEVELS
from features.knn import KNN
from features.entropy import Entropy

ROOT=Path(os.environ['ZR_ROOT']);OUT=Path(os.environ['ZR_OUT']);ART=Path(os.environ['ZR_ART']);CS=Path(os.environ['ZR_CS']);ACDC=Path(os.environ['ZR_ACDC']);CACHE=ROOT/'cache'
KINDS=['Z','R','ZR'];METHODS=['entropy','knn_entropy','Z_entropy','R_entropy','ZR_entropy'];MODES=['light','ens@0.5','ens@0.75','heavy'];DOMAINS=['cs_test','fog','night','rain','snow'];BUDGET=.5;QUANTILE=.95;WINDOW=50
LUT=np.full(256,255,np.uint8)
for a,b in {7:0,8:1,11:2,12:3,13:4,17:5,19:6,20:7,21:8,22:9,23:10,24:11,25:12,26:13,27:14,28:15,31:16,32:17,33:18}.items():LUT[a]=b

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.part');t.write_text(json.dumps(v,indent=2));t.replace(p)
def fixed(p,v):
 if p.exists():assert json.loads(p.read_text())==v,f'Inputs/code changed: {p}. Use a new output directory.'
 else:write(p,v)
def save(p,**kw):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.npz.part')
 with t.open('wb') as f:np.savez_compressed(f,**kw)
 t.replace(p)
def conf(gt,p):
 ok=gt!=255;return np.bincount(gt[ok].astype(np.int64)*19+p[ok].astype(np.int64),minlength=361).reshape(19,19)
def miou(c):
 c=c.astype(float);d=np.diag(c);u=c.sum(0)+c.sum(1)-d;return float(100*(d[u>0]/u[u>0]).mean())
def csv(name,rs):
 frame=pd.DataFrame(rs);frame['reference_reproduced']=False;frame['evaluation_mode']='local_scorer_after_failed_acceptance';frame.to_csv(OUT/name,index=False)

def inventory():
 fit=[];test=[]
 for city in ['frankfurt','lindau','munster']:
  files=sorted((CS/'leftImg8bit/val'/city).glob('*_leftImg8bit.png'));assert files,f'No Cityscapes {city}'
  for p in files:
   name=p.name[:-len('_leftImg8bit.png')];gt=CS/'gtFine/val'/city/(name+'_gtFine_labelIds.png');assert gt.is_file(),str(gt)
   r=dict(name=name,domain='cs_fit' if city=='frankfurt' else 'cs_test',sequence='_'.join(name.split('_')[:2]),rgb=str(p),gt=str(gt),size=[1024,512],cityscapes=True)
   (fit if city=='frankfurt' else test).append(r)
 p0=set(Path(__file__).with_name('p0_ids.txt').read_text().split());assert len(p0)==300
 test=[r for r in test if r['name'] not in p0]
 assert not any(r['name'] in p0 for r in test)
 for d in ['fog','night','rain','snow']:
  files=sorted((ACDC/'rgb_anon'/d/'val').rglob('*_rgb_anon.png'));assert files,f'No ACDC {d}'
  for p in files:
   stem=p.name[:-len('_rgb_anon.png')];gt=ACDC/'gt'/d/'val'/p.parent.name/(stem+'_gt_labelTrainIds.png');assert gt.is_file(),str(gt)
   test.append(dict(name=d+'_'+stem,domain=d,sequence=d+'/'+p.parent.name,rgb=str(p),gt=str(gt),size=[1024,576],cityscapes=False))
 assert len({r['name'] for r in fit+test})==len(fit)+len(test)
 assert len(fit)==267 and len(test)==498,(len(fit),len(test))
 return fit,test

def setup():
 OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
 for k in KINDS:
  for n in ['model.cbm','scores.parquet']:assert (ART/'runs'/(k+'_ae_b8_catboost')/n).is_file(),f'Missing {k}/{n}'
 weights=ART/'ckpt/ae_b8'
 assert len(list(weights.glob('s*b*.pt')))==8
 fit,test=inventory()
 fixed(OUT/'protocol.json',dict(schema='ottobiss-SAE-local-nonreproduced-v1',reference_reproduced=False,evaluation_mode='local_scorer_after_failed_acceptance',baseline_commit='8eb70cae33a3875a2588a350719ac6002d77347f',methods=METHODS,budget=BUDGET,quantile=QUANTILE,window=WINDOW,flag='mean full-frame scorer map; fixed clean-data q95',priority='native B0 logit entropy as baseline',B0='fp32 exact SAE SegModel',B5='fp32 common to all methods',sizes='CS 1024x512; ACDC 1024x576 per SAE guide',inference='full working-size probabilities before ensemble; controlled rerun, not published main values',cost='scale squared for B5 only; not latency or total FLOPs',primary='ZR_entropy minus knn_entropy, blocked stream',secondary='Z/R vs knn; all AE variants vs entropy; 5 fixed shuffled orders',selection='no best-method selection on evaluation; report all',bootstrap='2000 sequence and frame resamples of fixed predictions, conditional on realized routing; stream not rerouted inside bootstrap',status='previously reused CS/ACDC exploratory; exclude all SAE P0 from evaluation; 92 CS + 406 ACDC; unadjusted CIs',source={p.name:sha(p) for p in Path(__file__).parent.iterdir() if p.suffix in ['.py','.txt']},features={p.name:sha(p) for p in Path(__file__).parent.joinpath('features').glob('*.py')},weights={b.name:sha(b) for b in weights.glob('*.pt')},models={k:sha(ART/'runs'/(k+'_ae_b8_catboost')/'model.cbm') for k in KINDS},references={k:sha(ART/'runs'/(k+'_ae_b8_catboost')/'scores.parquet') for k in KINDS},runtime=dict(torch=str(torch.__version__),transformers=__import__('transformers').__version__,catboost=__import__('catboost').__version__,gpu=torch.cuda.get_device_name(0),tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,deterministic=torch.backends.cudnn.deterministic)))
 fixed(OUT/'manifest.json',dict(fit=fit,test=test));return fit,test

def small(r):
 with Image.open(r['rgb']) as im:a=np.array(im.convert('RGB').resize(tuple(r['size']),Image.Resampling.BILINEAR),copy=True)
 return torch.from_numpy(a).permute(2,0,1)
def ground(r):
 with Image.open(r['gt']) as im:g=np.array(im.resize(tuple(r['size']),Image.Resampling.NEAREST))
 if r['cityscapes']:g=LUT[g]
 assert set(np.unique(g))<=set(range(19))|{255};return g

class Engine:
 def __init__(self):
  self.b0=load_model('b0').requires_grad_(False);self.b5=load_model('b5').requires_grad_(False)
  self.scorers={k:ErrorScorer(k,root=ART,b0=self.b0) for k in KINDS}
  for scorer in self.scorers.values():
   for enc in scorer.enc.values():enc.requires_grad_(False)
  self.knn=KNN().cuda();self.entropy=Entropy().cuda()
  fixed(OUT/'model_revisions.json',dict(b0=getattr(self.b0.net.config,'_commit_hash',None),b5=getattr(self.b5.net.config,'_commit_hash',None)))
 def native(self):return SimpleNamespace(logits=self.b0.features['logits'][None],taps={'decoder':self.b0.features['C']})
 @torch.inference_mode()
 def forward(self,img,kind=None):
  w,h=img.shape[-1],img.shape[-2]
  if kind is None:
   probs=self.b0(img[None].cuda(),(w,h)).softmax(1)[0];return probs,{},float(self.entropy(self.native()))
  result=self.scorers[kind](img);return result['probs'],result,float(self.entropy(self.native()))
 @torch.inference_mode()
 def all_scores(self,img):
  probs,out,ent=self.forward(img,'ZR')
  f0=output_features(probs,out['pred']);parts={'F0':pd.DataFrame({c:f0[c] for c in F0}),**{k:pd.DataFrame(out[k]) for k in ['N','Z','R']}}
  maps={'ZR':out['score']}
  assert torch.isfinite(probs).all() and probs.shape[0]==19
  assert all(np.isfinite(out[k]).all() for k in ['N','Z','R'])
  from catboost import Pool
  for k in ['Z','R']:
   scorer=self.scorers[k];x=pd.concat([parts[p] for p in scorer.parts],axis=1);x.columns=scorer.model.feature_names_;x['cls']=x['cls'].astype(np.int32);score=scorer.model.predict(Pool(x,cat_features=['cls']),prediction_type='Probability')[:,1];maps[k]=score.reshape(out['score'].shape).astype(np.float32)
  for key,values in maps.items():
   assert values.shape==(img.shape[-2]//16,img.shape[-1]//16) and np.isfinite(values).all() and ((values>=0)&(values<=1)).all(), 'Invalid local score map: '+key
  knn=float(self.knn(self.native())) if self.knn.bank.numel() else None
  return probs,maps,ent,knn
 @torch.inference_mode()
 def heavy_probs(self,img,scale):
  h,w=img.shape[-2:];x=img[None].cuda().float()
  if scale!=1.:x=F.interpolate(x,scale_factor=scale,mode='bilinear',align_corners=False,antialias=True)
  return self.b5(x,(w,h)).softmax(1)[0]
 @torch.inference_mode()
 def predict(self,img,probs,level):
  if not level.scale:return probs.argmax(0).byte().cpu().numpy()
  heavy=self.heavy_probs(img,level.scale)
  return ((probs+heavy)/2 if level.ensemble else heavy).argmax(0).byte().cpu().numpy()

@torch.inference_mode()
def acceptance():
 fit,test=setup();engine=Engine();refs={k:pd.read_parquet(ART/'runs'/(k+'_ae_b8_catboost')/'scores.parquet') for k in KINDS};checks=[]
 selected=[next(r for r in fit if r['name']=='frankfurt_000000_000576')]+[next(r for r in test if r['domain']==d) for d in ['fog','night','rain','snow']]
 # Include author's published ACDC acceptance frame if present.
 selected += [r for r in test if r['name']=='night_GOPR0351_frame_000033' and r not in selected]
 for r in selected:
  img=small(r);p,maps,_,_=engine.all_scores(img)
  plain=engine.b0(img[None].cuda(),tuple(r['size'])).softmax(1)[0];assert torch.equal(p,plain),'Scorer hooks altered B0 logits/probabilities'
  for k in KINDS:
   ref=refs[k];ref=ref[(ref.dataset==('cityscapes' if r['cityscapes'] else 'acdc'))&(ref.image_id==r['name'])];assert len(ref)>0,r['name']
   d=np.abs(maps[k].ravel()[ref.patch.to_numpy().astype(int)]-ref.score.to_numpy());checks.append(dict(name=r['name'],method=k,patches=len(ref),max_error=float(d.max()),mean_error=float(d.mean()),p99_error=float(np.quantile(d,.99))))
  print('Acceptance',r['name'],flush=True)
 write(OUT/'acceptance.json',checks);passed=all(r['max_error']<1e-6 for r in checks);write(OUT/'acceptance_status.json',dict(passed=passed));print(pd.DataFrame(checks).to_string(index=False),flush=True)
 assert passed,'Acceptance failed: do not relax tolerance or continue. Reports saved in acceptance.json.'

def require_acceptance():
 mode=json.loads((OUT/'evaluation_mode.json').read_text())
 assert mode['mode']=='local_scorer_after_failed_acceptance'
 evidence=OUT/'reference_acceptance.json'
 assert sha(evidence)==mode['acceptance_evidence_sha256'], 'Acceptance evidence changed'
 assert json.loads((OUT/'reference_acceptance_status.json').read_text())['passed'] is False
 rows=json.loads(evidence.read_text())
 assert rows and {r['method'] for r in rows}==set(KINDS)
 assert all(r['patches']>0 and all(np.isfinite(r[k]) and r[k]>=0 for k in ['max_error','mean_error','p99_error']) for r in rows)
 assert any(r['max_error']>=1e-6 for r in rows), 'This branch specifically records failed reference reproduction'
 assert mode['strict_reference_tolerance']==1e-6


@torch.inference_mode()
def fit():
 train,test=setup();require_acceptance();engine=Engine()
 # Exact baseline KNN bank sampling, fitted only on clean Frankfurt frames.
 for i,r in enumerate(train):
  engine.forward(small(r));engine.knn.update(engine.native())
  if (i+1)%20==0:print('KNN bank',i+1,'/',len(train),flush=True)
 engine.knn.finish();scores=[]
 for i,r in enumerate(train):
  _,maps,ent,knn=engine.all_scores(small(r));scores.append([knn]+[float(maps[k].mean()) for k in KINDS])
  if (i+1)%20==0:print('Fit flags',i+1,'/',len(train),flush=True)
 thresholds={method:float(np.quantile(np.array(scores)[:,j],QUANTILE)) for j,method in enumerate(METHODS[1:])}
 value=dict(bank=engine.knn.bank.cpu(),thresholds=thresholds,protocol_sha=sha(OUT/'protocol.json'))
 dest=OUT/'state.pt';tmp=dest.with_suffix('.pt.part');torch.save(value,tmp);tmp.replace(dest);save(OUT/'fit_scores.npz',scores=np.array(scores),names=np.array([r['name'] for r in train]));write(OUT/'thresholds.json',thresholds);print(thresholds,flush=True)

def load_state(engine):
 state=torch.load(OUT/'state.pt',map_location='cpu',weights_only=True);assert state['protocol_sha']==sha(OUT/'protocol.json');engine.knn.bank=state['bank'].cuda();return state

def routers(state):
 result={}
 for method in METHODS:
  r=Router(BUDGET,QUANTILE,WINDOW)
  if method!='entropy':r.threshold=state['thresholds'][method]
  result[method]=r
 return result

def flag(scores,method):return None if method=='entropy' else float(scores[METHODS.index(method)])

@torch.inference_mode()
def cache():
 train,test=setup();require_acceptance();engine=Engine();load_state(engine);start=time.monotonic()
 for i,r in enumerate(test):
  dest=CACHE/(r['name']+'.npz')
  if dest.exists():continue
  img=small(r);probs,maps,ent,knn=engine.all_scores(img);g=ground(r);cs=[conf(g,probs.argmax(0).byte().cpu().numpy())]
  for level in MODES[1:]:cs.append(conf(g,engine.predict(img,probs,LEVELS[level])))
  save(dest,confs=np.stack(cs),scores=np.array([ent,knn]+[float(maps[k].mean()) for k in KINDS]))
  if (i+1)%5==0 or i+1==len(test):print('Cache',i+1,'/',len(test),'elapsed',round((time.monotonic()-start)/60,1),'min',flush=True)
 write(OUT/'cache_complete.json',dict(complete=True,frames=len(test)));analyze()

def bootstrap(a,b,groups):
 unique=np.unique(groups);aa=np.stack([a[groups==g].sum(0) for g in unique]).astype(float);bb=np.stack([b[groups==g].sum(0) for g in unique]).astype(float);rng=np.random.default_rng(0)
 weights=rng.multinomial(len(unique),np.ones(len(unique))/len(unique),size=2000)
 def many(c):
  c=c.reshape(-1,19,19);d=np.diagonal(c,axis1=1,axis2=2);u=c.sum(1)+c.sum(2)-d
  return 100*np.nanmean(np.divide(d,u,out=np.full_like(d,np.nan),where=u>0),axis=1)
 differences=many(weights@aa.reshape(len(unique),-1))-many(weights@bb.reshape(len(unique),-1))
 return tuple(float(x) for x in np.percentile(differences,[2.5,97.5]))

def replay(test,state,order):
 rs=routers(state);cs=np.zeros((len(test),len(METHODS),19,19),np.int64);cost=np.zeros((len(test),len(METHODS)));levels=np.zeros((len(test),len(METHODS)),int)
 for i in order:
  with np.load(CACHE/(test[i]['name']+'.npz'),allow_pickle=False) as z:
   for j,m in enumerate(METHODS):
    level=rs[m].decide(flag(z['scores'],m),float(z['scores'][0]));index=MODES.index(level.name);levels[i,j]=index;cost[i,j]=level.cost;cs[i,j]=z['confs'][index]
 return cs,cost,levels

def analyze():
 train,test=setup();state=torch.load(OUT/'state.pt',map_location='cpu',weights_only=True);summary=[];boots=[];names=np.array([r['name'] for r in test]);groups=np.array([r['sequence'] for r in test]);domains=np.array([r['domain'] for r in test]);base=[];full=[]
 for r in test:
  with np.load(CACHE/(r['name']+'.npz'),allow_pickle=False) as z:base.append(z['confs'][0]);full.append(z['confs'][-1])
 base=np.stack(base);full=np.stack(full)
 for seed in [None,0,1,2,3,4]:
  order=np.arange(len(test)) if seed is None else np.random.default_rng(seed).permutation(len(test));label='blocked' if seed is None else 'shuffle'+str(seed);cs,cost,levels=replay(test,state,order);save(OUT/(label+'_confusions.npz'),confs=cs,cost=cost,levels=levels,names=names,groups=groups,domains=domains,methods=np.array(METHODS),order=order)
  for domain in DOMAINS+['acdc_all','pool']:
   ix=np.ones(len(test),bool) if domain=='pool' else domains!='cs_test' if domain=='acdc_all' else domains==domain
   for j,m in enumerate(METHODS):summary.append(dict(order=label,domain=domain,method=m,miou=miou(cs[ix,j].sum(0)),frames=int(ix.sum()),mean_b5_area_cost=float(cost[ix,j].mean())))
   for name,c in [('b0',base),('b5_full',full)]:summary.append(dict(order=label,domain=domain,method=name,miou=miou(c[ix].sum(0)),frames=int(ix.sum()),mean_b5_area_cost=0 if name=='b0' else 1))
   pairs=[('ZR_entropy','knn_entropy'),('Z_entropy','knn_entropy'),('R_entropy','knn_entropy'),('knn_entropy','entropy')]+[(k+'_entropy','entropy') for k in KINDS]
   for a,b in pairs:
    aa=cs[ix,METHODS.index(a)];bb=cs[ix,METHODS.index(b)]
    # Group CI for every order; frame CI for blocked only to limit CPU work.
    for unit,gg in [('sequence',groups[ix])]+([('frame',names[ix])] if seed is None else []):
     lo,hi=bootstrap(aa,bb,gg);boots.append(dict(order=label,domain=domain,comparison=a+' minus '+b,unit=unit,groups=len(np.unique(gg)),delta_pp=miou(aa.sum(0))-miou(bb.sum(0)),ci_low=lo,ci_high=hi,primary=label=='blocked' and domain=='pool' and a=='ZR_entropy' and b=='knn_entropy',scope='exploratory; unadjusted; fixed realized stream decisions'))
 csv('final_miou.csv',summary);csv('bootstrap.csv',boots);print(pd.DataFrame(summary).query("order=='blocked' and domain in ['pool','acdc_all']").to_string(index=False));print(pd.DataFrame(boots).query("order=='blocked' and domain=='pool' and unit=='sequence'").to_string(index=False));write(OUT/'quality_complete.json',dict(complete=True,frames=len(test),orders=6,speed_claim=False))

@torch.inference_mode()
def timing():
 train,test=setup();require_acceptance();engine=Engine();state=load_state(engine);rs=routers(state);selected=[]
 for d in DOMAINS:
  ix=[i for i,r in enumerate(test) if r['domain']==d];selected += [ix[j] for j in np.linspace(0,len(ix)-1,min(8,len(ix))).astype(int)]
 # Restore the exact preceding stream state for each timed frame, outside the clock.
 snapshots={};decisions={}
 for i,r in enumerate(test):
  with np.load(CACHE/(r['name']+'.npz'),allow_pickle=False) as z:
   for method in METHODS:
    if i in selected:snapshots[i,method]=copy.deepcopy(rs[method])
    decisions[i,method]=rs[method].decide(flag(z['scores'],method),float(z['scores'][0])).name
 def call(r,method,router):
  torch.cuda.synchronize();start=time.perf_counter();img=small(r)
  if method=='b5_full':answer=engine.heavy_probs(img,1.).argmax(0).byte().cpu().numpy();level='heavy'
  else:
   kind=method.split('_')[0] if method.split('_')[0] in KINDS else None
   probs,out,ent=engine.forward(img,kind)
   fl=float(out['score'].mean()) if kind else float(engine.knn(engine.native())) if method=='knn_entropy' else None
   decision=router.decide(fl,ent);answer=engine.predict(img,probs,decision);level=decision.name
  torch.cuda.synchronize();return answer,level,(time.perf_counter()-start)*1000
 for method in METHODS+['b5_full']:call(test[selected[0]],method,copy.deepcopy(snapshots.get((selected[0],method))))
 for pos,i in enumerate(selected):
  dest=OUT/'timing_records'/(test[i]['name']+'.json')
  if dest.exists():continue
  record={m:[] for m in METHODS+['b5_full']};tasks=[(m,rep) for m in record for rep in range(2)];np.random.default_rng(i).shuffle(tasks)
  for method,rep in tasks:
   answer,level,ms=call(test[i],method,copy.deepcopy(snapshots.get((i,method))))
   with np.load(CACHE/(test[i]['name']+'.npz'),allow_pickle=False) as z:expected=z['confs'][MODES.index(level)]
   assert level==('heavy' if method=='b5_full' else decisions[i,method]),f'Live level mismatch: {test[i]["name"]} {method}'
   assert np.array_equal(conf(ground(test[i]),answer),expected),f'Live/cache quality mismatch: {test[i]["name"]} {method}'
   record[method].append(float(ms))
  write(dest,record);print('Timing',pos+1,'/',len(selected),flush=True)
 reports=[]
 for method in METHODS+['b5_full']:
  times=np.array([json.loads((OUT/'timing_records'/(test[i]['name']+'.json')).read_text())[method] for i in selected]).ravel();reports.append(dict(method=method,mean_ms=float(times.mean()),p95_ms=float(np.percentile(times,95)),frames=len(selected),repeats=2,scope='fixed 40-frame subset, blocked-stream states, CPU decode to CPU mask, excludes GT; not all498'))
 csv('timing.csv',reports);print(pd.DataFrame(reports).to_string(index=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','acceptance','fit','cache','analyze','timing']);args=p.parse_args();torch.set_num_threads(4);assert torch.cuda.is_available(),'Enable GPU'
 if args.stage=='preflight':f,t=setup();print('Inputs OK:',len(f),'fit frames,',len(t),'evaluation frames')
 else:globals()[args.stage]()
