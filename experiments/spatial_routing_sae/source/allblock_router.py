"""Frozen all-block N/Z gain router. New architecture and paired controls; no CatBoost reuse."""
import os
os.environ['USE_TF']='0'
import argparse,json,time,hashlib,random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import GroupKFold
from ae_model import BottleneckAE

ROOT=Path(os.environ['ALLBLOCK_ROOT']);OUT=Path(os.environ['ALLBLOCK_OUT']);CS=Path(os.environ['ALLBLOCK_CS']);AE=Path(os.environ['ALLBLOCK_AE']);ACDC=Path(os.environ.get('ALLBLOCK_ACDC','/content/no_acdc'))
CACHE=ROOT/'cache';VARIANTS=['confidence','confidence_N','confidence_NZ'];SEEDS=[0,1,2];EPOCHS=18;KS=[1,2,3];PRIMARY_K=1
BLOCKS=[f's{s}b{j}' for s in range(1,5) for j in [1,2]]
B0='nvidia/segformer-b0-finetuned-cityscapes-512-1024';B5='nvidia/segformer-b5-finetuned-cityscapes-1024-1024'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.part');t.write_text(json.dumps(value,indent=2));t.replace(p)
def freeze(p,value):
 if p.exists():assert json.loads(p.read_text())==value,f'Changed protocol: {p}; use a new experiment directory'
 else:write(p,value)
def save(p,**kw):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.npz.part')
 with t.open('wb') as f:np.savez_compressed(f,**kw)
 t.replace(p)
def tsave(p,value):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.pt.part');torch.save(value,t);t.replace(p)
def ids():
 r=Path(__file__).with_name('p0_ids.txt').read_text().split();assert len(r)==300 and len(set(r))==300;return r
def group(name):return '_'.join(name.split('_')[:2])
def boxes(h):return [(x,y,min(x+256,1024),min(y+256,h)) for y in range(0,h,256) for x in range(0,1024,256)]
def pick(score,k):return np.argsort(-score,kind='stable')[:k]
def conf(gt,p):
 ok=gt!=255;return np.bincount(gt[ok].astype(np.int64)*19+p[ok],minlength=361).reshape(19,19)
def miou(c):
 c=c.astype(float);d=np.diag(c);u=c.sum(0)+c.sum(1)-d;return float(100*(d[u>0]/u[u>0]).mean())
def rgb(path,size):
 with Image.open(path) as im:return np.array(im.convert('RGB').resize(size,Image.Resampling.BILINEAR),copy=True)
def gt_cs(name):
 from cityscapesscripts.helpers.labels import labels
 lut=np.full(256,255,np.uint8)
 for l in labels:
  if 0<=l.id<256 and 0<=l.trainId<19:lut[l.id]=l.trainId
 with Image.open(CS/'gtFine/val'/name.split('_')[0]/(name+'_gtFine_labelIds.png')) as im:return lut[np.array(im.resize((1024,512),Image.Resampling.NEAREST))]
def settings():
 OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
 for b in BLOCKS:assert (AE/(b+'.pt')).is_file(),f'Missing AE: {b}'
 freeze(OUT/'protocol.json',dict(schema='allblock-NZ-gain-v1',script_sha=sha(__file__),ae_source_sha=sha(Path(__file__).with_name('ae_model.py')),p0_ids_sha=sha(Path(__file__).with_name('p0_ids.txt')),ae_sha={b:sha(AE/(b+'.pt')) for b in BLOCKS},b0=B0,b5=B5,variants=VARIANTS,seeds=SEEDS,epochs=EPOCHS,primary_k=PRIMARY_K,secondary_ks=[2,3],folds='5 GroupKFold city+sequence',target='(corrected-spoiled)/65536',channels=548,architecture='Conv1x1 548->32 ReLU; Conv3x3 32->32 ReLU; avgpool; concat20global; Linear52->32 ReLU->1',optimization='AdamW lr .002 weight_decay .0002 batch128 SmoothL1 beta .02 fixed18epochs',selection='mean OOF mIoU at k1 over seeds; deterministic tie order confidence,N,NZ',normalization='N log1p; N and signed Z mean/std on fold training images only; zero unused channels',torch=str(torch.__version__),transformers=__import__('transformers').__version__,status='new controls, not numerically comparable to old base; ACDC exploratory reused; no speed claim',acdc_used_for_selection=False))

def pool_grid(x,stage,hw,reduce='mean'):
 hp,wp=hw;r=[4,8,16,32][stage]
 def axis(n,grid):
  if r<16:return np.arange(n)[:,None]*(16//r)+np.arange(16//r)[None,:]
  return np.clip(np.floor((16*np.arange(n)+7.5)/r+.5).astype(int)[:,None],0,grid-1)
 iy=torch.as_tensor(axis(hp,x.shape[0]),device=x.device);ix=torch.as_tensor(axis(wp,x.shape[1]),device=x.device)
 y=x[iy][:,:,ix]
 return y.mean((1,3)) if reduce=='mean' else y.amax((1,3))

class Router(nn.Module):
 def __init__(self):
  super().__init__();self.local=nn.Sequential(nn.Conv2d(548,32,1),nn.ReLU(),nn.Conv2d(32,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten());self.head=nn.Sequential(nn.Linear(52,32),nn.ReLU(),nn.Linear(32,1))
 def forward(self,x,g):return self.head(torch.cat([self.local(x),g],1)).squeeze(1)

class Engine:
 def __init__(self):
  from transformers import SegformerForSemanticSegmentation
  self.b0=SegformerForSemanticSegmentation.from_pretrained(B0).cuda().eval().requires_grad_(False)
  self.b5=SegformerForSemanticSegmentation.from_pretrained(B5).cuda().eval().requires_grad_(False)
  self.enc={}
  for b in BLOCKS:
   ck=torch.load(AE/(b+'.pt'),map_location='cpu',weights_only=True);assert ck['kind']=='ae' and ck['block']==b
   e=BottleneckAE(**ck['config']);e.load_state_dict(ck['state']);self.enc[b]=e.cuda().eval().requires_grad_(False)
  self.mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]*255;self.std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]*255
  freeze(OUT/'model_revisions.json',dict(b0=getattr(self.b0.config,'_commit_hash',None),b5=getattr(self.b5.config,'_commit_hash',None)))
 def image(self,array):return (torch.from_numpy(array).permute(2,0,1)[None].cuda().float()-self.mean)/self.std
 @torch.inference_mode()
 def features(self,array,sanity=False):
  h,w=array.shape[:2];acts={};grids={};handles=[]
  for s in range(4):handles.append(self.b0.segformer.encoder.patch_embeddings[s].register_forward_hook(lambda m,a,o,s=s:grids.__setitem__(s,(o[1],o[2]))))
  for b in BLOCKS:
   s=int(b[1])-1;j=int(b[3])-1
   handles.append(self.b0.segformer.encoder.block[s][j].mlp.dense2.register_forward_pre_hook(lambda m,a,b=b:acts.__setitem__(b,a[0].detach())))
  try:logits=self.b0(pixel_values=self.image(array)).logits
  finally:
   for handle in handles:handle.remove()
  if sanity:assert torch.equal(logits,self.b0(pixel_values=self.image(array)).logits),'Hooks changed B0'
  p=F.interpolate(logits,(h,w),mode='bilinear',align_corners=False).softmax(1)[0];pred=p.argmax(0);entropy=-(p*p.clamp_min(1e-12).log()).sum(0);top=p.topk(2,dim=0).values
  edge=torch.zeros_like(pred,dtype=torch.float);edge[:,:-1]=torch.maximum(edge[:,:-1],(pred[:,:-1]!=pred[:,1:]).float());edge[:-1]=torch.maximum(edge[:-1],(pred[:-1]!=pred[1:]).float())
  yy,xx=torch.meshgrid(torch.linspace(0,1,h,device='cuda'),torch.linspace(0,1,w,device='cuda'),indexing='ij');base=torch.cat([p,entropy[None],(top[0]-top[1])[None],edge[None],xx[None],yy[None]])
  ns=[];zs=[]
  for b in BLOCKS:
   s=int(b[1])-1;a=acts[b][0].reshape(*grids[s],-1);e=self.enc[b];x=e.normalize(a.reshape(-1,a.shape[-1]));n=x.square().mean(1).reshape(*a.shape[:2],1)
   ns.append(pool_grid(n,s,(h//16,w//16),'mean'))
   if s<2:ns.append(pool_grid(n,s,(h//16,w//16),'max'))
   code=e.encode(x).reshape(*a.shape[:2],-1);zs.append(pool_grid(code,s,(h//16,w//16),'mean'))
  n=torch.cat(ns,-1);z=torch.cat(zs,-1);assert n.shape[-1]==12 and z.shape[-1]==512
  geo=torch.cat([n,z],-1).permute(2,0,1)[None]
  local=[];es=[]
  for x0,y0,x1,y1 in boxes(h):
   bx=F.interpolate(base[:,y0:y1,x0:x1][None],(16,16),mode='bilinear',align_corners=False)[0]
   gx=F.interpolate(geo[:,:,y0//16:y1//16,x0//16:x1//16],(16,16),mode='bilinear',align_corners=False)[0]
   local.append(torch.cat([bx,gx]));es.append(float(entropy[y0:y1,x0:x1].mean()))
  x=torch.stack(local).cpu().numpy();assert x.shape==(len(boxes(h)),548,16,16) and np.isfinite(x).all();assert np.isfinite(x.astype(np.float16)).all(),'Feature overflow in fp16 cache'
  g=torch.cat([p.mean((1,2)),entropy.mean()[None]]).cpu().numpy();return x.astype(np.float16),g.astype(np.float32),np.array(es,np.float32),pred.byte().cpu().numpy()
 @torch.inference_mode()
 def heavy(self,array,h,cs):
  result=np.zeros((h,1024),np.uint8)
  for x0,y0,x1,y1 in boxes(h):
   wx=min(max(x0-128,0),512);wy=min(max(y0-128,0),h-512);s=2 if cs else 1
   crop=array[s*wy:s*(wy+512),s*wx:s*(wx+512)]
   assert crop.shape==(s*512,s*512,3)
   out=self.b5(pixel_values=self.image(crop)).logits
   pred=F.interpolate(out,(512,512),mode='bilinear',align_corners=False).argmax(1)[0].byte().cpu().numpy()
   result[y0:y1,x0:x1]=pred[y0-wy:y1-wy,x0-wx:x1-wx]
  return result

def prepare_cs(smoke=False):
 settings();names=ids()
 for name in names:
  city=name.split('_')[0]
  for p in [CS/'leftImg8bit/val'/city/(name+'_leftImg8bit.png'),CS/'gtFine/val'/city/(name+'_gtFine_labelIds.png')]:assert p.is_file(),str(p)
 engine=Engine();start=time.monotonic()
 for i,name in enumerate(names[:2] if smoke else names):
  dest=CACHE/'cs'/(name+'.npz')
  if dest.exists():continue
  image=CS/'leftImg8bit/val'/name.split('_')[0]/(name+'_leftImg8bit.png');gt=gt_cs(name);x,g,entropy,b0=engine.features(rgb(image,(1024,512)),sanity=i==0)
  with Image.open(image) as im:assert im.size==(2048,1024);original=np.array(im.convert('RGB'),copy=True)
  b5=engine.heavy(original,512,True);effects=[];gains=[];corrected=[];spoiled=[]
  for x0,y0,x1,y1 in boxes(512):
   sl=np.s_[y0:y1,x0:x1];valid=gt[sl]!=255;corr=int(((b5[sl]==gt[sl])&(b0[sl]!=gt[sl])&valid).sum());spoil=int(((b0[sl]==gt[sl])&(b5[sl]!=gt[sl])&valid).sum());gains.append((corr-spoil)/65536);corrected.append(corr);spoiled.append(spoil);effects.append(conf(gt[sl],b5[sl])-conf(gt[sl],b0[sl]))
  assert np.array_equal(conf(gt,b0)+np.stack(effects).sum(0),conf(gt,b5)),'Tile merge mismatch'
  save(dest,x=x,g=g,y=np.array(gains,np.float32),entropy=entropy,c0=conf(gt,b0),effects=np.stack(effects),corrected=np.array(corrected),spoiled=np.array(spoiled))
  elapsed=time.monotonic()-start;print(f'CS {i+1}/{len(names)} elapsed {elapsed/60:.1f} min',flush=True)
 if not smoke:write(OUT/'cache_complete.json',dict(complete=True,frames=len(names)))

def load_data():
 names=ids();records=[]
 for name in names:
  p=CACHE/'cs'/(name+'.npz');assert p.is_file(),f'Run cache: {p}'
  with np.load(p,allow_pickle=False) as z:records.append({k:z[k].copy() for k in z.files})
 return names,records

def stats(x,indices):
 s=np.zeros(524,np.float64);ss=s.copy();count=0
 for i in indices:
  a=x[i][:,24:].astype(np.float64);a[:,:12]=np.log1p(np.maximum(a[:,:12],0));s+=a.sum((0,2,3));ss+=(a*a).sum((0,2,3));count+=a.shape[0]*a.shape[2]*a.shape[3]
 mean=s/count;std=np.maximum(np.sqrt(np.maximum(ss/count-mean*mean,0)),1e-3);return mean.astype(np.float32),std.astype(np.float32)
def inputs(x,mean,std,variant):
 x=x.float().clone();geo=x[:,24:];geo[:,:12]=torch.log1p(geo[:,:12].clamp_min(0));geo.sub_(mean[None,:,None,None]).div_(std[None,:,None,None])
 if variant=='confidence':geo.zero_()
 elif variant=='confidence_N':geo[:,12:]=0
 return x

def fit(records,train,test,variant,seed,mean,std):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 model=Router().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0002);loss_fn=nn.SmoothL1Loss(beta=.02)
 x=torch.from_numpy(np.concatenate([records[i]['x'] for i in train]));g=torch.from_numpy(np.concatenate([np.repeat(records[i]['g'][None],8,axis=0) for i in train]));y=torch.from_numpy(np.concatenate([records[i]['y'] for i in train]))
 mean=torch.as_tensor(mean,device='cuda');std=torch.as_tensor(std,device='cuda');generator=torch.Generator().manual_seed(seed);history=[]
 for epoch in range(EPOCHS):
  model.train();perm=torch.randperm(len(y),generator=generator);total=0
  for ix in perm.split(128):
   bx=inputs(x[ix].cuda(),mean,std,variant);pred=model(bx,g[ix].cuda());loss=loss_fn(pred,y[ix].cuda());opt.zero_grad(set_to_none=True);loss.backward();opt.step();total+=float(loss.detach())*len(ix)
  history.append(dict(epoch=epoch+1,train_loss=total/len(y)))
 model.eval();predictions=[]
 with torch.inference_mode():
  for i in test:
   r=records[i];bx=inputs(torch.from_numpy(r['x']).cuda(),mean,std,variant);gg=torch.from_numpy(np.repeat(r['g'][None],len(bx),axis=0)).cuda();predictions.append(model(bx,gg).cpu().numpy())
 return model,np.array(predictions),history

def boot(a,b,groups):
 unique=np.unique(groups);aa=np.stack([a[groups==g].sum(0) for g in unique]);bb=np.stack([b[groups==g].sum(0) for g in unique]);rng=np.random.default_rng(0);ds=[]
 for _ in range(2000):
  ix=rng.integers(len(unique),size=len(unique));ds.append(np.mean([miou(aa[ix,s].sum(0))-miou(bb[ix,s].sum(0)) for s in range(aa.shape[1])]))
 return tuple(float(z) for z in np.percentile(ds,[2.5,97.5]))

def cv():
 settings();names,records=load_data();groups=np.array([group(n) for n in names]);folds=list(GroupKFold(5).split(names,groups=groups));allx=[r['x'] for r in records]
 freeze(OUT/'folds.json',[dict(train=[names[i] for i in tr],val=[names[i] for i in va]) for tr,va in folds]);oof=np.zeros((len(names),len(SEEDS),len(VARIANTS),8),np.float32)
 for f,(tr,va) in enumerate(folds):
  assert set(groups[tr]).isdisjoint(groups[va]);mean,std=stats(allx,tr)
  for s,seed in enumerate(SEEDS):
   for v,variant in enumerate(VARIANTS):
    dest=OUT/'cv'/f'fold{f}_seed{seed}_{variant}.npz'
    if dest.exists():
     with np.load(dest,allow_pickle=False) as z:assert list(z['names'])==[names[i] for i in va];prediction=z['predictions']
    else:
     model,prediction,history=fit(records,tr,va,variant,seed,mean,std);save(dest,names=np.array(names)[va],predictions=prediction);write(dest.with_suffix('.json'),history);del model
    oof[va,s,v]=prediction;print('CV',f+1,'/5 seed',seed,variant,'done',flush=True)
 save(OUT/'oof_predictions.npz',names=np.array(names),groups=groups,predictions=oof,variants=np.array(VARIANTS),seeds=np.array(SEEDS))
 cs=np.zeros((len(names),len(KS),len(SEEDS),len(VARIANTS)+1,19,19),np.int64)
 for i,r in enumerate(records):
  for j,k in enumerate(KS):
   for s,seed in enumerate(SEEDS):
    for v,variant in enumerate(VARIANTS+['entropy']):
     score=oof[i,s,v] if v<len(VARIANTS) else r['entropy'];cs[i,j,s,v]=r['c0']+r['effects'][pick(score,k)].sum(0)
 save(OUT/'oof_confusions.npz',confs=cs,names=np.array(names),groups=groups,methods=np.array(VARIANTS+['entropy']),ks=np.array(KS))
 summary=[];bs=[]
 for j,k in enumerate(KS):
  for v,variant in enumerate(VARIANTS+['entropy']):
   values=[miou(cs[:,j,s,v].sum(0)) for s in range(len(SEEDS))];summary.append(dict(method=variant,k=k,mean_miou=float(np.mean(values)),seed_min=min(values),seed_max=max(values)))
  for a,b in [('confidence_N','confidence'),('confidence_NZ','confidence_N'),('confidence_NZ','confidence'),('confidence','entropy')]:
   aa=cs[:,j,:, (VARIANTS+['entropy']).index(a)];bb=cs[:,j,:, (VARIANTS+['entropy']).index(b)];lo,hi=boot(aa,bb,groups);delta=np.mean([miou(aa[:,s].sum(0))-miou(bb[:,s].sum(0)) for s in range(len(SEEDS))]);bs.append(dict(k=k,comparison=a+' minus '+b,delta_pp=float(delta),ci_low=lo,ci_high=hi,groups=len(np.unique(groups)),primary=k==PRIMARY_K and a=='confidence_NZ' and b=='confidence_N',status='exploratory; unadjusted sequence bootstrap conditional on trained folds'))
 pd.DataFrame(summary).to_csv(OUT/'cv_summary.csv',index=False);pd.DataFrame(bs).to_csv(OUT/'cv_bootstrap.csv',index=False)
 chosen=max(VARIANTS,key=lambda v:next(r['mean_miou'] for r in summary if r['method']==v and r['k']==PRIMARY_K));write(OUT/'selection.json',dict(variant=chosen,chosen_on='P0 grouped OOF mean over seeds k1',primary_k=PRIMARY_K,acdc_used=False,speed_claim=False));print(pd.DataFrame(summary).to_string(index=False));print(pd.DataFrame(bs).to_string(index=False))

def final():
 settings();assert (OUT/'selection.json').is_file(),'Run CV first';names,records=load_data();mean,std=stats([r['x'] for r in records],range(len(records)))
 for variant in VARIANTS:
  dest=OUT/('router_'+variant+'.pt')
  if dest.exists():continue
  model,_,history=fit(records,np.arange(len(records)),[],variant,0,mean,std);tsave(dest,dict(state=model.cpu().state_dict(),mean=torch.from_numpy(mean),std=torch.from_numpy(std),variant=variant,seed=0,channels=548,epochs=EPOCHS,ae_sha={b:sha(AE/(b+'.pt')) for b in BLOCKS}));write(dest.with_suffix('.json'),history)
 write(OUT/'training_complete.json',dict(complete=True,checkpoints={v:sha(OUT/('router_'+v+'.pt')) for v in VARIANTS},selection=json.loads((OUT/'selection.json').read_text()),acdc_run=False));print('Final routers saved. B0, B5 and AE unchanged.',flush=True)

def preflight():
 settings();names=ids();missing=[]
 for name in names:
  city=name.split('_')[0]
  for p in [CS/'leftImg8bit/val'/city/(name+'_leftImg8bit.png'),CS/'gtFine/val'/city/(name+'_gtFine_labelIds.png')]:
   if not p.is_file():missing.append(str(p))
 assert not missing,'Missing Cityscapes inputs:\n'+'\n'.join(missing[:10]);print('Preflight OK: 300 CS frames, GT, 8 AE checkpoints',flush=True)


def acdc():
 settings();complete=json.loads((OUT/'training_complete.json').read_text());assert complete['complete']
 models={}
 for variant in VARIANTS:
  path=OUT/('router_'+variant+'.pt');assert sha(path)==complete['checkpoints'][variant]
  ck=torch.load(path,map_location='cpu',weights_only=True);model=Router().cuda().eval();model.load_state_dict(ck['state']);models[variant]=(model,ck['mean'].cuda(),ck['std'].cuda())
 manifest=[]
 for domain in ['fog','night','rain','snow']:
  files=sorted((ACDC/'rgb_anon'/domain/'val').rglob('*_rgb_anon.png'));assert files,f'No ACDC RGB: {domain}'
  for p in files:
   stem=p.name[:-len('_rgb_anon.png')];gt=ACDC/'gt'/domain/'val'/p.parent.name/(stem+'_gt_labelTrainIds.png');assert gt.is_file(),str(gt)
   manifest.append(dict(name=domain+'_'+stem,domain=domain,group=domain+'/'+p.parent.name,rgb=str(p),gt=str(gt)))
 freeze(OUT/'acdc_manifest.json',manifest);engine=Engine()
 with torch.inference_mode():
  for i,r in enumerate(manifest):
   dest=CACHE/'acdc'/(r['name']+'.npz')
   if dest.exists():continue
   array=rgb(r['rgb'],(1024,576));x,g,entropy,b0=engine.features(array,sanity=i==0)
   with Image.open(r['gt']) as im:gt=np.array(im.resize((1024,576),Image.Resampling.NEAREST))
   assert set(np.unique(gt))<=set(range(19))|{255}
   b5=engine.heavy(array,576,False);effects=[]
   for x0,y0,x1,y1 in boxes(576):
    sl=np.s_[y0:y1,x0:x1];effects.append(conf(gt[sl],b5[sl])-conf(gt[sl],b0[sl]))
   predictions=[]
   for variant in VARIANTS:
    model,mean,std=models[variant];xx=inputs(torch.from_numpy(x).cuda(),mean,std,variant);gg=torch.from_numpy(np.repeat(g[None],len(x),0)).cuda();predictions.append(model(xx,gg).cpu().numpy())
   predictions.append(entropy);save(dest,c0=conf(gt,b0),effects=np.stack(effects),scores=np.stack(predictions));print('ACDC',i+1,'/',len(manifest),flush=True)
 cs=[]
 for r in manifest:
  with np.load(CACHE/'acdc'/(r['name']+'.npz'),allow_pickle=False) as z:
   cs.append([[z['c0']+z['effects'][pick(z['scores'][v],k)].sum(0) for v in range(4)] for k in KS])
 cs=np.array(cs);names=np.array([r['name'] for r in manifest]);groups=np.array([r['group'] for r in manifest]);domains=np.array([r['domain'] for r in manifest]);methods=VARIANTS+['entropy'];save(OUT/'acdc_confusions.npz',confs=cs,names=names,groups=groups,methods=np.array(methods),ks=np.array(KS))
 summary=[];bs=[]
 for domain in ['fog','night','rain','snow','all']:
  ix=np.ones(len(names),bool) if domain=='all' else domains==domain
  for j,k in enumerate(KS):
   for v,method in enumerate(methods):summary.append(dict(domain=domain,k=k,method=method,miou=miou(cs[ix,j,v].sum(0)),frames=int(ix.sum())))
   for a,b in [('confidence_N','confidence'),('confidence_NZ','confidence_N'),('confidence_NZ','confidence'),('confidence','entropy'),('confidence_N','entropy'),('confidence_NZ','entropy')]:
    aa=cs[ix,j,methods.index(a)][:,None];bb=cs[ix,j,methods.index(b)][:,None]
    if len(np.unique(groups[ix]))<2:lo=hi=None
    else:lo,hi=boot(aa,bb,groups[ix])
    bs.append(dict(domain=domain,k=k,comparison=a+' minus '+b,delta_pp=miou(aa[:,0].sum(0))-miou(bb[:,0].sum(0)),ci_low=lo,ci_high=hi,groups=len(np.unique(groups[ix])),status='reused ACDC exploratory, no time measurement; unadjusted CI'))
 pd.DataFrame(summary).to_csv(OUT/'acdc_quality.csv',index=False);pd.DataFrame(bs).to_csv(OUT/'acdc_bootstrap.csv',index=False);print(pd.DataFrame(summary).to_string(index=False));print(pd.DataFrame(bs).to_string(index=False));write(OUT/'acdc_complete.json',dict(complete=True,frames=len(names),status='exploratory',speed_claim=False))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','smoke','cache','cv','final','acdc']);args=ap.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 assert torch.cuda.is_available(),'Enable GPU'
 if args.stage=='smoke':prepare_cs(True)
 elif args.stage=='cache':prepare_cs(False)
 else:globals()[args.stage]()
