"""E9 diagnostic: B0 F3 -> B5 mid/final state. Frozen segmenters, teacher-assisted routing."""
import os
os.environ['USE_TF']='0'
import argparse, csv, hashlib, json, time, shutil
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from common import CACHE, RESULTS, SIZE, TILE, SegSet, load_model, confusion, miou, top_tiles

OUT=Path(os.environ['E9_OUT'])
LOCAL=Path(os.environ['E9_LOCAL'])
CONFIG={'schema':'e9-mid-final-adapter-v1','seed':0,'input':'B0 normalized encoder F3 (stage index 2 output)',
        'input_channels':160,'target_channels':320,'hidden':128,'target_states':['mid','final'],
        'epochs':20,'patience':4,'lr':.001,'weight_decay':.0001,'batch':2,
        'budgets':[.1,.2],'evaluation_domains':['cs_test'],'methods':['rank'],
        'teacher_assisted':True,'keep_real_b5_early_stages':True,'routed_entry':'real B5',
        'loss':'mean channel-standardized MSE, mid and final equal weight',
        'fit_domains':['cs_fit'],'size':[512,1024], 'compute_benchmark':False}


def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    p=path.with_suffix(path.suffix+'.part');p.write_text(json.dumps(data,indent=2));p.replace(path)

def atomic_torch(path,data):
    p=path.with_suffix('.pt.part');torch.save(data,p);p.replace(path)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def initialize():
    OUT.mkdir(parents=True,exist_ok=True);LOCAL.mkdir(parents=True,exist_ok=True)
    cfg=dict(CONFIG,source_sha256=sha(Path(__file__)),
             common_sha256=sha(Path(__file__).parent/'common.py'),
             tile_features_sha256=sha(CACHE/'e8_tiles_cityscapes.npz'))
    p=OUT/'config.json'
    if p.exists():assert json.loads(p.read_text())==cfg,'Configuration differs: use a new E9_OUT'
    else:write_json(p,cfg)
    return cfg


class ContextBlock(nn.Module):
    def __init__(self,layer,state,index,n):
        super().__init__();self.layer=layer;self.state=state;self.index=index;self.n=n
    def forward(self,x,*args,**kwargs):
        s=self.state
        if s.recording:
            if self.index==0:s.real['entry']=x
            if self.index==self.n//2:s.real['mid']=x
            out=self.layer(x,*args,**kwargs)
            if self.index==self.n-1:s.real['final']=out[0]
            return out
        if s.mask is None:return self.layer(x,*args,**kwargs)
        ctx=s.context
        out=self.layer(torch.where(s.mask,x,ctx),*args,**kwargs)
        y=torch.where(s.mask,out[0],ctx)
        if self.index==self.n-1:y=torch.where(s.mask,y,s.output)
        return (y,)+tuple(out[1:])


def install(model):
    state=SimpleNamespace(recording=False,real={},mask=None,context=None,output=None)
    layers=model.segformer.encoder.block[2]
    for i,layer in enumerate(layers):layers[i]=ContextBlock(layer,state,i,len(layers))
    return state

@torch.inference_mode()
def forward(model,state,img,recording=False,mask=None,context=None,output=None):
    state.recording=recording;state.mask=mask;state.context=context;state.output=output
    try:return model(pixel_values=img).logits
    finally:state.recording=False;state.mask=None;state.context=None;state.output=None

def prediction(logits):
    return F.interpolate(logits.float(),SIZE,mode='bilinear',align_corners=False).argmax(1)[0].byte().cpu().numpy()

def target_path(domain,name):return LOCAL/'targets'/domain/f'{name}.npz'

def check_target(path):
    if not path.is_file():return False
    with np.load(path,allow_pickle=False) as z:
        assert z['x'].shape==(160,32,64)
        assert all(z[k].shape==(320,32,64) for k in ['mid','final'])
        assert all(np.isfinite(z[k]).all() for k in ['x','mid','final'])
    return True


def split_ids(names):
    # City+sequence, not neighboring-frame random split.
    groups={n:'_'.join(n.split('_')[:2]) for n in names}
    counts=Counter(groups.values())
    assert len(counts)>=2,'Need at least two Frankfurt sequences'
    val_group=min(counts,key=lambda g:(counts[g],g))
    train=[n for n in names if groups[n]!=val_group]
    val=[n for n in names if groups[n]==val_group]
    assert train and val and len(val)<len(train),(counts, val_group)
    return {'train':train,'val':val,'val_sequence':val_group,'sequence_counts':dict(counts),
            'note':'Validation is smallest Frankfurt sequence; cs_test never used for selection.'}

@torch.inference_mode()
def export_targets():
    b0=load_model('b0').requires_grad_(False)
    b5=load_model('b5').requires_grad_(False);state=install(b5)
    assert len(b5.segformer.encoder.block[2])==40,'Expected B5 stage 3 depth=40'
    for domain in ['cs_fit','cs_test']:
        data=SegSet(domain)
        expected=267 if domain=='cs_fit' else 233
        assert len(data)==expected
        assert [p.stem for p,_ in data.items]==[p.stem for p in sorted((CACHE/domain).glob('*.npz'))]
        start=time.monotonic()
        for i,(image,gt_path) in enumerate(data.items):
            name=image.stem;p=target_path(domain,name)
            backup=OUT/'targets'/domain/p.name
            p.parent.mkdir(parents=True,exist_ok=True);backup.parent.mkdir(parents=True,exist_ok=True)
            if not p.exists() and backup.exists():shutil.copy2(backup,p)
            if not check_target(p):
                img,_,_=data[i];x=img[None].cuda().half()
                f3=b0(pixel_values=x,output_hidden_states=True).hidden_states[2][0]
                pred=prediction(forward(b5,state,x,recording=True))
                with np.load(CACHE/domain/p.name) as z:agree=float((pred==z['pred_b5']).mean())
                assert agree>.999,f'B5/cache mismatch {name}: {agree}'
                packed={'x':f3.cpu().numpy()}
                for k in ['mid','final']:
                    v=state.real[k]
                    assert v.shape==(1,2048,320),v.shape
                    packed[k]=v[0].T.reshape(320,32,64).cpu().numpy()
                tmp=p.with_suffix('.npz.part')
                with tmp.open('wb') as f:np.savez_compressed(f,**packed)
                tmp.replace(p);state.real={}
            if not backup.exists():
                tmp=backup.with_suffix('.npz.part');shutil.copy2(p,tmp);tmp.replace(backup)
            if (i+1)%10==0 or i+1==len(data):
                print(f'Targets {domain}: {i+1}/{len(data)}, {(time.monotonic()-start)/60:.1f} мин',flush=True)
    names=[p.stem for p,_ in SegSet('cs_fit').items]
    splits=split_ids(names)
    p=OUT/'splits.json'
    if p.exists():assert json.loads(p.read_text())==splits
    else:write_json(p,splits)
    print('Train:',len(splits['train']),'val:',len(splits['val']),'sequence:',splits['val_sequence'],flush=True)
    write_json(OUT/'targets_complete.json',{'complete':True,'domains':{'cs_fit':267,'cs_test':233}})


def ensure_local(domain,names):
    for n in names:
        p=target_path(domain,n)
        if not p.exists():
            p.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(OUT/'targets'/domain/p.name,p)
        check_target(p)


def normalization(names):
    p=OUT/'normalization.npz'
    if p.exists():
        with np.load(p) as z:return {k:z[k] for k in z.files}
    sums={k:np.zeros(d,np.float64) for k,d in [('x',160),('mid',320),('final',320)]}
    squares={k:v.copy() for k,v in sums.items()};n=0
    for name in names:
        with np.load(target_path('cs_fit',name)) as z:
            for k in sums:
                a=z[k].astype(np.float64).reshape(len(sums[k]),-1)
                sums[k]+=a.sum(1);squares[k]+=(a*a).sum(1)
            n+=a.shape[1]
    result={}
    for k in sums:
        mean=sums[k]/n
        std=np.sqrt(np.maximum(squares[k]/n-mean**2,1e-8))
        result[k+'_mean']=mean.astype(np.float32)
        result[k+'_std']=std.astype(np.float32)
    np.savez(p,**result)
    return result

class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(160,128,1),nn.GELU(),nn.Conv2d(128,640,1))
    def forward(self,x):return self.net(x)

class Maps(Dataset):
    def __init__(self,names,norm):self.names=names;self.norm=norm
    def __len__(self):return len(self.names)
    def __getitem__(self,i):
        with np.load(target_path('cs_fit',self.names[i])) as z:
            def standard(k):return (z[k].astype(np.float32)-self.norm[k+'_mean'][:,None,None])/self.norm[k+'_std'][:,None,None]
            return torch.from_numpy(standard('x')),torch.from_numpy(np.concatenate([standard('mid'),standard('final')]))

@torch.no_grad()
def val_loss(model,loader,device):
    model.eval();s=0.;n=0
    for x,y in loader:
        x=x.to(device);y=y.to(device)
        s+=F.mse_loss(model(x),y,reduction='sum').item();n+=y.numel()
    return s/n


def train(device="cuda"):
    torch.manual_seed(0);np.random.seed(0)
    splits=json.loads((OUT/'splits.json').read_text())
    ensure_local('cs_fit',splits['train']+splits['val'])
    norm=normalization(splits['train'])
    done=OUT/'training_complete.json'
    if done.exists():
        d=json.loads(done.read_text())
        assert d['checkpoint_sha256']==sha(OUT/'adapter_best.pt')
        print('Адаптер уже обучен, пропускаем.',flush=True);return
    model=Adapter().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
    val_loader=DataLoader(Maps(splits['val'],norm),batch_size=2,num_workers=0)
    last=OUT/'adapter_last.pt';best=float('inf');bad=0;history=[];start=0;best_epoch=None
    if last.exists():
        checkpoint=torch.load(last,map_location=device,weights_only=True)
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
        best=checkpoint['best'];bad=checkpoint['bad'];history=checkpoint['history'];start=checkpoint['epoch']
        best_epoch=checkpoint['best_epoch']
        print('Возобновляем после эпохи',start,flush=True)
    for epoch in range(start,CONFIG['epochs']):
        if bad>=CONFIG['patience']:break
        generator=torch.Generator().manual_seed(CONFIG['seed']+epoch)
        loader=DataLoader(Maps(splits['train'],norm),batch_size=CONFIG['batch'],shuffle=True,generator=generator,num_workers=0)
        model.train();total=0.;count=0
        for x,y in loader:
            x=x.to(device);y=y.to(device);optimizer.zero_grad(set_to_none=True)
            pred=model(x);loss=F.mse_loss(pred,y)
            assert torch.isfinite(loss),'Nonfinite loss'
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step();total+=loss.item()*y.numel();count+=y.numel()
        val=val_loss(model,val_loader,device)
        row={'epoch':epoch+1,'train_normalized_mse':total/count,'val_normalized_mse':val}
        history.append(row)
        if val<best:
            best=val;bad=0;best_epoch=epoch+1
            atomic_torch(OUT/'adapter_best.pt',{'model':model.state_dict(),'epoch':best_epoch,'val_mse':best})
        else:bad+=1
        atomic_torch(last,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1,
                          'best':best,'bad':bad,'best_epoch':best_epoch,'history':history})
        write_json(OUT/'training_history.json',history)
        print(row,flush=True)
    best_checkpoint=torch.load(OUT/'adapter_best.pt',map_location=device,weights_only=True)
    model.load_state_dict(best_checkpoint['model']);model.eval()
    diag={k:{'error':0.,'energy':0.,'mean_error':0.} for k in ['mid','final']}
    with torch.no_grad():
        for name in splits['val']:
            with np.load(target_path('cs_fit',name)) as z:
                x=(z['x'].astype(np.float32)-norm['x_mean'][:,None,None])/norm['x_std'][:,None,None]
                y=model(torch.from_numpy(x)[None].to(device))[0].cpu().numpy()
                for i,k in enumerate(['mid','final']):
                    target=z[k].astype(np.float64)
                    output=y[i*320:(i+1)*320]*norm[k+'_std'][:,None,None]+norm[k+'_mean'][:,None,None]
                    diag[k]['error']+=float(((output-target)**2).sum())
                    diag[k]['energy']+=float((target**2).sum())
                    diag[k]['mean_error']+=float(((norm[k+'_mean'][:,None,None]-target)**2).sum())
    diagnostics={k:{'relative_squared_error':d['error']/max(d['energy'],1e-12),
                    'error_vs_train_mean_baseline':d['error']/max(d['mean_error'],1e-12)} for k,d in diag.items()}
    write_json(OUT/'adapter_validation.json',diagnostics)
    summary={'complete':True,'best_epoch':best_epoch,'val_normalized_mse':best,
             'parameters':sum(p.numel() for p in model.parameters()),'train_frames':len(splits['train']),
             'val_frames':len(splits['val']),'checkpoint_sha256':sha(OUT/'adapter_best.pt'),
             'normalization_sha256':sha(OUT/'normalization.npz')}
    write_json(done,summary);print(summary,diagnostics,flush=True)

@torch.inference_mode()
def estimate(model,norm,array,device):
    x=(array.astype(np.float32)-norm['x_mean'][:,None,None])/norm['x_std'][:,None,None]
    y=model(torch.from_numpy(x)[None].to(device))[0]
    result={}
    for i,k in enumerate(['mid','final']):
        output=y[i*320:(i+1)*320]
        output=output*torch.as_tensor(norm[k+'_std'],device=device)[:,None,None]+torch.as_tensor(norm[k+'_mean'],device=device)[:,None,None]
        result[k]=output.flatten(1).T[None].to(dtype=torch.float16 if str(device).startswith('cuda') else torch.float32)
    return result


def rank(x):return x.ravel().argsort().argsort().reshape(x.shape)

@torch.inference_mode()
def evaluate(device="cuda"):
    summary=json.loads((OUT/'training_complete.json').read_text())
    assert summary['checkpoint_sha256']==sha(OUT/'adapter_best.pt')
    assert summary['normalization_sha256']==sha(OUT/'normalization.npz')
    adapter=Adapter().to(device).eval().requires_grad_(False)
    adapter.load_state_dict(torch.load(OUT/'adapter_best.pt',map_location=device,weights_only=True)['model'])
    with np.load(OUT/'normalization.npz') as z:norm={k:z[k] for k in z.files}
    model=load_model('b5',device=device).requires_grad_(False);state=install(model)
    feats=np.load(CACHE/'e8_tiles_cityscapes.npz')
    e8b_path=RESULTS.parent/'e8b_v1/e8b_cityscapes.json'
    old=json.loads(e8b_path.read_text()) if e8b_path.exists() else {}
    modes=['true_context_true_output','pred_context_true_output','true_context_pred_output','pred_context_pred_output','mean_context_mean_output']
    report_path=OUT/'e9_routing.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else {}
    for domain in CONFIG['evaluation_domains']:
        if domain in report:
            print(domain,'готов, пропускаем',flush=True);continue
        data=SegSet(domain);names=[p.stem for p,_ in data.items]
        ensure_local(domain,names)
        assert names==[p.stem for p in sorted((CACHE/domain).glob('*.npz'))]
        score=rank(feats[f'{domain}__entropy__mean'])+rank(feats[f'{domain}__knn__mean'])
        masks=[top_tiles(score,b) for b in CONFIG['budgets']]
        n=len(data);c=len(modes)+2
        confs=np.zeros((n,len(masks),c,19,19),np.int64)
        agreements=np.zeros((len(masks),len(modes)+1,2),np.int64)
        start=time.monotonic()
        dtype=torch.float16 if str(device).startswith('cuda') else torch.float32
        mean={k:torch.as_tensor(norm[k+'_mean'],device=device)[None,None,:].expand(1,2048,320).to(dtype=dtype) for k in ['mid','final']}
        for i,name in enumerate(names):
            img,_,_=data[i];img=img[None].to(device=device,dtype=dtype)
            full_pred=prediction(forward(model,state,img,recording=True))
            with np.load(CACHE/domain/f'{name}.npz') as z:gt=z['gt'];b0=z['pred_b0'];b5=z['pred_b5']
            assert (full_pred==b5).mean()>.999,f'B5/cache mismatch: {name}'
            with np.load(target_path(domain,name)) as z:pred=estimate(adapter,norm,z['x'],device)
            true=state.real
            options=[(true['mid'],true['final']),(pred['mid'],true['final']),
                     (true['mid'],pred['final']),(pred['mid'],pred['final']),(mean['mid'],mean['final'])]
            # All-true with predicted contexts/output must still reproduce the normal pass.
            if i==0:
                ones=torch.ones((1,2048,1),device=device,dtype=torch.bool)
                check=prediction(forward(model,state,img,mask=ones,context=pred['mid'],output=pred['final']))
                assert np.array_equal(check,full_pred),'All-true prediction mismatch'
            for j,m in enumerate(masks):
                px=m[i].repeat(TILE,0).repeat(TILE,1)
                tm=torch.from_numpy(m[i]).to(device).repeat_interleave(4,0).repeat_interleave(4,1).reshape(1,2048,1)
                confs[i,j,0]=confusion(b0,gt)
                confs[i,j,1]=confusion(np.where(px,b5,b0),gt)
                agreements[j,0]+=[int((b0==b5)[px].sum()),int(px.sum())]
                for k,(context,output) in enumerate(options):
                    modified=prediction(forward(model,state,img,mask=tm,context=context,output=output))
                    confs[i,j,k+2]=confusion(np.where(px,modified,b0),gt)
                    agreements[j,k+1]+=[int((modified==b5)[px].sum()),int(px.sum())]
            state.real={}
            if (i+1)%5==0 or i+1==n:
                elapsed=time.monotonic()-start
                print(f'E9 {domain}: {i+1}/{n}, {elapsed/60:.1f} мин, ETA ~{elapsed/(i+1)*(n-i-1)/60:.1f} мин',flush=True)
        rows=[]
        for j,budget in enumerate(CONFIG['budgets']):
            values=[float(miou(v)) for v in confs[:,j].sum(0)]
            for k,mode in enumerate(modes):
                a=agreements[j,k+1]
                row={'method':'rank','budget':budget,'variant':mode,'miou':values[k+2],
                     'miou_b0':values[0],'miou_full_tiles':values[1],
                     'delta_vs_b0_pp':values[k+2]-values[0],
                     'delta_vs_true_context_output_pp':values[k+2]-values[2],
                     'delta_vs_full_tiles_pp':values[k+2]-values[1],
                     'agree_b5_percent':100*int(a[0])/max(int(a[1]),1)}
                row['agree_b0_percent']=100*int(agreements[j,0,0])/max(int(agreements[j,0,1]),1)
                if k==0 and domain in old and f'rank@{budget}' in old[domain]:
                    previous=old[domain][f'rank@{budget}']
                    delta=row['miou']-previous['miou_mid']
                    row['control_delta_vs_e8b_mid_pp']=delta
                    assert abs(delta)<.01,f'True-control differs from E8b mid: {delta}'
                rows.append(row);print(row,flush=True)
        np.savez_compressed(OUT/f'{domain}_frame_confusions.npz',confs=confs,
                            names=np.array(names),budgets=np.array(CONFIG['budgets']),
                            modes=np.array(['b0','full_tiles']+modes),agreements=agreements)
        report[domain]=rows;write_json(report_path,report)
    flat=[dict(domain=d,**r) for d,rows in report.items() for r in rows]
    fields=list(dict.fromkeys(k for r in flat for k in r))
    with (OUT/'e9_routing.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(flat)


def bootstrap():
    # Paired dataset mIoU, recomputed from summed confusion matrices.
    rows=[];rng=np.random.default_rng(0)
    for domain in CONFIG['evaluation_domains']:
        with np.load(OUT/f'{domain}_frame_confusions.npz') as z:
            conf=z['confs'];names=z['names'];modes=list(z['modes']);budgets=z['budgets']
        for unit in ['frame','sequence']:
            groups=np.arange(len(names)).astype(str) if unit=='frame' else np.array(['_'.join(n.split('_')[:2]) for n in names])
            unique=sorted(set(groups));grouped=np.stack([conf[groups==g].sum(0) for g in unique])
            for j,budget in enumerate(budgets):
                for reference in ['true_context_true_output','mean_context_mean_output','full_tiles']:
                    a=modes.index('pred_context_pred_output');b=modes.index(reference)
                    diffs=[]
                    for _ in range(1000):
                        idx=rng.integers(0,len(unique),len(unique));c=grouped[idx,j].sum(0)
                        diffs.append(miou(c[a])-miou(c[b]))
                    total=grouped[:,j].sum(0)
                    ci=np.percentile(diffs,[2.5,97.5])
                    rows.append({'domain':domain,'budget':float(budget),'unit':unit,'groups':len(unique),
                                 'comparison':f'pred_both minus {reference}',
                                 'delta_miou_pp':float(miou(total[a])-miou(total[b])),
                                 'ci_low':float(ci[0]),'ci_high':float(ci[1])})
    write_json(OUT/'bootstrap.json',rows)
    for row in rows:print(row,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['export_targets','train','evaluate','bootstrap'])
    args=parser.parse_args();initialize()
    if args.stage!='bootstrap':assert torch.cuda.is_available(),'Need GPU'
    import transformers,platform
    write_json(OUT/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,
        'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    globals()[args.stage]()
if __name__=='__main__':main()
