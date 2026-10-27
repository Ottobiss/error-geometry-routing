"""E9-k4: B0 F3 -> four midpoint contexts plus final. Teacher-assisted diagnostic."""
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

OUT=Path(os.environ['E9K_OUT'])
LOCAL=Path(os.environ['E9K_LOCAL'])
PREVIOUS=Path(os.environ.get('E9K_PREVIOUS',str(RESULTS.parent/'e9_adapter_v1')))
STATES=['s5','s15','s25','s35','final']
CONTEXTS=STATES[:-1]
FEATURE_MODELS={}
CONFIG={'schema':'e9-k4-five-state-adapter-v1','seed':0,'input':'B0 normalized encoder F3 (stage index 2 output)',
        'input_channels':160,'target_channels':320,'hidden':128,'target_states':STATES,
        'epochs':20,'patience':4,'lr':.001,'weight_decay':.0001,'batch':2,
        'budgets':[.1,.2],'evaluation_domains':['cs_test'],'methods':['rank'],
        'teacher_assisted':True,'keep_real_b5_early_stages':True,'routed_entry':'real B5',
        'loss':'0.5 mean normalized MSE over four contexts + 0.5 normalized final MSE',
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
             tile_features_sha256=sha(CACHE/'e8_tiles_cityscapes.npz'),
             previous_split_sha256=sha(PREVIOUS/'splits.json'))
    if (PREVIOUS/'config.json').exists():
        old=json.loads((PREVIOUS/'config.json').read_text())
        assert old['common_sha256']==cfg['common_sha256'] and old['tile_features_sha256']==cfg['tile_features_sha256'], 'Previous E9 used different preprocessing or tile features'
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
            if self.index in [5,15,25,35]:s.real['s'+str(self.index)]=x
            out=self.layer(x,*args,**kwargs)
            if self.index==self.n-1:s.real['final']=out[0]
            return out
        if s.mask is None:return self.layer(x,*args,**kwargs)
        ctx=s.contexts[CONTEXTS[self.index*4//self.n]]
        out=self.layer(torch.where(s.mask,x,ctx),*args,**kwargs)
        y=torch.where(s.mask,out[0],ctx)
        if self.index==self.n-1:y=torch.where(s.mask,y,s.output)
        return (y,)+tuple(out[1:])


def install(model):
    state=SimpleNamespace(recording=False,real={},mask=None,contexts=None,output=None)
    layers=model.segformer.encoder.block[2]
    assert len(layers)==40,'Expected B5 stage3 with 40 blocks'
    for i,layer in enumerate(layers):layers[i]=ContextBlock(layer,state,i,len(layers))
    return state

@torch.inference_mode()
def forward(model,state,img,recording=False,mask=None,contexts=None,output=None):
    state.recording=recording;state.mask=mask;state.contexts=contexts;state.output=output
    try:return model(pixel_values=img).logits
    finally:state.recording=False;state.mask=None;state.contexts=None;state.output=None


def prediction(logits):
    return F.interpolate(logits.float(),SIZE,mode='bilinear',align_corners=False).argmax(1)[0].byte().cpu().numpy()

def target_path(domain,name):return LOCAL/'targets'/domain/f'{name}.npz'

def check_target(path):
    if not path.is_file():return False
    with np.load(path,allow_pickle=False) as z:
        assert z['x'].shape==(160,32,64)
        assert all(z[k].shape==(320,32,64) for k in STATES)
        assert all(np.isfinite(z[k]).all() for k in ['x']+STATES)
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
def feature(domain,name,img=None,device='cuda'):
    p=LOCAL/'b0_features'/domain/(name+'.npy')
    if p.is_file():
        a=np.load(p,allow_pickle=False)
    else:
        candidates=[PREVIOUS/'targets'/domain/(name+'.npz'),
                    LOCAL.parent/'e9_adapter_v1/targets'/domain/(name+'.npz')]
        source=next((x for x in candidates if x.is_file()),None)
        if source:
            with np.load(source,allow_pickle=False) as z:a=z['x']
        else:
            assert img is not None,'Need image to regenerate missing B0 F3'
            if device not in FEATURE_MODELS:
                FEATURE_MODELS[device]=load_model('b0',device=device).requires_grad_(False)
            a=FEATURE_MODELS[device](pixel_values=img,output_hidden_states=True).hidden_states[2][0].cpu().numpy()
        p.parent.mkdir(parents=True,exist_ok=True)
        temp=p.with_suffix('.npy.part')
        with temp.open('wb') as f:np.save(f,a)
        temp.replace(p)
    assert a.shape==(160,32,64) and np.isfinite(a).all()
    return a


def normalized_loss(pred,target):
    assert pred.shape[1]==target.shape[1]==1600
    return .5*F.mse_loss(pred[:,:1280],target[:,:1280])+.5*F.mse_loss(pred[:,1280:],target[:,1280:])

@torch.inference_mode()
def export_targets():
    b5=load_model('b5').requires_grad_(False);state=install(b5)
    assert len(b5.segformer.encoder.block[2])==40,'Expected B5 stage 3 depth=40'
    for domain in ['cs_fit']:
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
                f3=feature(domain,name,x,'cuda')
                pred=prediction(forward(b5,state,x,recording=True))
                with np.load(CACHE/domain/p.name) as z:agree=float((pred==z['pred_b5']).mean())
                assert agree>.999,f'B5/cache mismatch {name}: {agree}'
                packed={'x':f3}
                for k in STATES:
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
    expected=split_ids(names)
    splits=json.loads((PREVIOUS/'splits.json').read_text())
    assert splits['train']==expected['train'] and splits['val']==expected['val'], 'Old split differs'
    p=OUT/'splits.json'
    if p.exists():assert json.loads(p.read_text())==splits
    else:write_json(p,splits)
    print('Train:',len(splits['train']),'val:',len(splits['val']),'sequence:',splits['val_sequence'],flush=True)
    write_json(OUT/'targets_complete.json',{'complete':True,'domains':{'cs_fit':267},'cs_test_targets_exported':False})


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
    sums={k:np.zeros(d,np.float64) for k,d in [('x',160)]+[(name,320) for name in STATES]}
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
        self.net=nn.Sequential(nn.Conv2d(160,128,1),nn.GELU(),nn.Conv2d(128,1600,1))
    def forward(self,x):return self.net(x)

class Maps(Dataset):
    def __init__(self,names,norm):self.names=names;self.norm=norm
    def __len__(self):return len(self.names)
    def __getitem__(self,i):
        with np.load(target_path('cs_fit',self.names[i])) as z:
            def standard(k):return (z[k].astype(np.float32)-self.norm[k+'_mean'][:,None,None])/self.norm[k+'_std'][:,None,None]
            return torch.from_numpy(standard('x')),torch.from_numpy(np.concatenate([standard(k) for k in STATES]))

@torch.no_grad()
def val_loss(model,loader,device):
    model.eval();s=0.;n=0
    for x,y in loader:
        x=x.to(device);y=y.to(device)
        s+=normalized_loss(model(x),y).item()*len(x);n+=len(x)
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
            pred=model(x);loss=normalized_loss(pred,y)
            assert torch.isfinite(loss),'Nonfinite loss'
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step();total+=loss.item()*len(x);count+=len(x)
        val=val_loss(model,val_loader,device)
        row={'epoch':epoch+1,'train_weighted_normalized_mse':total/count,'val_weighted_normalized_mse':val}
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
    diag={k:{'error':0.,'energy':0.,'mean_error':0.} for k in STATES}
    with torch.no_grad():
        for name in splits['val']:
            with np.load(target_path('cs_fit',name)) as z:
                x=(z['x'].astype(np.float32)-norm['x_mean'][:,None,None])/norm['x_std'][:,None,None]
                y=model(torch.from_numpy(x)[None].to(device))[0].cpu().numpy()
                for i,k in enumerate(STATES):
                    target=z[k].astype(np.float64)
                    output=y[i*320:(i+1)*320]*norm[k+'_std'][:,None,None]+norm[k+'_mean'][:,None,None]
                    diag[k]['error']+=float(((output-target)**2).sum())
                    diag[k]['energy']+=float((target**2).sum())
                    diag[k]['mean_error']+=float(((norm[k+'_mean'][:,None,None]-target)**2).sum())
    diagnostics={k:{'relative_squared_error':d['error']/max(d['energy'],1e-12),
                    'error_vs_train_mean_baseline':d['error']/max(d['mean_error'],1e-12)} for k,d in diag.items()}
    write_json(OUT/'adapter_validation.json',diagnostics)
    summary={'complete':True,'best_epoch':best_epoch,'val_weighted_normalized_mse':best,
             'parameters':sum(p.numel() for p in model.parameters()),'train_frames':len(splits['train']),
             'val_frames':len(splits['val']),'checkpoint_sha256':sha(OUT/'adapter_best.pt'),
             'normalization_sha256':sha(OUT/'normalization.npz')}
    write_json(done,summary);print(summary,diagnostics,flush=True)

@torch.inference_mode()
def estimate(model,norm,array,device):
    x=(array.astype(np.float32)-norm['x_mean'][:,None,None])/norm['x_std'][:,None,None]
    y=model(torch.from_numpy(x)[None].to(device))[0]
    result={}
    for i,k in enumerate(STATES):
        output=y[i*320:(i+1)*320]
        output=output*torch.as_tensor(norm[k+'_std'],device=device)[:,None,None]+torch.as_tensor(norm[k+'_mean'],device=device)[:,None,None]
        result[k]=output.flatten(1).T[None].to(dtype=torch.float16 if str(device).startswith('cuda') else torch.float32)
    return result


def rank(x):return x.ravel().argsort().argsort().reshape(x.shape)


def csv_rows(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def frame_controls():
    """Exact floor/ceil counts around the *arithmetic* k4 budget; no linear mIoU interpolation."""
    cost_root=RESULTS.parent/'e8c_k_sweep_v1'
    measured=json.loads((cost_root/'cost_measured.json').read_text())
    scenarios=json.loads((cost_root/'cost_scenarios.json').read_text())['rows']
    b0_cost=measured['models']['b0']['fvcore']['counted_total_gops']
    b5_cost=measured['models']['b5']['fvcore']['counted_total_gops']
    data=SegSet('cs_test');names=[p.stem for p,_ in data.items];n=len(names)
    assert names==[p.stem for p in sorted((CACHE/'cs_test').glob('*.npz'))]
    with np.load(CACHE/'e8_tiles_cityscapes.npz') as z:
        entropy=z['cs_test__entropy__mean'];knn=z['cs_test__knn__mean']
    scores={'raw_entropy':entropy.mean((1,2)),'rank_mean':(rank(entropy)+rank(knn)).mean((1,2))}
    conf=np.zeros((n,2,19,19),np.int64)
    for i,name in enumerate(names):
        with np.load(CACHE/'cs_test'/(name+'.npz')) as z:
            conf[i,0]=confusion(z['pred_b0'],z['gt']);conf[i,1]=confusion(z['pred_b5'],z['gt'])
    rows=[];per_frame=[];labels=[]
    for b in CONFIG['budgets']:
        scenario=next(r for r in scenarios if r['k']==4 and r['budget']==b)
        assert scenario['adapter_parameters_scenario']==227008
        target=float(scenario['hybrid_planning_count_halo_aware'])
        fraction=(target-b0_cost)/b5_cost
        assert 0<=fraction<=1
        raw=fraction*n
        counts=[('floor',int(np.floor(raw))),('ceil',int(np.ceil(raw)))]
        for method,score in scores.items():
            order=np.argsort(-score,kind='stable')
            for side,count in counts:
                mask=np.zeros(n,bool);mask[order[:count]]=True
                matrices=np.where(mask[:,None,None],conf[:,1],conf[:,0])
                actual=b0_cost+float(mask.mean())*b5_cost
                label=f'{method}_b{b}_{side}'
                rows.append({'label':label,'tile_budget_reference':b,'method':method,'bracket':side,
                             'target_frame_fraction':fraction,'actual_frame_fraction':float(mask.mean()),
                             'selected_frames':count,'cached_miou':float(miou(matrices.sum(0))),
                             'target_hybrid_count':target,'actual_counted_gops':actual,
                             'count_difference':actual-target,
                             'scope':'arithmetic-budget bracket, not matched measured latency'})
                per_frame.append(matrices);labels.append(label)
    np.savez_compressed(OUT/'frame_controls.npz',confs=np.stack(per_frame,axis=1),names=np.array(names),labels=np.array(labels))
    write_json(OUT/'frame_controls.json',{'rows':rows,'cost_source':str(cost_root),
        'cost_source_sha256':sha(cost_root/'cost_scenarios.json'),
        'warning':'The ROI target is an unimplemented hybrid arithmetic scenario. Whole-frame quality is cached. This is not a measured equal-time comparison.'})
    csv_rows(OUT/'frame_controls.csv',rows)
    for row in rows:print(row,flush=True)


def reference_k4(domain):
    path=RESULTS.parent/'e8c_k_sweep_v1/e8c_quality.json'
    rows=json.loads(path.read_text())[domain]['rows']
    refs={float(r['budget']):float(r['miou']) for r in rows if r['variant']=='k4'}
    assert all(b in refs for b in CONFIG['budgets'])
    return refs


def save_eval_progress(path,completed,confs,agreements,names,modes,model_sha):
    p=path.with_suffix('.npz.part')
    with p.open('wb') as f:
        np.savez_compressed(f,completed=np.array(completed),confs=confs[:completed],agreements=agreements,
                            names=np.array(names),modes=np.array(['b0','full_tiles']+modes),
                            budgets=np.array(CONFIG['budgets']),checkpoint_sha256=np.array(model_sha))
    p.replace(path)

@torch.inference_mode()
def evaluate(device='cuda'):
    summary=json.loads((OUT/'training_complete.json').read_text());model_sha=sha(OUT/'adapter_best.pt')
    assert summary['checkpoint_sha256']==model_sha
    assert summary['normalization_sha256']==sha(OUT/'normalization.npz')
    report_path=OUT/'routing.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else {'checkpoint_sha256':model_sha}
    assert report['checkpoint_sha256']==model_sha,'Checkpoint changed: use new output directory'
    adapter=Adapter().to(device).eval().requires_grad_(False)
    adapter.load_state_dict(torch.load(OUT/'adapter_best.pt',map_location=device,weights_only=True)['model'])
    with np.load(OUT/'normalization.npz') as z:norm={k:z[k] for k in z.files}
    model=load_model('b5',device=device).requires_grad_(False)
    state=install(model);dtype=torch.float16 if str(device).startswith('cuda') else torch.float32
    modes=['true_context_true_output','pred_context_true_output','true_context_pred_output',
           'pred_context_pred_output','mean_context_mean_output']
    previous_path=PREVIOUS/'e9_routing.json'
    previous=json.loads(previous_path.read_text()) if previous_path.exists() else {}
    with np.load(CACHE/'e8_tiles_cityscapes.npz') as z:feats={k:z[k] for k in z.files}
    for domain in CONFIG['evaluation_domains']:
        if domain in report:
            print(domain,'готов, пропускаем',flush=True);continue
        refs=reference_k4(domain)
        data=SegSet(domain);names=[p.stem for p,_ in data.items];n=len(names)
        assert names==[p.stem for p in sorted((CACHE/domain).glob('*.npz'))]
        score=rank(feats[f'{domain}__entropy__mean'])+rank(feats[f'{domain}__knn__mean'])
        masks=[top_tiles(score,b) for b in CONFIG['budgets']]
        confs=np.zeros((n,len(masks),len(modes)+2,19,19),np.int64)
        agreements=np.zeros((len(masks),len(modes)+1,2),np.int64)
        progress=OUT/f'{domain}_progress.npz';done=0
        if progress.exists():
            with np.load(progress,allow_pickle=False) as z:
                assert str(z['checkpoint_sha256'])==model_sha
                assert list(z['names'])==names and list(z['modes'])==['b0','full_tiles']+modes
                assert np.array_equal(z['budgets'],CONFIG['budgets'])
                done=int(z['completed']);assert 0<=done<=n
                confs[:done]=z['confs'];agreements[:]=z['agreements']
            print('Продолжаем оценку после',done,'кадров',flush=True)
        mean={k:torch.as_tensor(norm[k+'_mean'],device=device)[None,None,:].expand(1,2048,320).to(dtype=dtype) for k in STATES}
        start=time.monotonic()
        for i in range(done,n):
            name=names[i];img,_,_=data[i];img=img[None].to(device=device,dtype=dtype)
            full_logits=forward(model,state,img,recording=True);full_pred=prediction(full_logits)
            with np.load(CACHE/domain/(name+'.npz')) as z:gt=z['gt'];b0=z['pred_b0'];b5=z['pred_b5']
            assert (full_pred==b5).mean()>.999,f'B5/cache mismatch {name}'
            pred=estimate(adapter,norm,feature(domain,name,img,device),device)
            true=state.real
            options=[(true,true['final']),(pred,true['final']),(true,pred['final']),
                     (pred,pred['final']),(mean,mean['final'])]
            if i==done:
                ones=torch.ones((1,2048,1),device=device,dtype=torch.bool)
                check=forward(model,state,img,mask=ones,contexts=pred,output=pred['final'])
                assert torch.equal(check,full_logits),'All-true changed logits'
                write_json(OUT/'sanity.json',{'all_true_predicted_states_equal':True,'first_checked_image':name})
            for j,m in enumerate(masks):
                px=m[i].repeat(TILE,0).repeat(TILE,1)
                tm=torch.from_numpy(m[i]).to(device).repeat_interleave(4,0).repeat_interleave(4,1).reshape(1,2048,1)
                confs[i,j,0]=confusion(b0,gt);confs[i,j,1]=confusion(np.where(px,b5,b0),gt)
                agreements[j,0]+=[int((b0==b5)[px].sum()),int(px.sum())]
                for k,(contexts,output) in enumerate(options):
                    modified=prediction(forward(model,state,img,mask=tm,contexts=contexts,output=output))
                    confs[i,j,k+2]=confusion(np.where(px,modified,b0),gt)
                    agreements[j,k+1]+=[int((modified==b5)[px].sum()),int(px.sum())]
            state.real={}
            if (i+1)%10==0 or i+1==n:save_eval_progress(progress,i+1,confs,agreements,names,modes,model_sha)
            if (i+1)%5==0 or i+1==n:
                elapsed=time.monotonic()-start
                print(f'E9-k4 {domain}: {i+1}/{n}, {elapsed/60:.1f} мин, ETA ~{elapsed/(i+1-done)*(n-i-1)/60:.1f} мин',flush=True)
        checks=[{'budget':b,'delta_vs_k4_pp':float(miou(confs[:,j,2].sum(0)))-refs[b]} for j,b in enumerate(CONFIG['budgets'])]
        write_json(OUT/'reference_check.json',checks)
        assert all(abs(r['delta_vs_k4_pp'])<.01 for r in checks),'Teacher k4 control differs from E8c'
        rows=[]
        for j,b in enumerate(CONFIG['budgets']):
            values=[float(miou(v)) for v in confs[:,j].sum(0)]
            old=next((r['miou'] for r in previous.get(domain,[]) if r['budget']==b and r['variant']=='pred_context_pred_output'),None)
            for k,mode in enumerate(modes):
                a=agreements[j,k+1]
                row={'domain':domain,'budget':b,'variant':mode,'miou':values[k+2],
                     'delta_vs_b0_pp':values[k+2]-values[0],
                     'delta_vs_true_k4_pp':values[k+2]-values[2],
                     'delta_vs_full_tiles_pp':values[k+2]-values[1],
                     'agree_b5_percent':100*int(a[0])/max(int(a[1]),1),
                     'previous_e9_pred_both_miou':old}
                if old is not None:row['delta_vs_previous_e9_pred_both_pp']=values[k+2]-old
                rows.append(row);print(row,flush=True)
        report[domain]=rows;write_json(report_path,report)
    csv_rows(OUT/'routing.csv',[r for d in CONFIG['evaluation_domains'] for r in report[d]])


def bootstrap():
    report=json.loads((OUT/'routing.json').read_text())
    with np.load(OUT/'frame_controls.npz') as z:frame_confs=z['confs'];frame_names=z['names'];labels=list(z['labels'])
    frame_meta=json.loads((OUT/'frame_controls.json').read_text())['rows']
    old_path=PREVIOUS/'cs_test_frame_confusions.npz'
    old=None
    if old_path.exists():
        with np.load(old_path) as z:old={k:z[k] for k in z.files}
    rng=np.random.default_rng(0);rows=[]
    for domain in CONFIG['evaluation_domains']:
        assert domain in report
        with np.load(OUT/f'{domain}_progress.npz') as z:
            conf=z['confs'];names=z['names'];modes=list(z['modes']);budgets=z['budgets']
            assert len(conf)==len(names)==int(z['completed'])
        assert np.array_equal(names,frame_names)
        for unit in ['frame','sequence']:
            ids=np.arange(len(names)).astype(str) if unit=='frame' else np.array(['_'.join(n.split('_')[:2]) for n in names])
            unique=sorted(set(ids))
            def group(x):return np.stack([x[ids==g].sum(0) for g in unique])
            for j,b in enumerate(budgets):
                ours=conf[:,j,modes.index('pred_context_pred_output')]
                references={mode:conf[:,j,modes.index(mode)] for mode in ['true_context_true_output','mean_context_mean_output','full_tiles']}
                for item in frame_meta:
                    if item['tile_budget_reference']==b:references['frame_'+item['label']]=frame_confs[:,labels.index(item['label'])]
                if old is not None:
                    assert np.array_equal(old['names'],names)
                    oj=list(old['budgets']).index(b);ok=list(old['modes']).index('pred_context_pred_output')
                    references['previous_e9_pred_both']=old['confs'][:,oj,ok]
                a=group(ours)
                for label,reference in references.items():
                    other=group(reference);diff=[]
                    for _ in range(1000):
                        idx=rng.integers(0,len(unique),len(unique))
                        diff.append(miou(a[idx].sum(0))-miou(other[idx].sum(0)))
                    ci=np.percentile(diff,[2.5,97.5])
                    rows.append({'domain':domain,'budget':float(b),'unit':unit,'groups':len(unique),
                                 'comparison':'pred_all minus '+label,
                                 'delta_miou_pp':float(miou(a.sum(0))-miou(other.sum(0))),
                                 'ci_low':float(ci[0]),'ci_high':float(ci[1]),
                                 'scope':'paired quality comparison; frame controls matched by arithmetic proxy only'})
    write_json(OUT/'bootstrap.json',rows);csv_rows(OUT/'bootstrap.csv',rows)
    for row in rows:print(row,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['frame_controls','export_targets','train','evaluate','bootstrap'])
    args=parser.parse_args();initialize()
    if args.stage not in ['frame_controls','bootstrap']:assert torch.cuda.is_available(),'Need GPU'
    import transformers,platform
    write_json(OUT/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,
        'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    globals()[args.stage]()
if __name__=='__main__':main()
