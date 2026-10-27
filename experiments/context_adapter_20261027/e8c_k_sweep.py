"""E8c: piecewise frozen context, teacher states only. No training/speed claims."""
import os
os.environ['USE_TF']='0'
import argparse,csv,hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from common import CACHE,RESULTS,SIZE,TILE,SegSet,load_model,confusion,miou,top_tiles
OUT=Path(os.environ['E8C_OUT'])
CONFIG={'schema':'e8c-midpoint-k-sweep-v1','domains':['cs_test'],'methods':['rank'],'budgets':[.1,.2],
        'variants':['k1','k2','k4','k8'], 'training':False,
        'restore':'true B5 final outside selected tiles','early_stages':'full B5',
        'ks':[1,2,4,8],'checkpoint_every_frames':10,'bootstrap_repeats':1000,'reference_states':'true B5 per image'}

def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    p=path.with_suffix(path.suffix+'.part');p.write_text(json.dumps(data,indent=2));p.replace(path)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def initialize():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=dict(CONFIG,source_sha256=sha(__file__),common_sha256=sha(Path(__file__).parent/'common.py'),
             tile_features_sha256=sha(CACHE/'e8_tiles_cityscapes.npz'))
    p=OUT/'quality_config.json'
    if p.exists():assert json.loads(p.read_text())==cfg,'Use a new E8C_OUT for changed configuration'
    else:write_json(p,cfg)


def sources(k,n):
    assert 1<=k<=n
    return [int((i*k//n+.5)*n/k) for i in range(n)]

def context_index(i,n,mode):
    if mode=='per_block':return i
    if mode=='mid':return n//2
    return sources(int(mode[1:]),n)[i]


class Frozen(nn.Module):
    def __init__(self,layer,state,index,n):
        super().__init__();self.layer=layer;self.state=state;self.index=index;self.n=n
    def forward(self,x,*args,**kwargs):
        s=self.state
        if s.recording:
            s.inputs[self.index]=x
            out=self.layer(x,*args,**kwargs)
            if self.index==self.n-1:s.final=out[0]
            return out
        if s.mask is None:return self.layer(x,*args,**kwargs)
        ctx=s.inputs[context_index(self.index,self.n,s.mode)]
        out=self.layer(torch.where(s.mask,x,ctx),*args,**kwargs)
        y=torch.where(s.mask,out[0],ctx)
        if self.index==self.n-1:y=torch.where(s.mask,y,s.final)
        return (y,)+tuple(out[1:])

def install(model):
    blocks=model.segformer.encoder.block[2]
    state=SimpleNamespace(recording=False,mask=None,mode='mid',inputs=[None]*len(blocks),final=None)
    for i,block in enumerate(blocks):blocks[i]=Frozen(block,state,i,len(blocks))
    return state

@torch.inference_mode()
def logits(model,state,img,recording=False,mask=None,mode='mid'):
    state.recording=recording;state.mask=mask;state.mode=mode
    try:return model(pixel_values=img).logits
    finally:state.recording=False;state.mask=None

def prediction(z):
    return F.interpolate(z.float(),SIZE,mode='bilinear',align_corners=False).argmax(1)[0].byte().cpu().numpy()

def mask_tokens(m,device):
    k=TILE//16
    return torch.as_tensor(m,device=device).repeat_interleave(k,0).repeat_interleave(k,1).reshape(1,-1,1)

def rank(x):return x.ravel().argsort().argsort().reshape(x.shape)

def reference_values(domain):
    """Require the actual prior report, rather than trusting hardcoded rounded numbers."""
    p=RESULTS.parent/'e9_adapter_v1/e9_routing.json'
    if p.exists():
        data=json.loads(p.read_text());rows=data.get(domain,[])
        refs={float(r['budget']):float(r['miou']) for r in rows if r['variant']=='true_context_true_output' and r['method']=='rank'}
        if all(b in refs for b in CONFIG['budgets']):return refs,str(p)
    p=RESULTS.parent/'e8b_v1/e8b_cityscapes.json'
    if p.exists():
        data=json.loads(p.read_text()).get(domain,{})
        if all(f'rank@{b}' in data for b in CONFIG['budgets']):
            return {b:float(data[f'rank@{b}']['miou_mid']) for b in CONFIG['budgets']},str(p)
    raise RuntimeError('Не найден эталон E9/E8b для k=1. Восстанови e9_routing.json или e8b_cityscapes.json на Drive.')

@torch.inference_mode()
def strict_sanity(model,state,img,plain_logits,cached):
    recorded=logits(model,state,img,recording=True)
    assert torch.equal(recorded,plain_logits),'Recording changed the ordinary pass'
    baseline=prediction(plain_logits)
    assert (baseline==cached).mean()>.999,'First-frame cache mismatch'
    ones=mask_tokens(np.ones((8,16),bool),img.device)
    for mode in CONFIG['variants']:
        test=logits(model,state,img,mask=ones,mode=mode)
        assert torch.equal(test,plain_logits),'All-true mismatch: '+mode
    checker=(np.indices((8,16)).sum(axis=0)%2)==0  # exactly half the tiles
    test=logits(model,state,img,mask=mask_tokens(checker,img.device),mode='per_block')
    error=float((test.float()-plain_logits.float()).abs().max().item())
    assert torch.allclose(test,plain_logits,rtol=1e-4,atol=1e-4),f'k=40 replay logits mismatch: {error}'
    assert np.array_equal(prediction(test),baseline),'k=40 replay must reproduce the WHOLE prediction'
    state.inputs=[None]*len(state.inputs);state.final=None
    return {'recording_equal':True,'all_true_equal_for_all_k':True,'k40_half_tiles_whole_prediction_equal':True,
            'k40_max_abs_logit_error':error,'cached_agreement':float((baseline==cached).mean())}


def save_progress(path,completed,confs,agree,names):
    temp=path.with_suffix('.npz.part')
    with temp.open('wb') as f:
        np.savez_compressed(f,completed=np.array(completed),confs=confs[:completed],agreements=agree,
                            names=np.array(names),modes=np.array(['b0','full_tiles']+CONFIG['variants']),
                            budgets=np.array(CONFIG['budgets']))
    temp.replace(path)

@torch.inference_mode()
def quality(device='cuda'):
    report_path=OUT/'e8c_quality.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else {}
    feats=np.load(CACHE/'e8_tiles_cityscapes.npz')
    dtype=torch.float16 if str(device).startswith('cuda') else torch.float32
    model=load_model('b5',device=device).requires_grad_(False)
    first_domain=CONFIG['domains'][0];first=SegSet(first_domain)
    assert len(first)>0,'No images'
    first_img=first[0][0][None].to(device=device,dtype=dtype)
    plain=model(pixel_values=first_img).logits
    state=install(model);nblocks=len(state.inputs)
    assert nblocks==40,f'Expected B5 depth40, got {nblocks}'
    with np.load(CACHE/first_domain/(first.items[0][0].stem+'.npz')) as z:cached=z['pred_b5']
    sanity=strict_sanity(model,state,first_img,plain,cached)
    write_json(OUT/'sanity.json',sanity)
    del first_img,plain
    for domain in CONFIG['domains']:
        if domain in report:
            print(domain,'готов, пропускаем',flush=True);continue
        reference,reference_source=reference_values(domain)
        dataset=SegSet(domain);names=[p.stem for p,_ in dataset.items]
        assert names==[p.stem for p in sorted((CACHE/domain).glob('*.npz'))]
        scores=rank(feats[f'{domain}__entropy__mean'])+rank(feats[f'{domain}__knn__mean'])
        masks=[top_tiles(scores,b) for b in CONFIG['budgets']]
        modes=CONFIG['variants'];c=2+len(modes);n=len(dataset)
        confs=np.zeros((n,len(masks),c,19,19),np.int64)
        agree=np.zeros((len(masks),len(modes)+1,2),np.int64)
        progress_path=OUT/f'{domain}_progress.npz';done=0
        if progress_path.exists():
            with np.load(progress_path,allow_pickle=False) as z:
                assert list(z['names'])==names,'Progress image order differs'
                assert list(z['modes'])==['b0','full_tiles']+modes
                assert np.array_equal(z['budgets'],CONFIG['budgets'])
                done=int(z['completed']);assert 0<=done<=n
                assert z['confs'].shape==confs[:done].shape
                confs[:done]=z['confs'];agree[:]=z['agreements']
            print(f'Продолжаем {domain} после {done}/{n} кадров',flush=True)
        start=time.monotonic()
        for i in range(done,n):
            name=names[i];img,_,_=dataset[i];img=img[None].to(device=device,dtype=dtype)
            full=logits(model,state,img,recording=True);pred_full=prediction(full)
            with np.load(CACHE/domain/(name+'.npz')) as z:gt=z['gt'];b0=z['pred_b0'];b5=z['pred_b5']
            assert (pred_full==b5).mean()>.999,f'Cache mismatch: {name}'
            for j,m in enumerate(masks):
                px=m[i].repeat(TILE,0).repeat(TILE,1);tm=mask_tokens(m[i],device)
                confs[i,j,0]=confusion(b0,gt);confs[i,j,1]=confusion(np.where(px,b5,b0),gt)
                agree[j,0]+=[int((b0==b5)[px].sum()),int(px.sum())]
                for k,mode in enumerate(modes):
                    modified=prediction(logits(model,state,img,mask=tm,mode=mode))
                    confs[i,j,k+2]=confusion(np.where(px,modified,b0),gt)
                    agree[j,k+1]+=[int((modified==b5)[px].sum()),int(px.sum())]
            state.inputs=[None]*nblocks;state.final=None
            if (i+1)%CONFIG['checkpoint_every_frames']==0 or i+1==n:
                save_progress(progress_path,i+1,confs,agree,names)
            if (i+1)%5==0 or i+1==n:
                elapsed=time.monotonic()-start
                eta=elapsed/(i+1-done)*(n-i-1)/60
                print(f'E8c-k {domain}: {i+1}/{n}, {elapsed/60:.1f} мин, ETA ~{eta:.1f} мин',flush=True)
        # Fail before publishing any interpretation if the fixed k=1 reference changed.
        checks=[]
        for j,b in enumerate(CONFIG['budgets']):
            actual=float(miou(confs[:,j,2].sum(0)))
            delta=actual-reference[b]
            checks.append({'budget':b,'actual_k1_miou':actual,'reference_miou':reference[b],
                           'delta_pp':delta,'ok':abs(delta)<.01})
        write_json(OUT/f'{domain}_reference_check.json',{'source':reference_source,'checks':checks})
        assert all(r['ok'] for r in checks),'MISMATCH: k=1 differs from actual E9/E8b report; stop and inspect cache/model/masks'
        rows=[]
        for j,b in enumerate(CONFIG['budgets']):
            values=[float(miou(z)) for z in confs[:,j].sum(0)]
            for k,mode in enumerate(modes):
                a=agree[j,k+1];full_gain=values[1]-values[0]
                row={'budget':b,'method':'rank','variant':mode,'k':int(mode[1:]),'miou':values[k+2],
                     'gain_vs_b0_pp':values[k+2]-values[0],'miou_b0':values[0],'miou_full_tiles':values[1],
                     'delta_vs_full_tiles_pp':values[k+2]-values[1],'delta_vs_k1_pp':values[k+2]-values[2],
                     'recovered_full_gain_pct':100*(values[k+2]-values[0])/full_gain if abs(full_gain)>1e-12 else None,
                     'agree_b5_percent':100*int(a[0])/max(int(a[1]),1),
                     'state_inputs':sorted(set(sources(int(mode[1:]),nblocks)))}
                rows.append(row);print(row,flush=True)
        # Complete checkpoint already contains all raw frame matrices.
        report[domain]={'rows':rows,'frames':n,'reference_check':checks,'sanity':sanity,
                        'schedules':{mode:sources(int(mode[1:]),nblocks) for mode in modes}}
        write_json(report_path,report)
    rows=[dict(domain=d,**r) for d,v in report.items() for r in v['rows']]
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/'e8c_quality.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def bootstrap():
    # Domain completion/reference checks must have passed before bootstrap is allowed.
    report=json.loads((OUT/'e8c_quality.json').read_text())
    rows=[];rng=np.random.default_rng(0)
    for domain in CONFIG['domains']:
        assert domain in report
        with np.load(OUT/f'{domain}_progress.npz') as z:
            conf=z['confs'];names=z['names'];budgets=z['budgets'];modes=list(z['modes'])
            assert int(z['completed'])==len(names)==len(conf)
        for unit in ['frame','sequence']:
            groups=np.arange(len(names)).astype(str) if unit=='frame' else np.array(['_'.join(n.split('_')[:2]) for n in names])
            unique=sorted(set(groups));g=np.stack([conf[groups==k].sum(0) for k in unique])
            for j,b in enumerate(budgets):
                for mode in CONFIG['variants']:
                    refs=['full_tiles'] if mode=='k1' else ['k1','full_tiles']
                    a=modes.index(mode)
                    for ref in refs:
                        k=modes.index(ref);diff=[]
                        for _ in range(CONFIG['bootstrap_repeats']):
                            idx=rng.integers(0,len(unique),len(unique));c=g[idx,j].sum(0)
                            diff.append(miou(c[a])-miou(c[k]))
                        total=g[:,j].sum(0);ci=np.percentile(diff,[2.5,97.5])
                        rows.append({'domain':domain,'budget':float(b),'unit':unit,'groups':len(unique),
                                     'comparison':mode+' minus '+ref,'delta_miou_pp':float(miou(total[a])-miou(total[k])),
                                     'ci_low':float(ci[0]),'ci_high':float(ci[1]),'repeats':CONFIG['bootstrap_repeats']})
    write_json(OUT/'e8c_bootstrap.json',rows)
    for row in rows:print(row,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['quality','bootstrap']);a=p.parse_args()
    initialize()
    if a.stage=='quality':assert torch.cuda.is_available(),'Select GPU runtime'
    globals()[a.stage]()
