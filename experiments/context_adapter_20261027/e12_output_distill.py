"""Bounded adapter-only output distillation. Frozen B5, explicit checkpoint tensor inputs."""
import os
os.environ['USE_TF']='0'
import argparse,hashlib,json,time,csv,shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from common import CACHE,RESULTS,SIZE,TILE,SegSet,load_model,confusion,miou,top_tiles

OUT=Path(os.environ['E12_OUT']);LOCAL=Path(os.environ['E12_LOCAL'])
BASE=Path(os.environ['E12_BASE'])
os.environ['E9K_OUT']=str(OUT);os.environ['E9K_LOCAL']=str(LOCAL)
os.environ['E9K_PREVIOUS']=str(RESULTS.parent/'e9_adapter_v1')
import e9k_base_eval as base
STATES=base.STATES;CONTEXTS=base.CONTEXTS
CONFIG={'schema':'e12-output-distill-v1','seed':0,'epochs':5,'patience':2,'batch':1,
        'lr':1e-4,'weight_decay':1e-4,'temperature':2.,'state_weight':.01,
        'train_mask_fraction':[.05,.30],'val_mask_fractions':[.1,.2],
        'clip_norm':1.,'amp_initial_scale':1024.,'checkpointing':'non-reentrant; explicit context/output/mask tensors',
        'teacher_assisted':True,'b0_frozen':True,'b5_frozen_eval':True,
        'loss':'T^2 KL(full B5 teacher || masked student) on selected full-resolution pixels + .01 balanced state MSE',
        'selection':'fixed validation loss, epoch0 baseline eligible',
        'gate':'training only if true-k4 reserve lower nominal 95% CI >0 at at least one hypothetical E11 budget; exploratory'}

def write_json(p,data):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.part')
    tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)

def atomic_torch(p,data):
    tmp=p.with_suffix('.pt.part');torch.save(data,tmp);tmp.replace(p)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def initialize():
    OUT.mkdir(parents=True,exist_ok=True);LOCAL.mkdir(parents=True,exist_ok=True)
    source=json.loads((BASE/'training_complete.json').read_text())
    assert source['parameters']==227008
    assert sha(BASE/'adapter_best.pt')==source['checkpoint_sha256']
    assert sha(BASE/'normalization.npz')==source['normalization_sha256']
    old=json.loads((BASE/'config.json').read_text())
    assert old['source_sha256']==sha(Path(__file__).parent/'e9k_base_eval.py'), 'Evaluation source differs from trained E9-k4'
    assert old['common_sha256']==sha(Path(__file__).parent/'common.py')
    assert old['tile_features_sha256']==sha(CACHE/'e8_tiles_cityscapes.npz')
    cfg=dict(CONFIG,source_sha256=sha(__file__),base_eval_sha256=sha(Path(__file__).parent/'e9k_base_eval.py'),
        initial_checkpoint_sha256=source['checkpoint_sha256'],normalization_sha256=source['normalization_sha256'],
        split_sha256=sha(BASE/'splits.json'),tile_features_sha256=old['tile_features_sha256'])
    p=OUT/'config.json'
    if p.exists():assert json.loads(p.read_text())==cfg,'Changed settings/source: use new E12_OUT/E12_LOCAL'
    else:write_json(p,cfg)
    for name in ['normalization.npz','splits.json','frame_controls.npz','frame_controls.json','frame_controls.csv']:
        dest=OUT/name
        if not dest.exists():shutil.copy2(BASE/name,dest)
    assert sha(OUT/'normalization.npz')==sha(BASE/'normalization.npz')
    assert sha(OUT/'splits.json')==sha(BASE/'splits.json')
    return cfg


def rank(a):return a.ravel().argsort().argsort().reshape(a.shape)

def paired(a,b,rng,repeats=1000):
    diffs=[];n=len(a)
    for _ in range(repeats):
        idx=rng.integers(0,n,n);diffs.append(miou(a[idx].sum(0))-miou(b[idx].sum(0)))
    ci=np.percentile(diffs,[2.5,97.5])
    return {'delta_miou_pp':float(miou(a.sum(0))-miou(b.sum(0))),'ci_low':float(ci[0]),'ci_high':float(ci[1])}


def gate():
    # All values read from actual reports; the E11 cost remains explicitly hypothetical.
    data=SegSet('cs_test');names=[p.stem for p,_ in data.items];n=len(names)
    assert n==233 and names==[p.stem for p in sorted((CACHE/'cs_test').glob('*.npz'))]
    kroot=RESULTS.parent/'e8c_k_sweep_v1'
    measured=json.loads((kroot/'cost_measured.json').read_text())
    scenarios=json.loads((kroot/'cost_scenarios.json').read_text())['rows']
    c0=measured['models']['b0']['fvcore']['counted_total_gops'];c5=measured['models']['b5']['fvcore']['counted_total_gops']
    with np.load(kroot/'cs_test_progress.npz') as z:
        assert list(z['names'])==names and int(z['completed'])==n
        teacher=z['confs'];teacher_modes=list(z['modes']);budgets=list(z['budgets'])
    with np.load(BASE/'cs_test_progress.npz') as z:
        assert list(z['names'])==names and int(z['completed'])==n
        initial=z['confs'];initial_modes=list(z['modes'])
    frame=np.zeros((n,2,19,19),np.int64)
    for i,name in enumerate(names):
        with np.load(CACHE/'cs_test'/(name+'.npz')) as z:
            frame[i,0]=confusion(z['pred_b0'],z['gt']);frame[i,1]=confusion(z['pred_b5'],z['gt'])
    with np.load(CACHE/'e8_tiles_cityscapes.npz') as z:
        h=z['cs_test__entropy__mean'];k=z['cs_test__knn__mean']
    scores={'raw_entropy':h.mean((1,2)),'rank_mean':(rank(h)+rank(k)).mean((1,2))}
    rows=[];matrices=[];labels=[];decisions=[];rng=np.random.default_rng(0)
    for b in [.1,.2]:
        scenario=next(r for r in scenarios if r['k']==4 and r['budget']==b)
        current=scenario['hybrid_planning_count_halo_aware']
        # User-proposed E11 arithmetic: regionalize all other B5 parts with the same halo fraction.
        hypothetical=(c0+scenario['adapter_core_gmac_scenario']+scenario['stage3_halo_aware_gmac']+
                      scenario['b5_non_stage3_counted_gops']*scenario['halo_query_fraction'])
        j=budgets.index(b);true_k4=teacher[:,j,teacher_modes.index('k4')]
        start=initial[:,j,initial_modes.index('pred_context_pred_output')]
        floor_rows=[]
        for scope,target in [('current_hybrid',current),('hypothetical_e11',hypothetical)]:
            fraction=(target-c0)/c5;assert 0<=fraction<=1
            for method,score in scores.items():
                order=np.argsort(-score,kind='stable')
                for bracket,count in [('floor',int(np.floor(fraction*n))),('ceil',int(np.ceil(fraction*n)))]:
                    mask=np.zeros(n,bool);mask[order[:count]]=True
                    conf=np.where(mask[:,None,None],frame[:,1],frame[:,0])
                    label=f'{scope}_{method}_b{b}_{bracket}'
                    row={'label':label,'budget':b,'scope':scope,'method':method,'bracket':bracket,
                         'target_arithmetic_count':target,'actual_frame_count':count,'actual_frame_fraction':count/n,
                         'actual_counted_gops':c0+count/n*c5,'frame_miou':float(miou(conf.sum(0))),
                         'teacher_k4_miou':float(miou(true_k4.sum(0))),'initial_adapter_miou':float(miou(start.sum(0)))}
                    if bracket=='floor':
                        row.update({'teacher_vs_frame_'+key:value for key,value in paired(true_k4,conf,rng).items()})
                        if scope=='hypothetical_e11':floor_rows.append(row)
                    rows.append(row);matrices.append(conf);labels.append(label)
        # Heuristic research gate, not proof of feasibility: true-k4 must beat BOTH predeclared floor controls.
        passed=all(r['teacher_vs_frame_ci_low']>0 for r in floor_rows)
        decisions.append({'budget':b,'positive_hypothetical_reserve':passed,
                          'checks':[{'method':r['method'],'delta':r['teacher_vs_frame_delta_miou_pp'],
                                     'ci_low':r['teacher_vs_frame_ci_low'],'ci_high':r['teacher_vs_frame_ci_high']} for r in floor_rows]})
    np.savez_compressed(OUT/'gate_frame_controls.npz',confs=np.stack(matrices,axis=1),names=np.array(names),labels=np.array(labels))
    report={'allow_training':any(d['positive_hypothetical_reserve'] for d in decisions),'decisions':decisions,'rows':rows,
            'warning':'Current-hybrid costs are arithmetic approximations; hypothetical E11 additionally scales non-stage3 parts by halo without an implementation or measured quality. Gate uses reused cs_test and nominal exploratory intervals; it is not proof of equal-time advantage.'}
    write_json(OUT/'gate.json',report);base.csv_rows(OUT/'gate.csv',rows)
    print(json.dumps({'allow_training':report['allow_training'],'decisions':decisions,'warning':report['warning']},indent=2),flush=True)


def require_gate():
    r=json.loads((OUT/'gate.json').read_text())
    assert r['allow_training'],'Гейт не дал уверенного запаса над покадровыми контролями. Обучение не запускается; разберите gate.json.'


def teacher_path(name):return LOCAL/'teacher_logits'/(name+'.npz')

def map_path(name):
    existing=LOCAL.parent/BASE.name/'targets/cs_fit'/(name+'.npz')
    return existing if existing.is_file() else LOCAL/'state_targets'/(name+'.npz')


def load_local_targets(names):
    for i,name in enumerate(names):
        p=map_path(name)
        if not p.exists():
            src=BASE/'targets/cs_fit'/(name+'.npz')
            assert src.is_file(),f'Нет предыдущих teacher-целей {src}; восстанови E9-k4 targets'
            p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,p)
        if not teacher_path(name).exists():
            p=teacher_path(name);p.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(OUT/'teacher_logits'/p.name,p)
        if (i+1)%50==0:print('Восстановлены цели:',i+1,'/',len(names),flush=True)

@torch.inference_mode()
def prepare():
    require_gate()
    splits=json.loads((OUT/'splits.json').read_text());names=splits['train']+splits['val']
    data=SegSet('cs_fit');index={p.stem:i for i,(p,_) in enumerate(data.items)}
    assert set(names)==set(index) and len(names)==267
    model=load_model('b5').requires_grad_(False);start=time.monotonic()
    for i,name in enumerate(names):
        p=teacher_path(name);backup=OUT/'teacher_logits'/p.name
        p.parent.mkdir(parents=True,exist_ok=True);backup.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists() and backup.exists():shutil.copy2(backup,p)
        if not p.exists():
            img=data[index[name]][0][None].cuda().half()
            logits=model(pixel_values=img).logits
            pred=F.interpolate(logits.float(),SIZE,mode='bilinear',align_corners=False).argmax(1)[0].byte().cpu().numpy()
            with np.load(CACHE/'cs_fit'/(name+'.npz')) as z:assert (pred==z['pred_b5']).mean()>.999
            tmp=p.with_suffix('.npz.part')
            with tmp.open('wb') as f:np.savez_compressed(f,logits=logits[0].cpu().numpy())
            tmp.replace(p)
        with np.load(p) as z:assert z['logits'].shape==(19,128,256) and np.isfinite(z['logits']).all()
        if not backup.exists():
            tmp=backup.with_suffix('.npz.part');shutil.copy2(p,tmp);tmp.replace(backup)
        if (i+1)%10==0 or i+1==len(names):print(f'Teacher logits: {i+1}/{len(names)}, {(time.monotonic()-start)/60:.1f} мин',flush=True)
    load_local_targets(names)
    write_json(OUT/'prepare_complete.json',{'complete':True,'frames':len(names),'source_states':str(BASE/'targets/cs_fit')})


class InjectBlock(nn.Module):
    def __init__(self,layer,state,i,n,use_checkpoint=True):
        super().__init__();self.layer=layer;self.state=state;self.i=i;self.n=n;self.use_checkpoint=use_checkpoint
    def forward(self,x,*args,**kwargs):
        s=self.state
        if s.mask is None:return self.layer(x,*args,**kwargs)
        # Capture immutable per-call tensor references BEFORE state resets. Backward replay never reads state.
        context=s.contexts[CONTEXTS[self.i*4//self.n]];output=s.output;mask=s.mask
        layer=self.layer;last=self.i==self.n-1
        def core(y,c,o,m):
            out=layer(torch.where(m,y,c),*args,**kwargs)
            value=torch.where(m,out[0],c)
            if last:value=torch.where(m,value,o)
            return (value,)+tuple(out[1:])
        if self.use_checkpoint and torch.is_grad_enabled():
            return checkpoint(core,x,context,output,mask,use_reentrant=False,preserve_rng_state=False)
        return core(x,context,output,mask)

class CheckpointBlock(nn.Module):
    def __init__(self,layer):super().__init__();self.layer=layer
    def forward(self,x,*args,**kwargs):
        layer=self.layer
        def core(y):return layer(y,*args,**kwargs)
        if torch.is_grad_enabled() and x.requires_grad:
            return checkpoint(core,x,use_reentrant=False,preserve_rng_state=False)
        return core(x)

class CheckpointHead(nn.Module):
    def __init__(self,head):super().__init__();self.head=head
    def forward(self,features):
        head=self.head
        def core(*xs):return head(xs)
        if torch.is_grad_enabled() and any(x.requires_grad for x in features):
            return checkpoint(core,*features,use_reentrant=False,preserve_rng_state=False)
        return core(*features)


def install_student(model,use_checkpoint=True):
    state=SimpleNamespace(mask=None,contexts=None,output=None)
    blocks=model.segformer.encoder.block[2]
    assert len(blocks)==40
    for i,block in enumerate(blocks):blocks[i]=InjectBlock(block,state,i,len(blocks),use_checkpoint)
    if use_checkpoint:
        blocks4=model.segformer.encoder.block[3]
        for i,block in enumerate(blocks4):blocks4[i]=CheckpointBlock(block)
        model.decode_head=CheckpointHead(model.decode_head)
    return state


def student_logits(model,state,image,mask,states):
    # Deliberately NO inference_mode/no_grad. Frozen parameters still transmit derivatives to adapter inputs.
    state.mask=mask;state.contexts=states;state.output=states['final']
    try:return model(pixel_values=image).logits
    finally:state.mask=None;state.contexts=None;state.output=None


def decode_states(output,norm,hidden_channels=320):
    states={}
    for j,k in enumerate(STATES):
        a=output[:,j*hidden_channels:(j+1)*hidden_channels]
        a=a*norm[k+'_std'][None,:,None,None]+norm[k+'_mean'][None,:,None,None]
        states[k]=a.flatten(2).transpose(1,2).to(dtype=norm['b5_dtype'])
    return states


def pixel_kl(student,teacher,pixels,size=SIZE,temperature=2.):
    # Teacher detached; score exactly the selected working-resolution pixels, including region boundaries.
    s=F.interpolate(student.float(),size,mode='bilinear',align_corners=False).permute(0,2,3,1)[pixels]
    t=F.interpolate(teacher.detach().float(),size,mode='bilinear',align_corners=False).permute(0,2,3,1)[pixels]
    assert len(s)>0
    return F.kl_div(F.log_softmax(s/temperature,dim=-1),F.softmax(t/temperature,dim=-1),reduction='batchmean')*temperature**2


def fixed_rng(name,epoch,tag):
    value=int(hashlib.sha256(f'{CONFIG["seed"]}:{name}:{epoch}:{tag}'.encode()).hexdigest()[:16],16)
    return np.random.default_rng(value)


def random_mask(name,epoch,fraction=None):
    rng=fixed_rng(name,epoch,'val'+str(fraction) if fraction is not None else 'train')
    b=fraction if fraction is not None else rng.uniform(*CONFIG['train_mask_fraction'])
    mask=np.zeros(128,bool);mask[rng.permutation(128)[:max(1,round(b*128))]]=True
    return mask.reshape(8,16)


def load_sample(name,norm,device,image_index,data):
    image=data[image_index[name]][0][None].to(device=device,dtype=norm['b5_dtype'])
    with np.load(map_path(name)) as z:
        x=torch.from_numpy(z['x'].astype(np.float32))[None].to(device)
        x=(x-norm['x_mean'][None,:,None,None])/norm['x_std'][None,:,None,None]
        target=torch.from_numpy(np.concatenate([(z[k].astype(np.float32)-norm[k+'_mean_np'][:,None,None])/norm[k+'_std_np'][:,None,None] for k in STATES]))[None].to(device)
    with np.load(teacher_path(name)) as z:teacher=torch.from_numpy(z['logits'].astype(np.float32))[None].to(device)
    return image,x,target,teacher


def loss_for(adapter,model,state,sample,norm,tile_mask):
    image,x,target,teacher=sample
    native=torch.as_tensor(tile_mask,device=x.device).repeat_interleave(4,0).repeat_interleave(4,1).reshape(1,2048,1)
    pixels=torch.as_tensor(tile_mask,device=x.device).repeat_interleave(64,0).repeat_interleave(64,1)[None]
    output=adapter(x.float());states=decode_states(output,norm)
    student=student_logits(model,state,image,native,states)
    kl=pixel_kl(student,teacher,pixels,temperature=CONFIG['temperature'])
    mse=base.normalized_loss(output,target)
    return kl+CONFIG['state_weight']*mse,kl,mse


def make_norm(device):
    with np.load(OUT/'normalization.npz') as z:arrays={k:z[k] for k in z.files}
    result={k:torch.as_tensor(v,device=device) for k,v in arrays.items()}
    for k in STATES:result[k+'_mean_np']=arrays[k+'_mean'];result[k+'_std_np']=arrays[k+'_std']
    result['b5_dtype']=torch.float16 if str(device).startswith('cuda') else torch.float32
    return result

@torch.no_grad()
def validate(adapter,model,state,names,norm,device,index,data):
    adapter.eval();sums=np.zeros(3);count=0
    for name in names:
        sample=load_sample(name,norm,device,index,data)
        for b in CONFIG['val_mask_fractions']:
            values=loss_for(adapter,model,state,sample,norm,random_mask(name,0,b))
            sums+=np.array([float(v.item()) for v in values]);count+=1
    return dict(zip(['loss','kl','state_mse'],(sums/count).tolist()))


def base_unchanged(model,versions,buffers):
    assert all(not p.requires_grad and p.grad is None and p._version==versions[n] for n,p in model.named_parameters()),'B5 parameter changed or received gradients'
    assert all(torch.equal(v,buffers[n]) for n,v in model.named_buffers()),'B5 buffer changed'


def train(device="cuda"):
    require_gate()
    if (OUT/'training_complete.json').exists():
        r=json.loads((OUT/'training_complete.json').read_text())
        assert r['checkpoint_sha256']==sha(OUT/'adapter_best.pt')
        print('Дообучение уже завершено, пропускаем',flush=True);return
    splits=json.loads((OUT/'splits.json').read_text());load_local_targets(splits['train']+splits['val'])
    torch.manual_seed(CONFIG['seed']);np.random.seed(CONFIG['seed'])
    adapter=base.Adapter().to(device)
    adapter.load_state_dict(torch.load(BASE/'adapter_best.pt',map_location=device,weights_only=True)['model'])
    model=load_model('b5',device=device).requires_grad_(False).eval();state=install_student(model);model.eval()
    versions={n:p._version for n,p in model.named_parameters()};buffers={n:b.detach().clone() for n,b in model.named_buffers()}
    assert not set(map(id,adapter.parameters()))&set(map(id,model.parameters()))
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',init_scale=CONFIG['amp_initial_scale'],enabled=str(device).startswith('cuda'))
    norm=make_norm(device);data=SegSet('cs_fit');index={p.stem:i for i,(p,_) in enumerate(data.items)}
    last=OUT/'adapter_last.pt';start_epoch=0;bad=0;history=[];best_epoch=0;best=float('inf');best_weights=None
    if last.exists():
        r=torch.load(last,map_location=device,weights_only=True)
        adapter.load_state_dict(r['model']);optimizer.load_state_dict(r['optimizer']);scaler.load_state_dict(r['scaler'])
        start_epoch=r['epoch'];bad=r['bad'];history=r['history'];best_epoch=r['best_epoch'];best=r['best']
        best_weights={k:v.detach().cpu().clone() for k,v in r['best_model'].items()}
        atomic_torch(OUT/'adapter_best.pt',{'model':best_weights,'epoch':best_epoch,'val_loss':best})
        print('Продолжаем после эпохи',start_epoch,flush=True)
    if not last.exists():
        baseline=validate(adapter,model,state,splits['val'],norm,device,index,data)
        best=baseline['loss'];history=[{'epoch':0,'val':baseline,'source':'initial E9-k4; no update'}]
        best_weights={k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()}
        atomic_torch(OUT/'adapter_best.pt',{'model':best_weights,'epoch':0,'val_loss':best})
        write_json(OUT/'initial_validation.json',baseline);write_json(OUT/'training_history.json',history)
        print('Исходный адаптер, эпоха 0:',baseline,flush=True)
    # One backward without optimizer.step: confirm gradient path and memory before starting epochs.
    if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True);adapter.train()
    sample=load_sample(splits['train'][0],norm,device,index,data)
    try:
        loss,_,_=loss_for(adapter,model,state,sample,norm,random_mask(splits['train'][0],0,.1))
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters()), 'Nonfinite adapter gradients in probe'
        grads=adapter.net[2].weight.grad
        norms=[float(grads[j*320:(j+1)*320].float().norm().item()) for j in range(5)]
        assert all(np.isfinite(v) and v>0 for v in norms),'Missing/nonfinite adapter head gradients'
        base_unchanged(model,versions,buffers)
        write_json(OUT/'backward_probe.json',{'passed':True,'head_gradient_norms':norms,
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20 if str(device).startswith('cuda') else None,'b5_unchanged':True})
        print('Backward probe OK:',norms,flush=True)
        scaler.update();optimizer.zero_grad(set_to_none=True)
        del loss,sample
    except torch.cuda.OutOfMemoryError:
        write_json(OUT/'backward_probe.json',{'passed':False,'reason':'CUDA OOM; no optimizer update performed'})
        raise
    for epoch in range(start_epoch,CONFIG['epochs']):
        if bad>=CONFIG['patience']:break
        adapter.train();order=np.random.default_rng(CONFIG['seed']+epoch).permutation(len(splits['train']))
        sums=np.zeros(3);count=0;skipped=0;start=time.monotonic()
        for step,i in enumerate(order):
            name=splits['train'][int(i)];sample=load_sample(name,norm,device,index,data)
            optimizer.zero_grad(set_to_none=True)
            loss,kl,mse=loss_for(adapter,model,state,sample,norm,random_mask(name,epoch+1))
            assert torch.isfinite(loss),'Nonfinite loss'
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(adapter.parameters(),CONFIG['clip_norm'])
            scale_before=scaler.get_scale();scaler.step(optimizer);scaler.update()
            skipped+=int(scaler.get_scale()<scale_before)
            sums+=np.array([loss.item(),kl.item(),mse.item()]);count+=1
            if (step+1)%25==0:print(f'Эпоха {epoch+1}: {step+1}/{len(order)}, {(time.monotonic()-start)/60:.1f} мин',flush=True)
        validation=validate(adapter,model,state,splits['val'],norm,device,index,data)
        assert np.isfinite(validation['loss']), 'Nonfinite validation loss'
        row={'epoch':epoch+1,'train':dict(zip(['loss','kl','state_mse'],(sums/count).tolist())),'val':validation,
             'amp_skipped_steps':skipped,'amp_scale':scaler.get_scale()}
        history.append(row)
        if validation['loss']<best:
            best=validation['loss'];best_epoch=epoch+1;bad=0
            best_weights={k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()}
            atomic_torch(OUT/'adapter_best.pt',{'model':best_weights,'epoch':best_epoch,'val_loss':best})
        else:bad+=1
        base_unchanged(model,versions,buffers)
        atomic_torch(last,{'model':adapter.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                          'epoch':epoch+1,'best':best,'best_epoch':best_epoch,'bad':bad,'history':history,'best_model':best_weights})
        write_json(OUT/'training_history.json',history);print(row,flush=True)
    base_unchanged(model,versions,buffers)
    write_json(OUT/'training_complete.json',{'complete':True,'best_epoch':best_epoch,'val_loss':best,
        'parameters':227008,'checkpoint_sha256':sha(OUT/'adapter_best.pt'),
        'normalization_sha256':sha(OUT/'normalization.npz'),'initial_checkpoint_sha256':sha(BASE/'adapter_best.pt'),
        'b5_parameters_and_buffers_unchanged':True,'train_frames':len(splits['train']),'val_frames':len(splits['val'])})


def evaluate():
    require_gate();base.evaluate()
    rows=json.loads((OUT/'routing.json').read_text())['cs_test']
    old=json.loads((BASE/'routing.json').read_text())['cs_test']
    for row in rows:
        initial=next(r['miou'] for r in old if r['budget']==row['budget'] and r['variant']=='pred_context_pred_output')
        row['initial_k4_pred_all_miou']=initial;row['delta_vs_initial_k4_pp']=row['miou']-initial
    write_json(OUT/'routing.json',dict(json.loads((OUT/'routing.json').read_text()),cs_test=rows))
    base.csv_rows(OUT/'routing.csv',rows)


def bootstrap():
    with np.load(OUT/'cs_test_progress.npz') as z:
        ours=z['confs'];names=z['names'];budgets=list(z['budgets']);modes=list(z['modes']);assert int(z['completed'])==len(names)
    with np.load(BASE/'cs_test_progress.npz') as z:
        initial=z['confs'];assert np.array_equal(z['names'],names);initial_modes=list(z['modes'])
    with np.load(OUT/'gate_frame_controls.npz') as z:
        gate_confs=z['confs'];labels=list(z['labels']);assert np.array_equal(z['names'],names)
    meta=json.loads((OUT/'gate.json').read_text())['rows'];rng=np.random.default_rng(0);rows=[]
    for unit in ['frame','sequence']:
        groups=np.arange(len(names)).astype(str) if unit=='frame' else np.array(['_'.join(n.split('_')[:2]) for n in names])
        unique=sorted(set(groups))
        def group(c):return np.stack([c[groups==g].sum(0) for g in unique])
        for j,b in enumerate(budgets):
            a=group(ours[:,j,modes.index('pred_context_pred_output')])
            references={'initial_k4':initial[:,j,initial_modes.index('pred_context_pred_output')],
                        'true_k4':ours[:,j,modes.index('true_context_true_output')],
                        'full_tiles':ours[:,j,modes.index('full_tiles')]}
            for item in meta:
                if item['budget']==b and item['bracket']=='floor':references[item['label']]=gate_confs[:,labels.index(item['label'])]
            for label,c in references.items():
                result=paired(a,group(c),rng)
                rows.append({'budget':b,'unit':unit,'groups':len(unique),'comparison':'finetuned_pred_all minus '+label,**result,
                    'scope':'quality contrast; cost-matched labels use speculative arithmetic, not measured time'})
    write_json(OUT/'bootstrap.json',rows);base.csv_rows(OUT/'bootstrap.csv',rows)
    for row in rows:print(row,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['gate','prepare','train','evaluate','bootstrap']);args=p.parse_args()
    initialize()
    if args.stage not in ['gate','bootstrap']:assert torch.cuda.is_available(),'Select GPU runtime'
    import platform,transformers
    write_json(OUT/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,
        'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    globals()[args.stage]()
