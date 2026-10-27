"""Measured dense cost and explicit arithmetic scenarios; no claimed sparse latency."""
import os
os.environ['USE_TF']='0'
import csv,gc,hashlib,json,time,platform
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from common import CACHE,RESULTS,SIZE,TILE,SegSet,load_model,top_tiles
OUT=Path(os.environ['E8C_OUT'])
COST_CONFIG={'schema':'dense-cost-and-k-scenarios-v2','image_shape':[1,3,512,1024],
             'dtype':'float16 segmenters, float32 optional adapter core','images':5,'repeats_per_image':5,
             'warmup':3,'component_repeats':5,'kv_repeats':10,
             'flop_convention':'fvcore: one multiply-add counts as one operation',
             'no_sparse_runtime_implemented':True}

def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    p=path.with_suffix(path.suffix+'.part');p.write_text(json.dumps(data,indent=2));p.replace(path)

def save_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

class Logits(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,x):return self.model(pixel_values=x).logits

@torch.inference_mode()
def measured(fn,inputs,repeats=5,warmup=3):
    for _ in range(warmup):fn(inputs[0])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    gpu=[];wall=[]
    for x in inputs:
        for _ in range(repeats):
            torch.cuda.synchronize()
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            t=time.perf_counter();a.record();result=fn(x);b.record();torch.cuda.synchronize()
            wall.append(1000*(time.perf_counter()-t));gpu.append(a.elapsed_time(b))
            del result
    return {'samples':len(gpu),'gpu_mean_ms':float(np.mean(gpu)),
            'gpu_p95_ms':float(np.percentile(gpu,95)),
            'wall_mean_ms':float(np.mean(wall)),'wall_p95_ms':float(np.percentile(wall,95)),
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20}


def components(model):
    enc=model.segformer.encoder
    groups=[]
    for st in range(4):
        prefix=f'stage{st+1}'
        groups.append((prefix+'_patch',enc.patch_embeddings[st],enc.patch_embeddings[st]))
        groups.append((prefix+'_blocks',enc.block[st][0],enc.block[st][-1]))
        groups.append((prefix+'_norm',enc.layer_norm[st],enc.layer_norm[st]))
    groups.append(('decode_head',model.decode_head,model.decode_head))
    return groups

@torch.inference_mode()
def timed_components(model,x):
    # Non-overlapping groups; separate instrumented passes, not the latency benchmark above.
    pending={};pairs=defaultdict(list);handles=[]
    def begin(name):
        def hook(module,args):
            event=torch.cuda.Event(enable_timing=True);event.record();pending[name]=event
        return hook
    def end(name):
        def hook(module,args,out):
            event=torch.cuda.Event(enable_timing=True);event.record();pairs[name].append((pending.pop(name),event))
        return hook
    for name,first,last in components(model):
        handles.append(first.register_forward_pre_hook(begin(name)))
        handles.append(last.register_forward_hook(end(name)))
    try:
        for _ in range(COST_CONFIG['component_repeats']):model(pixel_values=x)
        torch.cuda.synchronize()
        return [{'component':name,'gpu_mean_ms':float(np.mean([a.elapsed_time(b) for a,b in events])),
                 'instrumented':True} for name,events in pairs.items()]
    finally:
        for h in handles:h.remove()

@torch.inference_mode()
def flop_counts(model,x):
    from fvcore.nn import FlopCountAnalysis
    wrapped=Logits(model).eval()
    f=FlopCountAnalysis(wrapped,(x,)).unsupported_ops_warnings(False).uncalled_modules_warnings(False).tracer_warnings('none')
    total=float(f.total());by=f.by_module()
    rows=[]
    for st in range(4):
        for label,path in [('patch',f'model.segformer.encoder.patch_embeddings.{st}'),
                           ('blocks',f'model.segformer.encoder.block.{st}'),
                           ('norm',f'model.segformer.encoder.layer_norm.{st}')]:
            value=float(by.get(path,0))
            rows.append({'component':f'stage{st+1}_{label}','counted_gops_fvcore':value/1e9,
                         'share_of_counted_total':value/max(total,1)})
    value=float(by.get('model.decode_head',0))
    rows.append({'component':'decode_head','counted_gops_fvcore':value/1e9,'share_of_counted_total':value/max(total,1)})
    return {'counted_total_gops':total/1e9,'components':rows,
            'by_operator':{k:float(v) for k,v in f.by_operator().items()},
            'unsupported_ops':dict(f.unsupported_ops()),
            'uncalled_modules':sorted(f.uncalled_modules()),
            'note':'Counted operations only; explicit unsupported ops are not silently treated as measured FLOPs.'}


def dominant_macs(block,height,width):
    a=block.attention.self
    q=a.query;k=a.key;v=a.value;o=block.attention.output.dense
    assert q.in_features==k.in_features==v.in_features
    assert q.out_features==k.out_features==v.out_features
    assert q.in_features==q.out_features,'Unpruned equal-width attention expected'
    n=height*width;d=q.in_features
    if a.sr_ratio>1:
        sr=a.sr
        kh,kw=sr.kernel_size;sh,sw=sr.stride;ph,pw=sr.padding;dh,dw=sr.dilation
        hk=(height+2*ph-dh*(kh-1)-1)//sh+1
        wk=(width+2*pw-dw*(kw-1)-1)//sw+1
        nk=hk*wk
        sr_mac=nk*sr.out_channels*(sr.in_channels//sr.groups)*kh*kw
    else:nk=n;sr_mac=0
    f=block.mlp;conv=f.dwconv.dwconv
    return {'sr':int(sr_mac),'kv':int(nk*(k.in_features*k.out_features+v.in_features*v.out_features)),
            'q':int(n*q.in_features*q.out_features),
            'attention_products':int(2*n*nk*q.out_features),
            'attention_output':int(n*o.in_features*o.out_features),
            'ffn_dense1':int(n*f.dense1.in_features*f.dense1.out_features),
            'ffn_dwconv':int(n*conv.out_channels*(conv.in_channels//conv.groups)*conv.kernel_size[0]*conv.kernel_size[1]),
            'ffn_dense2':int(n*f.dense2.in_features*f.dense2.out_features)}


def halo_fractions(tile_masks):
    native=torch.from_numpy(np.asarray(tile_masks)).float().repeat_interleave(TILE//16,1).repeat_interleave(TILE//16,2)
    expanded=F.max_pool2d(native[:,None],kernel_size=3,stride=1,padding=1)[:,0]
    return float(native.mean()),float(expanded.mean())


def sparse_macs(parts,rho,halo):
    fixed=parts['sr']+parts['kv']
    query_halo=parts['q']+parts['attention_products']+parts['attention_output']+parts['ffn_dense1']
    selected=parts['ffn_dwconv']+parts['ffn_dense2']
    return {'fixed_sr_kv':fixed,'full_dominant_macs':sum(parts.values()),
            'optimistic_no_halo_macs':fixed+rho*(query_halo+selected),
            'halo_aware_macs':fixed+halo*query_halo+rho*selected}

@torch.inference_mode()
def collect_stage3(model,x):
    layers=model.segformer.encoder.block[2]
    inputs=[None]*len(layers);shape={};handles=[]
    def pre(i):
        def hook(module,args):inputs[i]=args[0].detach();shape['height']=int(args[1]);shape['width']=int(args[2])
        return hook
    for i,block in enumerate(layers):handles.append(block.register_forward_pre_hook(pre(i)))
    try:model(pixel_values=x)
    finally:
        for h in handles:h.remove()
    return inputs,shape


def kv_path(block,x,height,width):
    # Precisely the full-context normalization/SR/K/V path, no queries or FFN.
    x=block.layer_norm_1(x);a=block.attention.self
    if a.sr_ratio>1:
        x=a.sr(x.transpose(1,2).reshape(x.shape[0],x.shape[2],height,width))
        x=a.layer_norm(x.flatten(2).transpose(1,2))
    return a.key(x),a.value(x)


@torch.inference_mode()
def optional_adapter():
    path=RESULTS.parent/'e9_adapter_v1/adapter_best.pt'
    if not path.exists():return {'available':False,'note':'E9 best checkpoint not found; no adapter latency measured'}
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    state=checkpoint['model'];in_channels=state['net.0.weight'].shape[1]
    hidden=state['net.0.weight'].shape[0];out_channels=state['net.2.weight'].shape[0]
    class Adapter(nn.Module):
        def __init__(self):
            super().__init__();self.net=nn.Sequential(nn.Conv2d(in_channels,hidden,1),nn.GELU(),nn.Conv2d(hidden,out_channels,1))
        def forward(self,x):return self.net(x)
    model=Adapter().cuda().float().eval().requires_grad_(False);model.load_state_dict(state)
    x=torch.zeros((1,in_channels,SIZE[0]//16,SIZE[1]//16),device='cuda',dtype=torch.float32)
    latency=measured(model,[x],repeats=20,warmup=3)
    n=x.shape[-2]*x.shape[-1]
    count=n*(in_channels*hidden+hidden*out_channels)
    result={'available':True,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'parameters':sum(p.numel() for p in model.parameters()),'core_gmac':count/1e9,'latency':latency,
            'note':'MLP core only, fp32 as E9. Input on GPU; normalization/denormalization, B0 and mask handling excluded.'}
    del model,x;gc.collect();torch.cuda.empty_cache()
    return result

@torch.inference_mode()
def measure_dense_and_initial_scenarios():
    assert torch.cuda.is_available(),'Select GPU runtime'
    OUT.mkdir(parents=True,exist_ok=True)
    config=dict(COST_CONFIG,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    config_path=OUT/'cost_config.json'
    if config_path.exists():assert json.loads(config_path.read_text())==config,'Changed code/config: use new E8C_OUT'
    else:write_json(config_path,config)
    import transformers
    hardware={'gpu':torch.cuda.get_device_name(0),'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
              'torch':torch.__version__,'cuda':torch.version.cuda,'transformers':transformers.__version__,
              'python':platform.python_version(),'tf32_matmul':torch.backends.cuda.matmul.allow_tf32}
    data=SegSet('cs_test');indices=np.linspace(0,len(data)-1,min(COST_CONFIG['images'],len(data)),dtype=int)
    imgs=[data[int(i)][0][None].cuda().half() for i in indices]
    report={'hardware':hardware,'config':config,'sample_names':[data.items[int(i)][0].stem for i in indices],
            'models':{},'note':'P95 is over this small microbenchmark, not dataset latency P95.'}
    latency_rows=[];component_rows=[]
    for name in ['b0','b5']:
        print('Cost:',name,flush=True)
        model=load_model(name).requires_grad_(False)
        raw=measured(lambda x:model(pixel_values=x).logits,imgs,COST_CONFIG['repeats_per_image'],COST_CONFIG['warmup'])
        def mask_forward(x):
            logits=model(pixel_values=x).logits
            return F.interpolate(logits.float(),SIZE,mode='bilinear',align_corners=False).argmax(1)[0].byte().cpu().numpy()
        masks=measured(mask_forward,imgs,COST_CONFIG['repeats_per_image'],COST_CONFIG['warmup'])
        parts=timed_components(model,imgs[0])
        flop=flop_counts(model,imgs[0])
        report['models'][name]={'parameters':sum(p.numel() for p in model.parameters()),
                                'forward_gpu_resident':raw,'to_cpu_mask':masks,'timed_components':parts,'fvcore':flop}
        latency_rows.extend([dict(model=name,path='forward_gpu_resident',**raw),dict(model=name,path='to_cpu_mask',**masks)])
        for row in flop['components']:
            timer=next(t for t in parts if t['component']==row['component'])
            component_rows.append(dict(model=name,**row,instrumented_gpu_mean_ms=timer['gpu_mean_ms']))
        if name=='b5':
            inputs,shape=collect_stage3(model,imgs[0]);height=shape['height'];width=shape['width']
            layers=model.segformer.encoder.block[2]
            dparts=[dominant_macs(block,height,width) for block in layers]
            summed={k:sum(p[k] for p in dparts) for k in dparts[0]}
            def full_kv(_):
                # Calls each layer's distinct weights on its actual dense-pass input.
                for block,x in zip(layers,inputs):out=kv_path(block,x,height,width)
                return out
            kv=measured(full_kv,[None],repeats=COST_CONFIG['kv_repeats'],warmup=3)
            report['stage3']={'height':height,'width':width,'blocks':len(layers),'dominant_macs':summed,
                'full_sr_kv_fraction_of_dominant_macs':(summed['sr']+summed['kv'])/sum(summed.values()),
                'full_context_kv_path_all_blocks':kv,
                'kv_note':'Standalone microbenchmark including LN1/SR/LN/K/V; not additive to measured forward latency. Masks/queries/FFN excluded.'}
            del inputs
        write_json(OUT/'cost_measured.json',report)
        del model;gc.collect();torch.cuda.empty_cache()
    report['adapter']=optional_adapter()
    write_json(OUT/'cost_measured.json',report)
    save_csv(OUT/'cost_latency.csv',latency_rows)
    save_csv(OUT/'cost_components.csv',component_rows)
    # Arithmetic scenarios: full B5 early/later stages remain; sparse stage3 not implemented.
    with np.load(CACHE/'e8_tiles_cityscapes.npz') as z:
        def rank(a):return a.ravel().argsort().argsort().reshape(a.shape)
        score=rank(z['cs_test__entropy__mean'])+rank(z['cs_test__knn__mean'])
    b0_count=report['models']['b0']['fvcore']['counted_total_gops']
    b5_count=report['models']['b5']['fvcore']['counted_total_gops']
    s3_count=next(r['counted_gops_fvcore'] for r in report['models']['b5']['fvcore']['components'] if r['component']=='stage3_blocks')
    fixed_other=b5_count-s3_count
    adapter_gmac=report['adapter'].get('core_gmac')
    scenarios=[]
    for budget in [.1,.2]:
        tile_mask=top_tiles(score,budget)
        selected,halo=halo_fractions(tile_mask)
        active=float(tile_mask.reshape(len(tile_mask),-1).any(1).mean())
        calc=sparse_macs(report['stage3']['dominant_macs'],selected,halo)
        row={'budget':budget,'selected_native_fraction':selected,'halo_query_fraction':halo,
             'active_frame_fraction':active,
             'stage3_full_dominant_gmac':calc['full_dominant_macs']/1e9,
             'stage3_fixed_sr_kv_gmac':calc['fixed_sr_kv']/1e9,
             'stage3_optimistic_no_halo_gmac':calc['optimistic_no_halo_macs']/1e9,
             'stage3_halo_aware_gmac':calc['halo_aware_macs']/1e9,
             'b5_non_stage3_counted_gops':fixed_other,'b0_full_counted_gops':b0_count,
             'adapter_core_gmac':adapter_gmac,
             'assumptions':'Planning arithmetic only: full B0, B5 outside stage3 unchanged, full SR/KV every block; sparse query/FFN; 1-token FFN halo. No sparse latency implementation.'}
        if adapter_gmac is not None:
            row['hybrid_planning_count_halo_aware']=b0_count+adapter_gmac+fixed_other+calc['halo_aware_macs']/1e9
            row['hybrid_planning_ratio_to_counted_b5']=row['hybrid_planning_count_halo_aware']/b5_count
            row['hybrid_planning_count_skip_empty_frames']=b0_count+active*(adapter_gmac+fixed_other)+(calc['halo_aware_macs']-(1-active)*calc['fixed_sr_kv'])/1e9
            row['hybrid_planning_ratio_skip_empty_frames']=row['hybrid_planning_count_skip_empty_frames']/b5_count
        scenarios.append(row)
    write_json(OUT/'cost_scenarios.json',{'rows':scenarios,
        'warning':'Hybrid approximation mixes fvcore supported-op counts for dense parts with dominant MACs for sparse stage3. It omits stage3 layernorm/softmax/GELU/residual/masking/indexing/assembly, adapter normalization, router, transfers and unhandled operations. It is not measured FLOPs or a latency/speedup prediction.'})
    save_csv(OUT/'cost_scenarios.csv',scenarios)
    frame_curve=frame_baseline(data,score,b0_count,b5_count)
    save_csv(OUT/'cached_frame_baseline.csv',frame_curve)
    print(json.dumps(report['stage3'],indent=2),flush=True)
    print('Measured latency:',latency_rows,flush=True)
    print('Arithmetic scenarios:',scenarios,flush=True)

def frame_baseline(data,tile_score,b0_count,b5_count):
    # Existing predictions only. Quality/control-budget curve, no live cascade timing.
    from common import confusion,miou
    with np.load(CACHE/'e8_tiles_cityscapes.npz') as z:
        methods={'raw_entropy':z['cs_test__entropy__mean'].mean((1,2)),
                 'rank_mean':tile_score.mean((1,2))}
    records=[]
    for image,_ in data.items:
        with np.load(CACHE/'cs_test'/(image.stem+'.npz')) as z:
            records.append((confusion(z['pred_b0'],z['gt']),confusion(z['pred_b5'],z['gt'])))
    records=np.asarray(records);rows=[];n=len(records)
    for method,score in methods.items():
        order=np.argsort(-score,kind='stable')
        for budget in [0.,.05,.1,.2,.3,.5,.75,1.]:
            m=np.zeros(n,bool);m[order[:round(budget*n)]]=True
            c=np.where(m[:,None,None],records[:,1],records[:,0]).sum(0)
            fraction=float(m.mean())
            rows.append({'method':method,'requested_frame_fraction':budget,'actual_frame_fraction':fraction,
                         'cached_miou':float(miou(c)),
                         'counted_gops_b0_plus_selected_full_b5':b0_count+fraction*b5_count,
                         'note':'Transductive domain-wide frame ranking, cached predictions; not measured cascade latency.'})
    return rows




OLD_DENSE_SOURCE_SHA256 = 'faa4d2ef339df630853dd1327627d170aadefaf88c0e74276168d2a975c0b071'

def reuse_measured():
    import shutil
    origin=Path(os.environ.get('E8C_MEASURED_REUSE',str(RESULTS.parent/'e8c_cost_v1')))
    required=['cost_measured.json','cost_latency.csv','cost_components.csv','cost.py']
    if not all((origin/name).is_file() for name in required):return False
    report=json.loads((origin/'cost_measured.json').read_text())
    source_hash=hashlib.sha256((origin/'cost.py').read_bytes()).hexdigest()
    if source_hash!=OLD_DENSE_SOURCE_SHA256 or report['config'].get('source_sha256')!=source_hash:
        print('Не используем старый профиль: не совпала версия кода.',flush=True);return False
    hw=report['hardware']
    if hw['gpu']!=torch.cuda.get_device_name(0) or hw['torch']!=torch.__version__ or hw['cuda']!=torch.version.cuda:
        print('Не используем старый профиль: другое GPU/окружение.',flush=True);return False
    if hw.get('tf32_matmul')!=torch.backends.cuda.matmul.allow_tf32:return False
    if report['config'].get('image_shape')!=COST_CONFIG['image_shape']:return False
    if not all(name in report['models'] for name in ['b0','b5']):return False
    if 'stage3' not in report:return False
    report['reuse']={'source':str(origin),'historical_measurements':True,
                     'not_measured_again':True,'original_code_sha256':source_hash}
    write_json(OUT/'cost_measured.json',report)
    for name in ['cost_latency.csv','cost_components.csv']:
        shutil.copy2(origin/name,OUT/name)
    write_json(OUT/'cost_config.json',dict(COST_CONFIG,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                         measured_reused_from=str(origin)))
    print('Используем уже измеренный профиль:',origin,flush=True)
    return True


def k_scenarios(report):
    with np.load(CACHE/'e8_tiles_cityscapes.npz') as z:
        def rank(a):return a.ravel().argsort().argsort().reshape(a.shape)
        score=rank(z['cs_test__entropy__mean'])+rank(z['cs_test__knn__mean'])
    b0_count=report['models']['b0']['fvcore']['counted_total_gops']
    b5_count=report['models']['b5']['fvcore']['counted_total_gops']
    s3_count=next(r['counted_gops_fvcore'] for r in report['models']['b5']['fvcore']['components'] if r['component']=='stage3_blocks')
    fixed_other=b5_count-s3_count
    n=report['stage3']['height']*report['stage3']['width']
    scenarios=[]
    for k in [1,2,4,8]:
        c_out=320*(k+1)  # k distinct contexts PLUS final restoration, not k outputs total.
        adapter_gmac=n*(160*128+128*c_out)/1e9
        parameters=160*128+128+128*c_out+c_out
        for budget in [.1,.2]:
            tile_mask=top_tiles(score,budget)
            rho,halo=halo_fractions(tile_mask)
            active=float(tile_mask.reshape(len(tile_mask),-1).any(1).mean())
            calc=sparse_macs(report['stage3']['dominant_macs'],rho,halo)
            total=b0_count+adapter_gmac+fixed_other+calc['halo_aware_macs']/1e9
            skipped=b0_count+active*(adapter_gmac+fixed_other)+(calc['halo_aware_macs']-(1-active)*calc['fixed_sr_kv'])/1e9
            scenarios.append({'k':k,'adapter_output_channels':c_out,'adapter_parameters_scenario':parameters,
              'adapter_core_gmac_scenario':adapter_gmac,'budget':budget,
              'selected_native_fraction':rho,'halo_query_fraction':halo,'active_frame_fraction':active,
              'stage3_fixed_sr_kv_gmac':calc['fixed_sr_kv']/1e9,
              'stage3_halo_aware_gmac':calc['halo_aware_macs']/1e9,
              'b5_non_stage3_counted_gops':fixed_other,'b0_full_counted_gops':b0_count,
              'hybrid_planning_count_halo_aware':total,'hybrid_planning_ratio_to_counted_b5':total/b5_count,
              'hybrid_planning_count_skip_empty_frames':skipped,
              'hybrid_planning_ratio_skip_empty_frames':skipped/b5_count,
              'status':'unimplemented arithmetic scenario; not measured cascade performance'})
    write_json(OUT/'cost_scenarios.json',{'rows':scenarios,
        'warning':'Dense costs and sparse arithmetic are separate. All early/later B5 parts remain full-frame. Halo is accounted for in stage3. k>1 adapters are hypothetical; only k=1 adapter has trained weights. No sparse latency is measured; do not attach E9 quality to a hypothetical regional-early-stage cost.'})
    save_csv(OUT/'cost_scenarios.csv',scenarios)
    frame_rows=frame_baseline(SegSet('cs_test'),score,b0_count,b5_count)
    save_csv(OUT/'cached_frame_baseline.csv',frame_rows)
    for row in scenarios:print(row,flush=True)


def main():
    assert torch.cuda.is_available(),'Select GPU runtime'
    OUT.mkdir(parents=True,exist_ok=True)
    if not reuse_measured():measure_dense_and_initial_scenarios()
    report=json.loads((OUT/'cost_measured.json').read_text())
    k_scenarios(report)
if __name__=='__main__':main()
