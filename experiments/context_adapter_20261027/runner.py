"""Cityscapes-only preparation and E8; frozen B0/B5. No ACDC tuning."""
import os
os.environ['USE_TF'] = '0'
import argparse, gc, hashlib, json, platform, tarfile, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from common import CACHE, RESULTS, SIZE, TILE, SegSet, load, load_model, confusion, miou

DOMAINS = ['cs_fit', 'cs_test']
SIGNATURE = 'e8-cs-colab-v1'

def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.part')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)

def checkpoint(domain):
    backup = os.environ.get('E8_BACKUP')
    if not backup:
        return
    dest = Path(backup); dest.mkdir(parents=True, exist_ok=True)
    name = dest / f'cache_{domain}.tar'
    tmp = name.with_suffix('.tar.part')
    with tarfile.open(tmp, 'w') as tar:
        tar.add(CACHE / domain, arcname=domain)
    tmp.replace(name)
    print('Кэш сохранён на Drive:', name, flush=True)

def valid_cache(path):
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as z:
        assert set(['gt', 'pred_b0', 'pred_b5', 'logits_b0', 'feat_b0']) <= set(z.files)
        assert all(z[k].shape == SIZE for k in ['gt', 'pred_b0', 'pred_b5'])
        assert z['logits_b0'].shape == (19, 128, 256)
        assert z['feat_b0'].shape == (256, 64, 128)
    return True

@torch.inference_mode()
def cache():
    models = [load_model(n) for n in ['b0', 'b5']]
    captured = {}
    handle = models[0].decode_head.classifier.register_forward_pre_hook(lambda m, x: captured.update(f=x[0]))
    report = {}
    try:
        for domain in DOMAINS:
            data = SegSet(domain)
            assert len(data) == (267 if domain == 'cs_fit' else 233)
            folder = CACHE / domain; folder.mkdir(parents=True, exist_ok=True)
            confs = [np.zeros((19,19), np.int64) for _ in range(2)]
            start = time.monotonic()
            for i in range(len(data)):
                img, gt, name = data[i]
                path = folder / f'{name}.npz'
                if not valid_cache(path):
                    x = img[None].cuda().half()
                    l0 = models[0](pixel_values=x).logits
                    feat = F.avg_pool2d(captured['f'], 2)[0].cpu().numpy()
                    preds = []
                    for logits in [l0, models[1](pixel_values=x).logits]:
                        preds.append(F.interpolate(logits.float(), SIZE, mode='bilinear', align_corners=False).argmax(1)[0].byte().cpu().numpy())
                    tmp = path.with_suffix('.npz.part')
                    with tmp.open('wb') as f:
                        np.savez_compressed(f, gt=gt.numpy(), pred_b0=preds[0], pred_b5=preds[1], logits_b0=l0[0].cpu().numpy(), feat_b0=feat)
                    tmp.replace(path)
                with np.load(path) as z:
                    for j, k in enumerate(['pred_b0', 'pred_b5']):
                        confs[j] += confusion(z[k], z['gt'])
                if (i+1) % 10 == 0 or i+1 == len(data):
                    print(f'E0 {domain}: {i+1}/{len(data)}, {(time.monotonic()-start)/60:.1f} мин', flush=True)
            report[domain] = dict(zip(['b0','b5'], [float(miou(c)) for c in confs]))
            write_json(RESULTS / 'e0_cityscapes.json', report)
            checkpoint(domain)
    finally:
        handle.remove()
    print(report, flush=True)

@torch.inference_mode()
def scores():
    """Only entropy + kNN required by E8. Bank fitted solely on cs_fit."""
    from e2_confidence import uncertainty
    from common import IGNORE
    reference = CACHE / 'e8_knn_bank.npy'
    if not reference.exists():
        rng = np.random.default_rng(0)
        samples = []
        for gt, feat in load('cs_fit', 'gt', 'feat_b0'):
            y = gt[4::8,4::8].ravel()
            valid = np.flatnonzero(y != IGNORE)
            idx = rng.choice(valid, min(500, len(valid)), replace=False)
            samples.append(feat.reshape(256,-1).T[idx].astype(np.float32))
        x = np.concatenate(samples)
        # Separate seeded RNG: exact bank saved, not claimed identical to old E3.
        idx = np.random.default_rng(0).permutation(len(x))[:20000]
        bank = F.normalize(torch.from_numpy(x[idx]), dim=1).numpy()
        np.save(reference, bank)
    bank = torch.from_numpy(np.load(reference)).cuda().float()
    assert bank.shape == (20000,256)
    output = {}
    for domain in DOMAINS:
        arrays = {'entropy':[], 'knn':[]}
        count = len(SegSet(domain))
        for i, (logits, feat) in enumerate(load(domain, 'logits_b0','feat_b0')):
            entropy = uncertainty(torch.from_numpy(logits).float())['entropy']
            entropy = F.avg_pool2d(entropy[None],2)[0].numpy()
            f = F.normalize(torch.from_numpy(feat.astype(np.float32)).cuda().flatten(1).T, dim=1)
            distances = []
            for chunk in f.split(512):
                distances.append((1-(chunk @ bank.T).topk(10,dim=1).values.mean(1)).cpu())
            knn = torch.cat(distances).numpy().reshape(64,128)
            for name, m in [('entropy',entropy),('knn',knn)]:
                # 8x8 feature cells per 64x64 tile.
                arrays[name].append(m.reshape(8,8,16,8).mean((1,3)))
            if (i+1)%10==0 or i+1==count:
                print(f'kNN {domain}: {i+1}/{count}', flush=True)
        for k,v in arrays.items():
            output[f'{domain}__{k}__mean'] = np.stack(v)
    path = CACHE / 'e8_tiles_cityscapes.npz'
    np.savez(path, **output)
    write_json(RESULTS/'knn_config.json', {'fit':'cs_fit only', 'bank':20000, 'k':10, 'sampling_seed':0,
        'bank_sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),
        'note':'Minimal E3 alternative; saved bank differs from historical E3 RNG. Do not compare as exact reproduction.'})
    print('Сохранено:', path, flush=True)

@torch.inference_mode()
def smoke():
    from e8_cityscapes import install, predict, token_mask
    model = load_model('b5'); state = install(model)
    data = SegSet('cs_fit'); img, _, name = data[0]
    all_masks = {st: token_mask(np.ones((8,16),bool),st) for st in state}
    plain = predict(model,state,img,{})
    masked = predict(model,state,img,all_masks)
    assert np.array_equal(plain,masked), 'All-true differs from unmasked model'
    path = CACHE/'cs_fit'/f'{name}.npz'
    assert path.is_file(), 'Сначала выполните E0'
    with np.load(path) as z:
        agreement = float((masked==z['pred_b5']).mean())
    assert agreement > .999, f'Cached agreement={agreement}'
    selected = np.zeros((8,16),bool); selected[3:5,6:10] = True
    routed = predict(model,state,img,{2:token_mask(selected,2)})
    assert routed.shape == SIZE
    write_json(RESULTS/'smoke.json', {'complete':True,'all_true_equal':True,'cached_agreement':agreement,'selected_mask_forward_ok':True})
    print('Smoke OK. Совпадение с кэшем:',agreement, flush=True)


def selftest():
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
    from e8_cityscapes import install
    torch.manual_seed(0)
    cfg=SegformerConfig(num_channels=3,num_encoder_blocks=4,depths=[1,1,2,1],hidden_sizes=[8,16,32,64],num_attention_heads=[1,2,4,8],sr_ratios=[8,4,2,1],decoder_hidden_size=16,num_labels=19,drop_path_rate=0.)
    m=SegformerForSemanticSegmentation(cfg).eval()
    x=torch.randn(1,3,64,128)
    with torch.inference_mode():
        original=m(pixel_values=x).logits
        states=install(m)
        for st,s in states.items():
            s.mask=torch.ones(1,(64//(4*2**st))*(128//(4*2**st)),1,dtype=torch.bool)
        assert torch.equal(original,m(pixel_values=x).logits)
        outputs=[]; handles=[]
        for st,s in states.items():
            s.mask=torch.zeros_like(s.mask)
            for block in m.segformer.encoder.block[st]:
                handles.append(block.register_forward_hook(lambda mod,inp,out:outputs.append(torch.equal(out[0],mod.state.entry))))
        m(pixel_values=x)
        for h in handles: h.remove()
        assert outputs and all(outputs)
    print('Tiny SegFormer: all-true / all-false OK',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['cache','scores','smoke','e8','selftest']);args=p.parse_args()
    if args.stage!='selftest': assert torch.cuda.is_available(), 'Нужен GPU runtime'
    RESULTS.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    import transformers
    write_json(RESULTS/'environment.json', {'signature':SIGNATURE,'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'size':SIZE,'tile':TILE,'domains':DOMAINS,'training':False})
    if args.stage=='e8':
        from e8_cityscapes import main
        main()
    else: globals()[args.stage]()
