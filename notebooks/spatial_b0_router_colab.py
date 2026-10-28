# %% [markdown]
# # Spatial router: только выход B0 → выбрать B5-плитки
#
# Основано на идее *Sparse Refinement for Efficient High-Resolution Semantic
# Segmentation* (ECCV 2024): дешёвое плотное предсказание выбирает небольшое
# число мест для дорогого уточнения. Это не очередной fusion B0/B5: тяжёлая B5
# уже посчитана и не запускается. Обучается только компактный router на P0.
#
# **Вход router для одной плитки** — пространственная карта B0 размером 16×16:
# 19 вероятностей классов, entropy, разница top-1/top-2, карта границ B0 и x/y;
# плюс глобальный контекст кадра: средние 19 вероятностей и средняя entropy.
# Router предсказывает ожидаемый выигрыш B5 над B0 в плитке. P1 не используется
# до OOF-защиты на P0.

# %%
from pathlib import Path
from google.colab import drive

MOUNT = Path('/content/gdrive')
if not (MOUNT / 'MyDrive').exists():
    drive.mount(str(MOUNT))
%pip -q install --upgrade-strategy only-if-needed transformers pyarrow

# %%
# Самодостаточная подготовка: если на Drive нет результатов, они строятся один раз и сохраняются.
import os, shutil, subprocess, sys, zipfile, numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
import gdown

assert torch.cuda.is_available(), 'В Colab включите T4 GPU.'
REPO = Path('/content/error-geometry-routing')
DRIVE_DIR = MOUNT / 'MyDrive' / 'error-geometry-routing'
OUT = DRIVE_DIR / 'spatial_b0_router_20261028'
OUT.mkdir(parents=True, exist_ok=True)

if not REPO.exists():
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'experiments/frozen-router-results-20261025',
                    'https://github.com/Ottobiss/error-geometry-routing.git', str(REPO)], check=True)
os.chdir(REPO)

BASE = OUT / 'baseline_results'
required = [BASE/'cityscapes/logits_b0.npy', BASE/'cityscapes/logits_b0_ids.txt',
            BASE/'cityscapes/masks', BASE/'cityscapes/cache_p0.npz', BASE/'cityscapes/cache_p1.npz']
old = sorted(DRIVE_DIR.glob('posthoc_router_*/results'))

def make_cityscapes_available():
    """Скачивает только официальные val images и gtFine, затем приводит пути к формату проекта."""
    download = Path('/content/cityscapes-archives'); download.mkdir(exist_ok=True)
    archives = {'val.zip':'1aErjLEt25oJEX48W-zxHwITEvNAeKxGJ',
                'gtFine_trainvaltest.zip':'1AxGkcacU_SZXPSjxUtzKgT3XSZM032bk'}
    for name, fid in archives.items():
        archive = download/name
        if not archive.exists(): gdown.download(id=fid, output=str(archive), quiet=False)
    data = REPO/'data/cityscapes'; data.mkdir(parents=True, exist_ok=True)
    for name in archives:
        with zipfile.ZipFile(download/name) as z:
            for item in z.infolist():
                if not (data/item.filename).exists(): z.extract(item, data)
    image_root = data/'leftImg8bit/val'
    for src in data.rglob('*_leftImg8bit.png'):
        if image_root in src.parents: continue
        dst=image_root/src.parent.name/src.name; dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists(): shutil.copy2(src,dst)
    images=list(image_root.rglob('*_leftImg8bit.png')); assert len(images)==500, len(images)
    wanted={p.name.removesuffix('_leftImg8bit.png') for p in images}; label_root=data/'gtFine/val'; label_root.mkdir(parents=True,exist_ok=True)
    for dst in label_root.rglob('*_gtFine_labelIds.png'):
        if dst.name.removesuffix('_gtFine_labelIds.png') not in wanted: dst.unlink()
    for src in data.rglob('*_gtFine_labelIds.png'):
        if src.name.removesuffix('_gtFine_labelIds.png') not in wanted: continue
        dst=label_root/src.parent.name/src.name; dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists(): shutil.copy2(src,dst)
    assert len(list(label_root.rglob('*_gtFine_labelIds.png')))==500

if all(p.exists() for p in required):
    shutil.copytree(BASE, REPO/'results', dirs_exist_ok=True)
    print('Восстановлены baseline-артефакты:', BASE)
elif old:
    shutil.copytree(old[-1], REPO/'results', dirs_exist_ok=True)
    print('Восстановлены прежние результаты:', old[-1])
else:
    make_cityscapes_available()
    def run_module(*args): subprocess.run([sys.executable,'-m',*args],cwd=REPO,check=True)
    # Для spatial B0→B5 router B2 вообще не используется. Подменяем только bootstrap-B2
    # на второй B0-проход, чтобы получить совместимый cache без тяжёлого B2 на полном кадре.
    # B5 ниже всё равно остаётся настоящей тяжёлой моделью и формирует target router.
    comp_file = REPO / 'features' / 'components.py'
    comp_source = comp_file.read_text(encoding='utf-8')
    comp_source = comp_source.replace('b0, b2 = load_model("b0"), load_model("b2")',
                                      'b0, b2 = load_model("b0"), load_model("b0")')
    comp_file.write_text(comp_source, encoding='utf-8')
    # Длительный, но лёгкий bootstrap: B0 masks и B0 logits. Он нужен один раз.
    run_module('features.components','--dataset','cityscapes')
    run_module('features.logits','--dataset','cityscapes')
    from core.evaluation import routing_cache
    routing_cache('cityscapes','p0'); routing_cache('cityscapes','p1')
    shutil.copytree(REPO/'results', BASE, dirs_exist_ok=True)
    print('Baseline сохранён для следующих запусков:', BASE)

# %%
from core.components import NC
from core.data import DATASETS, get_dataset, loader, population
from core.evaluation import _miou, _miou_batch, entropy_score, routing_cache
from core.models import load_model
from core.tiles import grid, tile_effects, tile_index
from core.tiles import window
from features.logits import LogitStore

TILE, MAP, TOP_K, EPOCHS, SEED = 256, 16, 3, 18, 0
TAG = f'b0spatial_map{MAP}_tile{TILE}_k{TOP_K}'
FEATURE_DIR = OUT / f'{TAG}_features'; FEATURE_DIR.mkdir(exist_ok=True)
size = DATASETS['cityscapes'].size
boxes, tidx = grid(size, TILE), tile_index(size, TILE)
n_tiles = len(boxes)
assert n_tiles == 32, n_tiles
logits = LogitStore('cityscapes')
cache0, cache1 = routing_cache('cityscapes', 'p0'), routing_cache('cityscapes', 'p1')
p0_ids, p1_ids = list(population('cityscapes', 'p0')), list(population('cityscapes', 'p1'))

# B5 запускается только на окнах. Каждый готовый кадр немедленно сохраняется на Drive.
# Если GPU оборвётся, повторная ячейка продолжит с первого отсутствующего файла.
B5_DIR = OUT / 'b5_tile_masks_t256_m128'; B5_DIR.mkdir(exist_ok=True)
ds_full = get_dataset('cityscapes', full=True)

@torch.inference_mode()
def ensure_b5(ids):
    missing = [i for i in ids if not (B5_DIR/f'{i}.npy').exists()]
    if not missing:
        print('B5 cache уже готов:', len(ids)); return
    wanted, model = set(missing), load_model('b5')
    scale, side = 2, TILE + 2*128  # Cityscapes: 2048×1024 → 1024×512
    side_src, started, done = side*scale, __import__('time').time(), 0
    for sample in loader(ds_full):
        image_id = sample['image_id']
        if image_id not in wanted: continue
        src = sample['image'].cuda(); answer = np.zeros((size[1], size[0]), np.uint8)
        for x0,y0,x1,y1 in boxes:
            wx,wy = window((x0,y0,x1,y1),TILE,128,size)
            sx=min(round(wx*scale),src.shape[2]-side_src); sy=min(round(wy*scale),src.shape[1]-side_src)
            q=model(src[:,sy:sy+side_src,sx:sx+side_src][None],(side,side))[0].argmax(0).cpu().numpy()
            answer[y0:y1,x0:x1]=q[y0-wy:y1-wy,x0-wx:x1-wx]
        np.save(B5_DIR/f'{image_id}.npy',answer); done+=1
        if done%10==0 or done==len(missing): print(f'B5 windows {done}/{len(missing)}; {__import__("time").time()-started:.0f} s')
    del model; torch.cuda.empty_cache()

# Только P0 нужен для первого обучения и OOF-защиты. P1 — только после её прохождения.
ensure_b5(p0_ids)

def b5_mask(image_id):
    path = B5_DIR / f'{image_id}.npy'
    assert path.exists(), f'Нет B5 mask: {path}'
    return np.load(path)

def boundary(pred):
    e = np.zeros(pred.shape, np.float32)
    e[:, :-1] = np.maximum(e[:, :-1], pred[:, :-1] != pred[:, 1:])
    e[:-1, :] = np.maximum(e[:-1, :], pred[:-1, :] != pred[1:, :])
    return e

@torch.inference_mode()
def make_feature(image_id, cache):
    """Ровно то, что доступно router до запуска B5: B0 logits/карта и координаты."""
    row = cache['pos'][image_id]
    b0, gt, b5 = cache['b0'][row], cache['gt'][row], b5_mask(image_id)
    p = logits[image_id].softmax(0)
    ent = entropy_score(p)[None]
    top = p.topk(2, dim=0).values
    margin = (top[0] - top[1])[None]
    edge = torch.from_numpy(boundary(b0))[None].cuda()
    yy, xx = torch.meshgrid(torch.linspace(0, 1, size[1], device='cuda'),
                            torch.linspace(0, 1, size[0], device='cuda'), indexing='ij')
    maps = torch.cat([p, ent, margin, edge, xx[None], yy[None]]) # 24 channels
    global_context = torch.cat([p.mean((1, 2)), ent.mean((1, 2))]).cpu().numpy().astype(np.float16)
    x, entropy = [], []
    for x0, y0, x1, y1 in boxes:
        x.append(F.interpolate(maps[:, y0:y1, x0:x1][None], size=(MAP, MAP),
                               mode='bilinear', align_corners=False)[0].cpu().numpy().astype(np.float16))
        entropy.append(float(ent[0, y0:y1, x0:x1].mean().cpu()))
    _, _, delta = tile_effects(gt, b0, b5, tidx, n_tiles)
    return {'x': np.stack(x), 'g': global_context, 'entropy': np.asarray(entropy, np.float32),
            'gain': (delta / (TILE*TILE)).astype(np.float32)}

def build_features(ids, cache, pop):
    for num, image_id in enumerate(ids, 1):
        path = FEATURE_DIR / f'{pop}_{image_id}.npz'
        if not path.exists():
            np.savez_compressed(path, **make_feature(image_id, cache))
        if num % 25 == 0 or num == len(ids): print(f'B0 router input {pop}: {num}/{len(ids)}')

# Создаём вход router лишь для P0. P1 появится только после OOF guard.
build_features(p0_ids, cache0, 'p0')

# %%
class SpatialRouter(nn.Module):
    def __init__(self, channels=24, global_dim=20):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.score = nn.Sequential(nn.Linear(32 + global_dim, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x, g):
        return self.score(torch.cat([self.local(x), g], 1)).squeeze(1)

def read_records(ids, pop):
    records = {}
    for image_id in ids:
        z = np.load(FEATURE_DIR / f'{pop}_{image_id}.npz')
        records[image_id] = {k: z[k] for k in z.files}
    return records

records0 = read_records(p0_ids, 'p0')

# %%
# OOF на P0: сравниваем router с entropy при одинаковом числе B5-плиток.
def fit_router(ids, records):
    x = np.concatenate([records[i]['x'] for i in ids]).astype(np.float32)
    g = np.concatenate([np.repeat(records[i]['g'][None], n_tiles, axis=0) for i in ids]).astype(np.float32)
    y = np.concatenate([records[i]['gain'] for i in ids]).astype(np.float32)
    model = SpatialRouter().cuda(); opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-4)
    order = np.arange(len(y)); gen = np.random.default_rng(SEED)
    for _ in range(EPOCHS):
        gen.shuffle(order)
        for start in range(0, len(y), 128):
            ix = order[start:start+128]
            pred = model(torch.from_numpy(x[ix]).cuda(), torch.from_numpy(g[ix]).cuda())
            loss = F.smooth_l1_loss(pred, torch.from_numpy(y[ix]).cuda(), beta=.02)
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()

@torch.inference_mode()
def scores(model, record):
    x = torch.from_numpy(record['x'].astype(np.float32)).cuda()
    g = torch.from_numpy(np.repeat(record['g'][None], len(x), axis=0).astype(np.float32)).cuda()
    return model(x, g).cpu().numpy()

def selected(score, k):
    return np.argsort(-score, kind='stable')[:k]

from sklearn.model_selection import GroupKFold
fold_scores = {}
for fold, (tr, va) in enumerate(GroupKFold(n_splits=5).split(p0_ids, groups=p0_ids), 1):
    model = fit_router(np.asarray(p0_ids)[tr], records0)
    for image_id in np.asarray(p0_ids)[va]: fold_scores[image_id] = scores(model, records0[image_id])
    del model; torch.cuda.empty_cache(); print('OOF fold', fold, '/ 5 готов')

def confs(ids, cache, records, learned):
    out = {k: {'entropy': [], 'spatial-router': []} for k in (1, 2, 3)}
    for image_id in ids:
        row = cache['pos'][image_id]; gt, b0, b5 = cache['gt'][row], cache['b0'][row], b5_mask(image_id)
        d, c0, _ = tile_effects(gt, b0, b5, tidx, n_tiles)
        for k in (1, 2, 3):
            for name, score in [('entropy', records[image_id]['entropy']), ('spatial-router', learned[image_id])]:
                out[k][name].append(c0 + d[selected(score, k)].sum(0))
    return out

oof_conf = confs(p0_ids, cache0, records0, fold_scores)
rng = np.random.default_rng(0); rows, approved = [], {}
for k in (1, 2, 3):
    base, new = np.stack(oof_conf[k]['entropy']), np.stack(oof_conf[k]['spatial-router'])
    w = rng.multinomial(len(base), np.full(len(base), 1/len(base)), size=2000)
    diff = _miou_batch(np.einsum('bn,nij->bij', w, new)) - _miou_batch(np.einsum('bn,nij->bij', w, base))
    delta = _miou(new.sum(0)) - _miou(base.sum(0)); lo, hi = np.percentile(diff, [2.5, 97.5])
    approved[k] = bool(delta > 0 and lo > 0)
    rows.append({'B5_patches_per_frame':k, 'method':'spatial B0 router', 'p0_oof_mIoU':_miou(new.sum(0)),
                 'delta_to_entropy':delta, 'ci_lo':lo, 'ci_hi':hi, 'approved_for_P1':approved[k]})
p0_report = pd.DataFrame(rows); p0_report.to_csv(OUT/f'{TAG}_p0_oof.csv', index=False); display(p0_report)

# %%
# Финальный P1, только если OOF на P0 действительно лучше entropy.
if not any(approved.values()):
    print('P0 не подтвердил преимущество. P1 не открываем и GPU не тратим.')
else:
    ensure_b5(p1_ids)
    build_features(p1_ids, cache1, 'p1')
    records1 = read_records(p1_ids, 'p1')
    final = fit_router(p0_ids, records0)
    final_scores = {i: scores(final, records1[i]) for i in p1_ids}
    p1_conf = confs(p1_ids, cache1, records1, final_scores)
    rows = []
    for k in (1, 2, 3):
        base, new = np.stack(p1_conf[k]['entropy']), np.stack(p1_conf[k]['spatial-router'])
        # Для бюджета, не прошедшего P0 guard, результат идентичен baseline.
        if not approved[k]: new = base.copy()
        w = rng.multinomial(len(base), np.full(len(base), 1/len(base)), size=2000)
        diff = _miou_batch(np.einsum('bn,nij->bij', w, new)) - _miou_batch(np.einsum('bn,nij->bij', w, base))
        rows.append({'B5_patches_per_frame':k, 'method':'spatial B0 router', 'mIoU':_miou(new.sum(0)),
                     'delta_to_entropy':_miou(new.sum(0))-_miou(base.sum(0)),
                     'ci_lo':np.percentile(diff,2.5), 'ci_hi':np.percentile(diff,97.5),
                     'used_router':approved[k]})
    report = pd.DataFrame(rows); report.to_csv(OUT/f'{TAG}_p1_bootstrap.csv', index=False); display(report)
