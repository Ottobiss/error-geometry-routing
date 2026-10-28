# %% [markdown]
# # Global-preview B5 + local B5 fusion
#
# Новый эксперимент после отрицательного результата pixel gate.  B0 обрабатывает
# весь кадр. B5 сначала видит **весь кадр в малом разрешении** (global preview),
# затем B5 в полном разрешении работает лишь на трёх плитках, выбранных энтропией
# B0. Линейный fusion-head получает лог-вероятности B0, global-preview B5 и
# local B5. Он обучается и выбирается только на P0; P1 не читается до P0-OOF guard.
#
# Все тяжёлые результаты по кадрам пишутся в Drive сразу. После потери GPU
# повторный запуск продолжит с готовых .npz.

# %%
# Ячейка 1. Подключение Drive и зависимости.
from pathlib import Path
from google.colab import drive

MOUNT = Path('/content/gdrive')
if not (MOUNT / 'MyDrive').exists():
    drive.mount(str(MOUNT))
%pip -q install --upgrade-strategy only-if-needed gdown transformers cityscapesscripts pyarrow

# %%
# Ячейка 2. Код, исходные B0-результаты и Cityscapes validation.
import os, shutil, subprocess, sys, zipfile, json, time
import gdown
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

assert torch.cuda.is_available(), 'В Colab включите Runtime → Change runtime type → T4 GPU.'
REPO = Path('/content/error-geometry-routing')
DRIVE_DIR = MOUNT / 'MyDrive' / 'error-geometry-routing'
OUT = DRIVE_DIR / 'global_preview_fusion_20261027'
OUT.mkdir(parents=True, exist_ok=True)

if not REPO.exists():
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'experiments/frozen-router-results-20261025',
                    'https://github.com/Ottobiss/error-geometry-routing.git', str(REPO)], check=True)
os.chdir(REPO)

# Нужны сохранённые B0 logits/masks из прошлого запуска. Берём последний подходящий каталог.
old = sorted(DRIVE_DIR.glob('posthoc_router_*/results'))
assert old, 'Не найдена папка posthoc_router_.../results в Google Drive.'
shutil.copytree(old[-1], REPO / 'results', dirs_exist_ok=True)

download = Path('/content/cityscapes-archives'); download.mkdir(exist_ok=True)
archives = {'val.zip': '1aErjLEt25oJEX48W-zxHwITEvNAeKxGJ',
            'gtFine_trainvaltest.zip': '1AxGkcacU_SZXPSjxUtzKgT3XSZM032bk'}
for name, file_id in archives.items():
    path = download / name
    if not path.exists() or path.stat().st_size == 0:
        print('Скачиваю', name)
        gdown.download(id=file_id, output=str(path), quiet=False)

data = REPO / 'data' / 'cityscapes'; data.mkdir(parents=True, exist_ok=True)
for name in archives:
    with zipfile.ZipFile(download / name) as zf:
        for item in zf.infolist():
            target = data / item.filename
            if not target.exists():
                zf.extract(item, data)

def run_module(*args):
    # В Colab stderr дочернего процесса иногда скрывается за CalledProcessError.
    # Печатаем его явно: следующая ошибка будет диагностируемой.
    p = subprocess.run([sys.executable, '-m', *args], cwd=REPO, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout)
    if p.returncode:
        raise RuntimeError(f'Не выполнился python -m {" ".join(args)}; см. текст выше.')

# Разные архивы Cityscapes иногда содержат дополнительную верхнюю папку. Нормализуем
# их в единственный путь, который ожидает core.data, не перемещая и не удаляя архивные файлы.
def normalise_cityscapes(pattern, expected):
    target_root = data / expected / 'val'
    for source in data.rglob(pattern):
        if target_root in source.parents:
            continue
        target = target_root / source.parent.name / source.name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

normalise_cityscapes('*_leftImg8bit.png', 'leftImg8bit')
normalise_cityscapes('*_gtFine_labelIds.png', 'gtFine')
images = list((data / 'leftImg8bit' / 'val').rglob('*_leftImg8bit.png'))
labels = list((data / 'gtFine' / 'val').rglob('*_gtFine_labelIds.png'))
assert len(images) == len(labels) == 500, (len(images), len(labels), data)

# components.parquet для этого эксперимента не нужен. Нужны только B0/B2 masks
# (routing_cache) и B0 logits. Если старый results неполный — это будет осмысленный
# fallback, а не скрытый запуск из-за отсутствия посторонней таблицы компонент.
masks = list((REPO / 'results/cityscapes/masks').glob('*.npz'))
if len(masks) != 500:
    print('В сохранённых results нет всех B0/B2 masks; строю их один раз.')
    run_module('features.components', '--dataset', 'cityscapes')
if not (REPO / 'results/cityscapes/logits_b0.npy').exists():
    run_module('features.logits', '--dataset', 'cityscapes')
print('Подготовка готова:', OUT)

# %%
# Ячейка 3. Конфигурация и вспомогательные функции.
# PREVIEW_* — дешёвый обзор B5 всей сцены. TILE/MARGIN — дорогой локальный B5.
from core.components import IGNORE, NC
from core.data import get_dataset, loader, population
from core.evaluation import _miou, _miou_batch, entropy_score, routing_cache
from core.models import load_model
from core.tiles import grid, window, tile_index
from features.logits import LogitStore
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GroupKFold

TILE, MARGIN = 256, 128
PREVIEW_INPUT_WH = (512, 256)       # B5 видит всю сцену, но в 16 раз меньше пикселей исходника
PREVIEW_LOGITS_WH = (256, 128)      # компактный checkpoint: примерно 0.6 GB на 500 кадров fp16
TOP_K, SAMPLES_PER_TILE, SEED = 3, 800, 17
TAG = f'preview{PREVIEW_INPUT_WH[0]}x{PREVIEW_INPUT_WH[1]}_tile{TILE}_m{MARGIN}_k{TOP_K}'
PREVIEW_DIR = OUT / f'{TAG}_preview_logits'; PREVIEW_DIR.mkdir(exist_ok=True)
LOCAL_DIR = OUT / f'{TAG}_local_logits'; LOCAL_DIR.mkdir(exist_ok=True)

res = __import__('core.paths', fromlist=['DatasetResults']).DatasetResults('cityscapes')
ds = get_dataset('cityscapes', full=True)
boxes = grid(ds.size, TILE)
tidx = tile_index(ds.size, TILE)
assert ds.size[0] % TILE == 0 and ds.size[1] % TILE == 0, 'Этот ноутбук предполагает целые 256-плитки.'
logits_b0 = LogitStore('cityscapes')
rng = np.random.default_rng(SEED)

def image_file(folder, image_id):
    return folder / f'{image_id}.npz'

def selected_tiles(image_id):
    """Ровно тот же честный selector, что baseline: средняя softmax-энтропия B0."""
    p = logits_b0[image_id].softmax(0)
    e = entropy_score(p).cpu().numpy()
    scores = np.array([e[y0:y1, x0:x1].mean() for x0, y0, x1, y1 in boxes])
    return np.argsort(-scores, kind='stable')[:TOP_K].astype(np.int16), scores.astype(np.float32)

def preview_logits_full(image_id):
    z = np.load(image_file(PREVIEW_DIR, image_id))['logits'].astype(np.float32)
    return F.interpolate(torch.from_numpy(z)[None], size=(ds.size[1], ds.size[0]),
                         mode='bilinear', align_corners=False)[0].numpy()

def frame_inputs(image_id, local):
    """Возвращает X для top-K плиток: [B0 logp, B5 preview logp, B5 local logp]."""
    b0 = logits_b0[image_id].cpu().numpy().astype(np.float32)
    prev = preview_logits_full(image_id)
    ids = local['tiles'].astype(int)
    loc = local['logits'].astype(np.float32)
    out = []
    for j, tile in enumerate(ids):
        x0, y0, x1, y1 = boxes[tile]
        a = torch.from_numpy(b0[:, y0:y1, x0:x1]).log_softmax(0).numpy()
        b = torch.from_numpy(prev[:, y0:y1, x0:x1]).log_softmax(0).numpy()
        c = torch.from_numpy(loc[j]).log_softmax(0).numpy()
        out.append(np.concatenate([a, b, c], axis=0).transpose(1, 2, 0).reshape(-1, 3 * NC))
    return out

def local_hard_mask(local):
    return [a.argmax(0).astype(np.uint8) for a in local['logits']]

def confusion(gt, pred):
    valid = gt != IGNORE
    return np.bincount(gt[valid].astype(np.int64) * NC + pred[valid], minlength=NC * NC).reshape(NC, NC)

def fusion_mask(b0, local, xs, clf, count):
    """B0 вне плиток; внутри первых count entropy-плиток — fusion-head."""
    pred = b0.copy()
    for j, tile in enumerate(local['tiles'][:count]):
        x0, y0, x1, y1 = boxes[int(tile)]
        pred[y0:y1, x0:x1] = clf.predict(xs[j]).reshape(y1-y0, x1-x0).astype(np.uint8)
    return pred

def b5_mask(b0, local, count):
    pred = b0.copy()
    for j, tile in enumerate(local['tiles'][:count]):
        x0, y0, x1, y1 = boxes[int(tile)]
        pred[y0:y1, x0:x1] = local['logits'][j].argmax(0).astype(np.uint8)
    return pred

print('Конфигурация:', TAG, '; плиток в кадре:', len(boxes))

# %%
# Ячейка 4. B5-preview и локальные B5 logits **только для P0**.
# Каждый кадр сохраняется сразу; можно безопасно перезапускать эту ячейку после отключения GPU.
@torch.inference_mode()
def save_b5_for_ids(ids):
    need = [x for x in ids if not image_file(PREVIEW_DIR, x).exists() or not image_file(LOCAL_DIR, x).exists()]
    if not need:
        print('Все нужные B5-кеши уже есть:', len(ids)); return
    model = load_model('b5')
    wanted = set(need); start = time.time(); done = 0
    for sample in loader(ds):
        image_id = sample['image_id']
        if image_id not in wanted:
            continue
        src = sample['image'].cuda()
        preview_path, local_path = image_file(PREVIEW_DIR, image_id), image_file(LOCAL_DIR, image_id)
        if not preview_path.exists():
            small = F.interpolate(src[None].float(), size=(PREVIEW_INPUT_WH[1], PREVIEW_INPUT_WH[0]),
                                  mode='bilinear', align_corners=False).to(torch.uint8)
            q = model(small, PREVIEW_LOGITS_WH)[0].cpu().numpy().astype(np.float16)
            np.savez_compressed(preview_path, logits=q)
        if not local_path.exists():
            chosen, entropy = selected_tiles(image_id)
            scale = src.shape[2] / ds.size[0]
            side = TILE + 2 * MARGIN; side_src = round(side * scale)
            answers = []
            for tile in chosen:
                x0, y0, x1, y1 = boxes[int(tile)]
                wx, wy = window((x0, y0, x1, y1), TILE, MARGIN, ds.size)
                sx = min(round(wx * scale), src.shape[2] - side_src)
                sy = min(round(wy * scale), src.shape[1] - side_src)
                crop = src[:, sy:sy+side_src, sx:sx+side_src][None]
                q = model(crop, (side, side))[0].cpu().numpy()
                answers.append(q[:, y0-wy:y1-wy, x0-wx:x1-wx].astype(np.float16))
            np.savez_compressed(local_path, tiles=chosen, entropy=entropy, logits=np.stack(answers))
        done += 1
        if done % 10 == 0 or done == len(need):
            print(f'B5 cache {done}/{len(need)}; {time.time()-start:.0f} s; {image_id}')
    del model
    torch.cuda.empty_cache()

p0_ids = list(population('cityscapes', 'p0'))
save_b5_for_ids(p0_ids)
print('P0 B5-cache сохранён в:', OUT)

# %%
# Ячейка 5. Обучение fusion-head и строгая OOF-проверка P0.
# P1 здесь НЕ используется даже для выбора параметров.
cache0 = routing_cache('cityscapes', 'p0')

def sampled_frame(image_id):
    k = cache0['pos'][image_id]
    gt = cache0['gt'][k]
    local = np.load(image_file(LOCAL_DIR, image_id))
    xs = frame_inputs(image_id, local)
    xx, yy = [], []
    for j, tile in enumerate(local['tiles']):
        x0, y0, x1, y1 = boxes[int(tile)]
        label = gt[y0:y1, x0:x1].reshape(-1)
        valid = np.flatnonzero(label != IGNORE)
        take = rng.choice(valid, min(SAMPLES_PER_TILE, len(valid)), replace=False)
        xx.append(xs[j][take]); yy.append(label[take])
    return np.concatenate(xx).astype(np.float32), np.concatenate(yy).astype(np.int64)

samples = {image_id: sampled_frame(image_id) for image_id in p0_ids}
print('P0 sampled pixels:', sum(len(y) for _, y in samples.values()))

def fit_head(ids):
    x = np.concatenate([samples[i][0] for i in ids])
    y = np.concatenate([samples[i][1] for i in ids])
    # Линейный residual-like fusion устойчивее маленькой MLP: меньше переобучение на P0.
    return SGDClassifier(loss='log_loss', alpha=3e-4, max_iter=80, tol=1e-3,
                         class_weight='balanced', random_state=SEED).fit(x, y)

fold_conf = {budget: {'b5': [], 'fusion': []} for budget in (1, 2, 3)}
groups = np.asarray(p0_ids)
for fold, (tr, va) in enumerate(GroupKFold(n_splits=5).split(groups, groups=groups), 1):
    model = fit_head(groups[tr])
    for image_id in groups[va]:
        k = cache0['pos'][image_id]; gt, b0 = cache0['gt'][k], cache0['b0'][k]
        local = np.load(image_file(LOCAL_DIR, image_id)); xs = frame_inputs(image_id, local)
        for budget in (1, 2, 3):
            fold_conf[budget]['b5'].append(confusion(gt, b5_mask(b0, local, budget)))
            fold_conf[budget]['fusion'].append(confusion(gt, fusion_mask(b0, local, xs, model, budget)))
    print('OOF fold', fold, '/ 5 готов')

rng_boot = np.random.default_rng(0)
rows, allowed = [], {}
for budget in (1, 2, 3):
    base = np.stack(fold_conf[budget]['b5']); fused = np.stack(fold_conf[budget]['fusion'])
    weights = rng_boot.multinomial(len(base), np.full(len(base), 1/len(base)), size=2000)
    d = _miou_batch((weights @ fused).reshape(-1, NC, NC)) - _miou_batch((weights @ base).reshape(-1, NC, NC))
    delta = _miou(fused.sum(0)) - _miou(base.sum(0))
    lo, hi = np.percentile(d, [2.5, 97.5])
    allowed[budget] = bool(delta > 0 and lo > 0)
    rows.append({'B5_patches_per_frame': budget, 'method': 'global-preview + local-logit fusion',
                 'p0_oof_mIoU': _miou(fused.sum(0)), 'delta_to_entropy_B5': delta,
                 'ci_lo': lo, 'ci_hi': hi, 'approved_for_P1': allowed[budget]})
p0_report = pd.DataFrame(rows)
p0_report.to_csv(OUT / f'{TAG}_p0_oof_guard.csv', index=False)
p0_report

# %%
# Ячейка 6. Только если P0 доказал выигрыш: один честный запуск P1.
# При отсутствии approved_for_P1 эта ячейка ничего тяжёлого не запускает.
if not any(allowed.values()):
    print('P0-OOF не доказал улучшение: P1 намеренно не запускаем. Это честный отрицательный результат.')
else:
    p1_ids = list(population('cityscapes', 'p1'))
    save_b5_for_ids(p1_ids)
    final_head = fit_head(p0_ids)
    cache1 = routing_cache('cityscapes', 'p1')
    result = []
    for budget in (1, 2, 3):
        base, fused = [], []
        for image_id in p1_ids:
            k = cache1['pos'][image_id]; gt, b0 = cache1['gt'][k], cache1['b0'][k]
            local = np.load(image_file(LOCAL_DIR, image_id)); xs = frame_inputs(image_id, local)
            base.append(confusion(gt, b5_mask(b0, local, budget)))
            # Если конкретный budget P0 не прошёл guard, P1 получает строго baseline.
            pred = fusion_mask(b0, local, xs, final_head, budget) if allowed[budget] else b5_mask(b0, local, budget)
            fused.append(confusion(gt, pred))
        base, fused = np.stack(base), np.stack(fused)
        weights = rng_boot.multinomial(len(base), np.full(len(base), 1/len(base)), size=2000)
        d = _miou_batch((weights @ fused).reshape(-1, NC, NC)) - _miou_batch((weights @ base).reshape(-1, NC, NC))
        result.append({'B5_patches_per_frame': budget, 'method': 'global-preview + local-logit fusion',
                       'mIoU': _miou(fused.sum(0)), 'delta_to_entropy_B5': _miou(fused.sum(0))-_miou(base.sum(0)),
                       'ci_lo': np.percentile(d, 2.5), 'ci_hi': np.percentile(d, 97.5),
                       'used_fusion': allowed[budget]})
    p1_report = pd.DataFrame(result)
    p1_report.to_csv(OUT / f'{TAG}_p1_miou_bootstrap.csv', index=False)
    print('Сохранено:', OUT / f'{TAG}_p1_miou_bootstrap.csv')
    display(p1_report)
