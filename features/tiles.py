"""Ответ B2 по патчам с контекстом и замеры времени — для каскада по патчам (routing.tiles).

    python -m features.tiles --dataset cityscapes --input image      # B2 на окнах из полного кадра
    python -m features.tiles --dataset cityscapes --input small      # из кадра рабочего разрешения
    python -m features.tiles --dataset acdc --input small --limit 3  # смоук-тест, ничего не пишется

Для каждого патча сетки tile×tile (core.tiles) B2 получает окно (tile + 2·margin)² вокруг него из кадра
input (image — исходное разрешение, small — рабочее; окно масштабируется вместе с кадром), ответ на окне
поднимается до рабочего разрешения, из него берётся патч. Так B2 видит только контекст окна, как в
реальном каскаде. Заодно — B2 на всём кадре того же входа (опорная точка).
Выход: results/<dataset>/tiles_<cfg>.npz — image_id, tiled и full (n, H, W) uint8;
tiles_<cfg>.json — время (мс, fp32): B0 на кадре, B2 на кадре, B2 на пачках из 1…max_batch окон.
"""
import argparse
import json
import time

import numpy as np
import torch

from core.components import IGNORE, NC
from core.data import DATASETS, get_dataset, loader
from core.models import load_model
from core.paths import DatasetResults
from core.tiles import batch_time, grid, window


def config_name(inp, tile, margin, heavy="b2"):
    """Имя кеша. Сохраняем старое имя B2, B5 хранится рядом без перезаписи baseline."""
    prefix = "" if heavy == "b2" else f"{heavy}_"
    return f"{prefix}{inp}_t{tile}_m{margin}"


def tiled_prediction(b2, src, size, boxes, tile, margin, max_batch):
    """Маска B2 по патчам (H, W) uint8: src — кадр uint8 (3, Hs, Ws) на GPU, size — рабочее (W, H)."""
    s = src.shape[2] / size[0]
    w = tile + 2 * margin
    ws = round(w * s)
    out = np.zeros((size[1], size[0]), np.uint8)
    crops, places = [], []
    for box in boxes:
        wx, wy = window(box, tile, margin, size)
        sx, sy = min(round(wx * s), src.shape[2] - ws), min(round(wy * s), src.shape[1] - ws)
        crops.append(src[:, sy:sy + ws, sx:sx + ws])
        places.append((box, wx, wy))
    for i in range(0, len(crops), max_batch):
        pred = b2(torch.stack(crops[i:i + max_batch]), (w, w)).argmax(1).cpu().numpy()
        for p, (box, wx, wy) in zip(pred, places[i:i + max_batch]):
            x0, y0, x1, y1 = box
            out[y0:y1, x0:x1] = p[y0 - wy:y1 - wy, x0 - wx:x1 - wx]
    return out


def ms(fn, reps=10, blocks=7):
    """Время вызова (мс): медиана по blocks блокам из reps вызовов — устойчиво к троттлингу ноутбука."""
    for _ in range(5):
        fn()
    out = []
    for _ in range(blocks):
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t) / reps * 1000)
    return float(np.median(out))


def timings(b0, b2, size, src_shape, ws, max_batch, include_full=True):
    """Время (мс): B0 на кадре рабочего разрешения, B2 на кадре src_shape, B2 на пачках окон ws×ws."""
    def rand(*shape):
        return torch.randint(0, 255, shape, dtype=torch.uint8, device="cuda")
    frame, src, win = rand(1, 3, size[1], size[0]), rand(1, *src_shape), rand(max_batch, 3, ws, ws)
    return {"b0_frame": ms(lambda: b0(frame, size)),
            "b2_frame": ms(lambda: b2(src, size)) if include_full else None,
            "b2_batch": [0.0] + [ms(lambda b=b: b2(win[:b], (ws, ws))) for b in range(1, max_batch + 1)]}


def miou(conf):
    d = np.diag(conf)
    u = conf.sum(0) + conf.sum(1) - d
    return 100 * np.nanmean(np.where(u > 0, d / np.maximum(u, 1), np.nan))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cityscapes")
    ap.add_argument("--input", choices=["image", "small"], default="small")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--margin", type=int, default=64)
    ap.add_argument("--heavy", choices=("b2", "b5"), default="b2",
                    help="тяжёлая модель; B5 пишет отдельный b5_* кеш")
    ap.add_argument("--max-batch", type=int, default=2,
                    help="окон в пачке B2; 8 ГБ видеопамяти — не больше 2 окон 768² или 1024²")
    ap.add_argument("--no-full", action="store_true",
                    help="не запускать тяжёлую модель на полном кадре: нужно для B5 на T4 и не влияет на роутинг по патчам")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--timing-only", action="store_true", help="только перемерить время в tiles_<cfg>.json")
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp32",
                    help="для --timing-only: fp16 — под autocast, пишется в ms_fp16 (ms — fp32)")
    args = ap.parse_args()

    if args.timing_only:
        res = DatasetResults(args.dataset)
        path = res.tiles(config_name(args.input, args.tile, args.margin, args.heavy), "json")
        info = json.loads(path.read_text(encoding="utf-8"))
        size = DATASETS[args.dataset].size
        src_shape = next(iter(loader(get_dataset(args.dataset, full=args.input == "image"))))[args.input].shape
        key = "ms" if args.precision == "fp32" else "ms_fp16"
        with torch.autocast("cuda", dtype=torch.float16, enabled=args.precision == "fp16"):
            info[key] = timings(load_model("b0"), load_model(args.heavy), size, tuple(src_shape),
                                info["window_input_px"], info["max_batch"], not args.no_full)
        path.write_text(json.dumps(info, ensure_ascii=False, indent=1))
        print(path, {k: v if k != "b2_batch" else [round(x, 1) for x in v] for k, v in info[key].items()})
        return

    full = args.input == "image"
    ds = get_dataset(args.dataset, full=full)
    if args.limit is not None:
        ds = get_dataset(args.dataset, ds.image_ids[: args.limit], full=full)
    res, cfg = DatasetResults(args.dataset), config_name(args.input, args.tile, args.margin, args.heavy)
    boxes = grid(ds.size, args.tile)
    b0, b2 = load_model("b0"), load_model(args.heavy)

    tiled, whole, t0 = [], [], time.time()
    conf = {k: np.zeros((NC, NC), np.int64) for k in (("b0", "tiled") if args.no_full else ("b0", "full", "tiled"))}
    for k, sample in enumerate(loader(ds), 1):
        src = sample[args.input].cuda()
        tiled.append(tiled_prediction(b2, src, ds.size, boxes, args.tile, args.margin, args.max_batch))
        if not args.no_full:
            whole.append(b2(src[None], ds.size).argmax(1)[0].cpu().numpy().astype(np.uint8))
        gt = sample["label"].numpy()
        v = gt != IGNORE
        b0_pred = np.load(res.mask(sample["image_id"]))["pred_b0"]
        predictions = [("b0", b0_pred), ("tiled", tiled[-1])]
        if not args.no_full:
            predictions.append(("full", whole[-1]))
        for key, pred in predictions:
            conf[key] += np.bincount(gt[v].astype(np.int64) * NC + pred[v], minlength=NC * NC).reshape(NC, NC)
        if k % 50 == 0 or k == len(ds):
            print(f"{k}/{len(ds)}  {time.time() - t0:.0f} с")

    s = sample[args.input].shape
    ws = round((args.tile + 2 * args.margin) * s[2] / ds.size[0])
    timing = timings(b0, b2, ds.size, tuple(s), ws, args.max_batch, not args.no_full)
    info = {"config": cfg, "heavy": args.heavy, "tile": args.tile, "margin": args.margin, "input": args.input,
            "window_input_px": ws, "n_tiles": len(boxes), "max_batch": args.max_batch,
            "ms": timing, "miou_all_frames": {k: miou(c) for k, c in conf.items()}}
    print(json.dumps({k: v for k, v in info.items() if k != "ms"}, ensure_ascii=False))
    all_windows = batch_time(len(boxes), timing["b2_batch"], args.max_batch)
    frame_ms = 'не измерялось' if timing['b2_frame'] is None else f"{timing['b2_frame']:.1f}"
    print(f"мс: B0 {timing['b0_frame']:.1f}, {args.heavy.upper()} на кадре {frame_ms}, {args.heavy.upper()} на всех {len(boxes)} "
          f"окнах {all_windows:.1f}, на одном {timing['b2_batch'][1]:.1f}")
    if args.limit is None:
        arrays = {"image_id": np.array(ds.image_ids), "tiled": np.stack(tiled)}
        if not args.no_full:
            arrays["full"] = np.stack(whole)
        np.savez_compressed(res.tiles(cfg), **arrays)
        res.tiles(cfg, "json").write_text(json.dumps(info, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
