"""Построение признаков компонент B0: подгонка на Cityscapes P0, применение ко всем кадрам датасета.

    python -m features.build embeddings
    python -m features.build prototypes
    python -m features.build ddu
    python -m features.build knn
    python -m features.build knn --limit 3        # смоук-тест: по 3 кадра каждой популяции, таблица не пишется
    python -m features.build prototypes -p q=0.2  # параметры построителя

Построитель (features/<name>.py) — класс с методами:
    fit(ctx)       подгонка: проходы по кадрам ctx.fit_ids() через ctx.iterate (только Cityscapes P0)
    apply(fd)      кадр → ({колонка: (n_comp,)}, {оценка: (H·W,) тензор, больше = подозрительнее});
                   попиксельные оценки используются только на кадрах тестовых популяций (fd.test)
    summary()      что записать в config (опционально), save(dir) — подогнанное состояние (опционально)
Общая часть: проход B0 по кадрам с хуками признаков, компоненты по сохранённой маске pred_b0
(порядок проверяется по таблице компонент), попиксельная AUROC «пиксель B0 ошибочный» на тестовых
популяциях (те же 5% пикселей, что и у остальных оценок) рядом с MSP и энтропией.

Выход: results/<dataset>/features/<name>.parquet (image_id, comp_id, колонки построителя) и
results/fits/<name>/ — состояние, pixel_auroc_<dataset>.csv, config_<dataset>.json.
"""
import argparse
import time
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
import torch

from core.components import NC, extract_components
from core.data import (FIT_DATASET, FIT_POP, get_dataset, loader, population, populations,
                       test_populations)
from core.evaluation import PixelSampler, entropy_score, msp_score
from core.models import SegModel, load_model
from core.paths import DatasetResults, fit_dir, parse_params, write_json


@dataclass
class FrameData:
    """Кадр после прохода B0: GT и маска на рабочем разрешении, softmax и признаки модели."""
    image_id: str
    gt: np.ndarray                 # (H, W) uint8, trainIds
    pred: np.ndarray               # (H, W) int64, сохранённая маска pred_b0
    probs: torch.Tensor            # (19, H·W) softmax B0 на GPU
    model: SegModel                # model.features — признаки этого кадра
    size: tuple                    # рабочее разрешение (W, H)
    test: bool = False             # кадр тестовой популяции: нужны попиксельные оценки

    @cached_property
    def components(self):
        """(comp (H, W) int64, classes (n,)) по маске pred_b0."""
        return extract_components(self.pred)

    @property
    def n(self):
        return len(self.components[1])

    @cached_property
    def comp_t(self):
        return torch.from_numpy(self.components[0].ravel().astype(np.int64)).cuda()

    @cached_property
    def pred_t(self):
        return torch.from_numpy(self.pred.ravel()).cuda()

    @cached_property
    def gt_t(self):
        return torch.from_numpy(self.gt.ravel().astype(np.int64)).cuda()

    @cached_property
    def z(self):
        """Вход classifier, поднятый до рабочего разрешения: (256, H·W)."""
        return self.model.c_map(self.size)

    @cached_property
    def zn(self):
        """То же, L2-нормированное в каждом пикселе."""
        return self.model.c_map(self.size, normalize=True)


class Context:
    def __init__(self, dataset, limit=None, seed=0):
        self.dataset, self.limit, self.seed = dataset, limit, seed
        self.b0 = load_model("b0")

    def fit_ids(self):
        """Кадры популяции подгонки, по порядку image_id."""
        return population(FIT_DATASET, FIT_POP)[: self.limit]

    def apply_ids(self):
        """Кадры датасета для применения: все или (смоук-тест) первые limit каждой популяции."""
        if self.limit is None:
            return None
        return sorted({i for pop in populations(self.dataset)
                       for i in population(self.dataset, pop)[: self.limit]})

    @torch.inference_mode()
    def iterate(self, ids, dataset=FIT_DATASET):
        """Проход B0 по кадрам ids (None — все кадры датасета): FrameData на каждый кадр."""
        ds, res = get_dataset(dataset, ids, full=False), DatasetResults(dataset)
        for sample in loader(ds):
            image_id = sample["image_id"]
            probs = self.b0.probs(sample, ds.size).reshape(NC, -1)
            pred = np.load(res.mask(image_id))["pred_b0"].astype(np.int64)
            yield FrameData(image_id, sample["label"].numpy(), pred, probs, self.b0, ds.size)


def builders():
    from features import ddu, embeddings, knn, prototypes
    return {"embeddings": embeddings.Embeddings, "prototypes": prototypes.Prototypes,
            "ddu": ddu.DDU, "knn": knn.KNN}


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=sorted(builders()))
    ap.add_argument("--dataset", default=FIT_DATASET)
    ap.add_argument("-p", "--param", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--pixel-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="смоук-тест: первые N кадров популяций")
    args = ap.parse_args()

    params = parse_params(args.param)
    ctx = Context(args.dataset, args.limit, args.seed)
    builder = builders()[args.name](**params)
    t0 = time.time()
    builder.fit(ctx)
    print(f"подгонка: {time.time() - t0:.0f} с")

    res = DatasetResults(args.dataset)
    table = pd.read_parquet(res.components, columns=["image_id", "comp_id", "class_id", "area"])
    rows = table.groupby("image_id", sort=False).indices
    test = {i: pop for pop in test_populations(args.dataset) for i in population(args.dataset, pop)}
    sampler = PixelSampler(args.pixel_frac, args.seed)
    aggs, keys, agree = [], [], []
    for k, fd in enumerate(ctx.iterate(ctx.apply_ids(), args.dataset), 1):
        comp, classes = fd.components
        t = table.iloc[rows[fd.image_id]]
        assert (t.comp_id.values == np.arange(fd.n)).all(), fd.image_id
        assert (t.class_id.values == classes).all(), fd.image_id
        assert (t.area.values == np.bincount(comp.ravel(), minlength=fd.n)).all(), fd.image_id
        agree.append((fd.probs.argmax(0) == fd.pred_t).double().mean().item())
        fd.test = fd.image_id in test
        cols, pixel = builder.apply(fd)
        aggs.append(pd.DataFrame(cols))
        keys.append(t[["image_id", "comp_id"]])
        if fd.test:
            sel = sampler.frame(f"{args.dataset}_{test[fd.image_id]}", fd.gt, fd.pred)
            sel_t = torch.from_numpy(sel).cuda()
            ps = fd.probs[:, sel_t]
            sampler.add("MSP", msp_score(ps).cpu().numpy())
            sampler.add("entropy", entropy_score(ps).cpu().numpy())
            for name, x in pixel.items():
                sampler.add(name, x[sel_t].cpu().numpy())
        if k % 50 == 0:
            print(f"применение: {k} кадров  {time.time() - t0:.0f} с")

    out_table = pd.concat([pd.concat(keys, ignore_index=True), pd.concat(aggs, ignore_index=True)],
                          axis=1)
    assert np.isfinite(out_table.drop(columns=["image_id", "comp_id"]).to_numpy(float)).all()
    if args.limit is None:
        assert out_table[["image_id", "comp_id"]].equals(table[["image_id", "comp_id"]])
        res.feature(args.name).parent.mkdir(parents=True, exist_ok=True)
        out_table.to_parquet(res.feature(args.name))

    out = fit_dir(args.name if args.limit is None else f"smoke_{args.name}")
    out.mkdir(parents=True, exist_ok=True)
    if hasattr(builder, "save"):
        builder.save(out)
    auroc = sampler.auroc_table() if sampler.cols else None
    if auroc is not None:
        auroc.to_csv(out / f"pixel_auroc_{args.dataset}.csv", index=False)
    write_json(out / f"config_{args.dataset}.json",
               {"builder": args.name, "params": params, "seed": args.seed,
                "pixel_frac": args.pixel_frac, "limit": args.limit,
                **(builder.summary() if hasattr(builder, "summary") else {}),
                "argmax_agreement_min": min(agree)})

    pd.set_option("display.width", 200)
    print(f"таблица признаков: {out_table.shape}"
          + ("" if args.limit is None else " (смоук-тест: не сохранена)")
          + f"; совпадение argmax B0 с маской: мин {min(agree):.6f}; {time.time() - t0:.0f} с")
    if auroc is not None:
        print("AUROC «пиксель B0 ошибочный»:")
        print(auroc.sort_values(["domain", "auroc"], ascending=[True, False]).round(4)
              .to_string(index=False))


if __name__ == "__main__":
    main()
