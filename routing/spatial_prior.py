"""Пространственный приор пользы B2: сосредоточена ли Δ по положению в кадре и по размеру объектов.

    python -m routing.spatial_prior                   # CS P1 и весь ACDC

Проверка перед обучением плотной головы Δ на признаках B0: если польза B2 сосредоточена по положению
(например, у горизонта — мелкие дальние объекты), голове есть что выучить сверх энтропии.
Разрешение — выход B0 (1/4 рабочего, логиты results/<dataset>/logits_b0.npy), маски берутся через 4 пикселя.
Пиксель: G — B2 права, B0 ошиблась; L — наоборот; польза пикселя y = G − L (только размеченные).
Оценки пикселя: энтропия B0; приор — средняя y по положению на CS P0 (сглажена окном 9×9);
сочетание — линейная регрессия y по [энтропия, приор, энтропия·приор] на пикселях CS P0.
На ACDC отдельно приор по остальным трём условиям ACDC (leave-one-condition-out).
Метрика — доля пользы оракула, собранная при бюджете b пикселей каждого кадра (B2 в b лучших по оценке
пикселях): Σ y в выбранных / Σ y оракула (b лучших пикселей по самой y). Это пиксельное приближение
роутинга, а не mIoU. Плюс AUROC «B2 исправляет пиксель» (G) и профиль пользы по высоте кадра и по
площади компонент B0. Bootstrap по кадрам разницы «оценка − энтропия» при 20%.
Выход: results/summary/spatial_prior.csv и spatial_prior_profile.csv.
"""
import argparse

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import uniform_filter, zoom
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

from core.components import IGNORE
from core.data import FIT_DATASET, FIT_POP, population
from core.evaluation import entropy_score, routing_cache
from core.paths import RESULTS, DatasetResults

TESTS = "cityscapes:p1,acdc:all"
BUDGETS = [0.05, 0.10, 0.20, 0.30]
STRIDE = 4


def pixels(dataset, pop):
    """Энтропия B0, польза y и маска размеченных (F, h, w), image_id и площадь компонент B0 пикселя."""
    res, cache = DatasetResults(dataset), routing_cache(dataset, pop)
    ids = population(dataset, pop)
    rows = {i: k for k, i in enumerate(res.logits_ids.read_text().split())}
    logits = np.load(res.logits, mmap_mode="r")
    ent, y, valid, comp_area = [], [], [], []
    for image_id in ids:
        k = cache["pos"][image_id]
        p = torch.from_numpy(np.asarray(logits[rows[image_id]], dtype=np.float32)).softmax(0)
        ent.append(entropy_score(p).numpy())
        gt, b0, b2 = (cache[n][k][::STRIDE, ::STRIDE] for n in ("gt", "b0", "b2"))
        v = gt != IGNORE
        y.append(((b2 == gt) & (b0 != gt)).astype(np.int8) - ((b0 == gt) & (b2 != gt)))
        valid.append(v)
        comp = cache["comp"][k].astype(np.int64)
        comp_area.append(np.bincount(comp.ravel())[comp][::STRIDE, ::STRIDE])
    return np.stack(ent), np.stack(y), np.stack(valid), ids, np.stack(comp_area)


def prior_map(y, valid):
    """Средняя польза по положению, сглаженная окном 9×9 (h, w)."""
    num = uniform_filter((y * valid).sum(0).astype(float), 9, mode="nearest")
    den = uniform_filter(valid.sum(0).astype(float), 9, mode="nearest")
    return num / np.maximum(den, 1e-9)


def captured(score, y, valid, budget):
    """Польза по кадрам (F,): Σ y в budget-доле лучших по score пикселей кадра; неразмеченные — 0."""
    F, n = len(y), y[0].size
    k = int(round(budget * n))
    s = np.where(valid, score, -np.inf).reshape(F, n)
    top = np.argpartition(-s, k - 1, axis=1)[:, :k]
    return np.take_along_axis((y * valid).reshape(F, n), top, 1).sum(1)


def fit_combo(ent, prior, y, valid, n=500_000, seed=0):
    """Линейная регрессия y по [e, p, e·p] на случайных размеченных пикселях."""
    rng = np.random.default_rng(seed)
    f, i, j = np.nonzero(valid)
    s = rng.choice(len(f), min(n, len(f)), replace=False)
    e, p = ent[f[s], i[s], j[s]], prior[i[s], j[s]]
    return LinearRegression().fit(np.c_[e, p, e * p], y[f[s], i[s], j[s]])


def combo_score(model, ent, prior):
    p = np.broadcast_to(prior, ent.shape)
    return (np.stack([ent, p, ent * p], -1) @ model.coef_ + model.intercept_).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default=TESTS)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    e0, y0, v0, _, _ = pixels(FIT_DATASET, FIT_POP)
    prior_cs = prior_map(y0, v0)
    combo_cs = fit_combo(e0, prior_cs, y0, v0)
    print(f"CS P0: приор пользы по положению от {prior_cs.min():+.4f} до {prior_cs.max():+.4f}, "
          f"среднее {(y0 * v0).sum() / v0.sum():+.4f}; сочетание: коэф. [e, p, e·p] = "
          f"{np.round(combo_cs.coef_, 4).tolist()}")

    rows, prof = [], []
    for spec in args.tests.split(","):
        dataset, pop = spec.split(":")
        ent, y, valid, ids, carea = pixels(dataset, pop)
        # Приор CS — в относительных координатах кадра (у ACDC другое соотношение сторон).
        prior = zoom(prior_cs, np.divide(ent.shape[1:], prior_cs.shape), order=1)
        scores = {"энтропия": ent, "приор CS": np.broadcast_to(prior, ent.shape),
                  "энтропия+приор CS": combo_score(combo_cs, ent, prior),
                  "оракул": y.astype(np.float32)}
        if dataset == "acdc":
            cond = np.array([i.split("_")[0] for i in ids])
            loco_p, loco_c = np.empty_like(ent), np.empty_like(ent)
            for c in np.unique(cond):
                tr, te = cond != c, cond == c
                p = prior_map(y[tr], valid[tr])
                loco_p[te] = p
                loco_c[te] = combo_score(fit_combo(ent[tr], p, y[tr], valid[tr]), ent[te], p)
            scores |= {"приор ACDC LOCO": loco_p, "энтропия+приор ACDC LOCO": loco_c}
        rng = np.random.default_rng(0)
        w = rng.multinomial(len(y), np.full(len(y), 1 / len(y)), size=args.n_boot)
        g = ((y == 1) & valid)[valid]
        for b in BUDGETS:
            orc = captured(scores["оракул"], y, valid, b)
            ref = captured(ent, y, valid, b)
            for name, s in scores.items():
                c = captured(s, y, valid, b)
                row = {"тест": spec, "оценка": name, "бюджет": b, "доля_оракула": c.sum() / orc.sum()}
                if b == 0.20:
                    diff = (w @ c - w @ ref) / (w @ orc)
                    row |= {"Δ_энтропия": row["доля_оракула"] - ref.sum() / orc.sum(),
                            "ci_lo": np.percentile(diff, 2.5), "ci_hi": np.percentile(diff, 97.5),
                            "AUROC_G": roc_auc_score(g, np.asarray(s)[valid]) if name != "оракул" else 1.0}
                rows.append(row)
        # Профиль: доля пользы и доля размеченных пикселей по полосам высоты и по площади компоненты B0.
        h = y.shape[1]
        band = np.broadcast_to((np.arange(h) * 8 // h)[None, :, None], y.shape)
        bins = np.digitize(carea, [256, 1024, 4096, 16384, 65536])
        net = (y * valid).sum()
        for kind, lab, names in (("высота", band, [f"{k}/8 сверху" for k in range(8)]),
                                 ("площадь компоненты B0", bins,
                                  ["<256", "256–1k", "1k–4k", "4k–16k", "16k–64k", "≥64k"])):
            for k, nm in enumerate(names):
                m = (lab == k) & valid
                prof.append({"тест": spec, "срез": kind, "значение": nm, "доля_пикселей": m.sum() / valid.sum(),
                             "доля_пользы": (y * m).sum() / net,
                             "польза_на_пиксель": (y * m).sum() / max(m.sum(), 1)})

    table, prof = pd.DataFrame(rows), pd.DataFrame(prof)
    out = RESULTS / "summary"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "spatial_prior.csv", index=False)
    prof.to_csv(out / "spatial_prior_profile.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    t = table.pivot_table(index=["тест", "оценка"], columns="бюджет", values="доля_оракула", sort=False)
    print("\nДоля пользы оракула по бюджету пикселей кадра:")
    print(t.round(3).to_string())
    print("\nПри 20%: разница с энтропией [95% bootstrap по кадрам], AUROC «B2 исправляет пиксель»:")
    print(table[table.бюджет == 0.20][["тест", "оценка", "Δ_энтропия", "ci_lo", "ci_hi", "AUROC_G"]]
          .round(3).to_string(index=False))
    print("\nПрофиль пользы:")
    print(prof.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
