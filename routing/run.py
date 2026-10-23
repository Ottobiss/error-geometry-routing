"""Запуск метода роутера: fit на Cityscapes P0, оценка на тестовой популяции, сохранение результатов.

    python -m routing.run --method entropy --fields                    # опорная таблица
    python -m routing.run --method md_output -p variant=softmax -p eps=1e-4 --name md_softmax_e1e-4
    python -m routing.run --method regr -p fields=mean_entropy,ddu:logp_mean --name regr_entropy_ddu \\
        --compare regr_entropy --reference regr_entropy                  # сочетание против контроля

Параметры метода: -p ключ=значение (повторяемо), значение — int, float, true/false или строка.
Протокол (одинаковый для всех экспериментов):
  - fit на компонентах Cityscapes P0 с меткой, оценка на всех компонентах тестовой популяции;
  - core.evaluation.evaluate_routing: mIoU итоговой маски и доля выигрыша оракула при 5/10/20/30/50%,
    Spearman с Δ/пиксель, парный bootstrap по кадрам (2000, сид 0) разницы с entropy при 10/20%;  - опорные строки: random, entropy, oracle, mIoU только B0 / только B2;
    --fields добавляет p- и MetaSeg-поля → Δ (обязательно в первой таблице серии);Результаты: results/runs/<датасет>_<популяция>/<name>/ — scores.parquet, metrics.json, config.json.
"""
import argparse
from datetime import date

import pandas as pd

from core.data import FIT_DATASET, FIT_POP, load_units
from core.evaluation import evaluate_routing, routing_cache
from core.paths import git_commit, parse_params, run_dir, write_json
from routing.methods import METHODS

REFERENCE = ["random", "entropy", "oracle"]
FIELDS = ["p_fields_delta", "metaseg_fields_delta"]


def journal_row(name, method, params, r, commit, extra_refs=()):
    """Строка EXPERIMENTS.md в фиксированных столбцах; вывод дописывается вручную.
    Δ к дополнительным опорным строкам (--reference) — в начале столбца «Вывод»."""
    extra = "; ".join(
        f"Δ к {ref}: @20% {r[f'Δ{ref}@20%']:+.2f} [{r[f'ci_lo[{ref}]@20%']:+.2f}, "
        f"{r[f'ci_hi[{ref}]@20%']:+.2f}], @10% {r[f'Δ{ref}@10%']:+.2f} "
        f"[{r[f'ci_lo[{ref}]@10%']:+.2f}, {r[f'ci_hi[{ref}]@10%']:+.2f}]" for ref in extra_refs)
    return (f"| {date.today()} | {name} | {method} {params or ''} | {r['mIoU@10%']:.2f} | "
            f"{r['mIoU@20%']:.2f} | {r['выигр@20%']:.3f} | {r['spearman']:.3f} | "
            f"{r['Δentropy@20%']:+.2f} [{r['ci_lo@20%']:+.2f}, {r['ci_hi@20%']:+.2f}] | {extra} | {commit} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(METHODS))
    ap.add_argument("-p", "--param", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--name", default=None, help="имя запуска (папка в results/runs)")
    ap.add_argument("--fields", action="store_true", help="добавить p- и MetaSeg-поля → Δ")
    ap.add_argument("--compare", default="", help="запуски через запятую: их scores.parquet "
                    "добавляются строками в ту же таблицу")
    ap.add_argument("--reference", default="", help="дополнительные опорные строки bootstrap через "
                    "запятую (имена строк таблицы, напр. regr_entropy из --compare); entropy — всегда")
    ap.add_argument("--test", default="cityscapes:p1", help="тестовая популяция <датасет>:<популяция>")
    args = ap.parse_args()

    params = parse_params(args.param)
    name = args.name or "__".join([args.method] + [f"{k}={v}" for k, v in sorted(params.items())])
    dataset, pop = args.test.split(":")
    train = load_units(FIT_DATASET, FIT_POP, labeled_only=True)
    test = load_units(dataset, pop)

    method = METHODS[args.method](**params)
    scores = {args.method: method.fit(train).score(test)}
    for ref in REFERENCE + (FIELDS if args.fields else []):
        if ref != args.method:
            scores[ref] = METHODS[ref]().fit(train).score(test)
    for exp in filter(None, args.compare.split(",")):
        prev = pd.read_parquet(run_dir(exp, args.test) / "scores.parquet")
        assert prev[["image_id", "comp_id"]].equals(test.table[["image_id", "comp_id"]]), exp
        scores[exp] = prev.score.values

    extra_refs = [r for r in args.reference.split(",") if r]
    missing = [r for r in extra_refs if r not in scores]
    assert not missing, f"опорных строк нет в таблице: {missing}"
    table, base = evaluate_routing(test.table, scores, routing_cache(dataset, pop),
                                   references=["entropy"] + extra_refs)
    out = run_dir(name, args.test)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"image_id": test.table.image_id, "comp_id": test.table.comp_id,
                  "score": scores[args.method]}).to_parquet(out / "scores.parquet")
    write_json(out / "config.json", {"method": args.method, "params": params, "fields": args.fields,
                                     "compare": args.compare, "reference": args.reference,
                                     "test": args.test})
    write_json(out / "metrics.json", {"miou_b0": base["B0"], "miou_b2": base["B2"],
                                      "rows": table.to_dict("records")},
               stamp=False)
    commit = git_commit()

    pd.set_option("display.width", 300)
    print(f"mIoU на {args.test}: только B0 {base['B0']:.2f}, только B2 {base['B2']:.2f}")
    print(table[["метод", "spearman"] + [c for c in table.columns if c.startswith("mIoU")]]
          .round(2).to_string(index=False))
    print(table[["метод"] + [c for c in table.columns if c.startswith("выигр")]].round(3)
          .to_string(index=False))
    for ref in ["entropy"] + extra_refs:
        tag = "" if ref == "entropy" else f"[{ref}]"
        cols = [f"Δ{ref}@10%", f"ci_lo{tag}@10%", f"ci_hi{tag}@10%",
                f"Δ{ref}@20%", f"ci_lo{tag}@20%", f"ci_hi{tag}@20%"]
        print(f"Разница mIoU с {ref}, парный bootstrap по кадрам (2000), 95%-интервал:")
        print(table[["метод"] + cols].round(2).to_string(index=False))
    print("\nСтрока для EXPERIMENTS.md:\n" + journal_row(name, args.method, params, table.iloc[0],
                                                          commit, extra_refs))


if __name__ == "__main__":
    main()
