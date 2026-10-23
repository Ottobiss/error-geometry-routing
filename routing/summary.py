"""Сводная таблица: все методы из сохранённых запусков на всех тестовых популяциях — сдвиг домена.

    python -m routing.summary                                   # CS P1 и ACDC (каждое условие и all)
    python -m routing.summary --tests cityscapes:p1,acdc:night

Методы — конфигурации (method, params) запусков results/runs/cityscapes_p1/*/config.json плюс
опорные random / entropy / oracle и p- / MetaSeg-поля → Δ. Каждый метод подгоняется один раз на
Cityscapes P0 (как в routing.run) и оценивается на каждой популяции тем же core.evaluation.evaluate_routing:
bootstrap по кадрам популяции против entropy и против regr_entropy
(регрессор Δ на одной энтропии — контроль для сочетаний «энтропия + геометрия»).
Выход: results/summary/routing.csv — строка на (популяция, метод); на экран — Δ к entropy@20% по популяциям.
"""
import argparse
import json

import pandas as pd

from core.data import FIT_DATASET, FIT_POP, load_units
from core.evaluation import evaluate_routing, routing_cache
from core.paths import RESULTS, run_dir, write_json
from routing.methods import METHODS
from routing.run import FIELDS, REFERENCE

TESTS = "cityscapes:p1,acdc:fog,acdc:night,acdc:rain,acdc:snow,acdc:all"
CONTROL = "regr_entropy"


def configs():
    """{имя строки: (method, params)} — опорные строки и все запуски CS P1 с методом роутера."""
    out = {m: (m, {}) for m in REFERENCE + FIELDS}
    seen = {(m, "{}") for m in out}
    for cfg_path in sorted(run_dir("", "cityscapes:p1").glob("*/config.json")):
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        key = (cfg.get("method"), json.dumps(cfg.get("params", {}), sort_keys=True))
        if key[0] is None or key in seen:             # не запуск роутера или повтор опорной строки
            continue
        seen.add(key)
        out[cfg_path.parent.name] = (cfg["method"], cfg["params"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default=TESTS)
    args = ap.parse_args()

    train = load_units(FIT_DATASET, FIT_POP, labeled_only=True)
    methods = {name: METHODS[m](**p).fit(train) for name, (m, p) in configs().items()}
    rows = []
    for test_spec in args.tests.split(","):
        dataset, pop = test_spec.split(":")
        test = load_units(dataset, pop)
        scores = {name: m.score(test) for name, m in methods.items()}
        table, base = evaluate_routing(test.table, scores, routing_cache(dataset, pop),
                                       references=["entropy", CONTROL])
        table.insert(0, "test", test_spec)
        table["miou_b0"], table["miou_b2"] = base["B0"], base["B2"]
        rows.append(table)
        print(f"{test_spec}: {len(test.table)} компонент, B0 {base['B0']:.2f}, B2 {base['B2']:.2f}")

    out = pd.concat(rows, ignore_index=True)
    out_dir = RESULTS / "summary"
    out_dir.mkdir(exist_ok=True)
    out.to_csv(out_dir / "routing.csv", index=False)
    write_json(out_dir / "config.json", {"script": "routing.summary", "tests": args.tests,
                                         "methods": {k: list(v) for k, v in configs().items()}})

    pd.set_option("display.width", 300)
    pd.set_option("display.max_rows", 200)
    cell = out.apply(lambda r: f"{r['Δentropy@20%']:+.2f} [{r['ci_lo@20%']:+.2f},{r['ci_hi@20%']:+.2f}]",
                     axis=1)
    print("\nΔ mIoU к entropy при 20% пикселей [95% bootstrap]:")
    print(out.assign(cell=cell).pivot(index="метод", columns="test", values="cell")
          [args.tests.split(",")].to_string())


if __name__ == "__main__":
    main()
