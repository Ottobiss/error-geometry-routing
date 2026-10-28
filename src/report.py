"""Report over all stages as Markdown with LaTeX formulas, tables and figures (VS Code and GitHub render it): settings,
training curves and validation metrics, the export, the threshold choice on Cityscapes val and the ACDC test with every
benchmark metric. All metrics are also collected into one metrics.json; the logs and the benchmark panels are copied.

    python -m src.report configs/report.yaml         # -> runs/report/report.md
"""
import argparse
import json
import math
import platform
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from src.config import load_config

NAMES = {"all": "все кадры", "fog": "туман", "night": "ночь", "rain": "дождь", "snow": "снег"}
FINE_TUNED = ("B0_ft", "B2_ft")
TRAIN_KEYS = ("model.variant", "model.pretrained", "model.manifold", "model.c", "model.classifier_scale", "model.patch",
              "loss.t", "loss.hr", "loss.class_weights", "loss.large_unc_weight", "loss.large_unc_loss",
              "data.batch_size", "data.crop", "data.large_uncertainty", "data.large_uncertainty_split", "train.iters",
              "train.lr", "train.weight_decay", "train.warmup", "train.amp", "seed")
BENCHMARK_KEYS = ("small", "large", "root", "split", "thresholds", "budgets", "patch", "top", "cost_ratio",
                  "bootstrap", "amp")

SCHEME = """## Схема эксперимента

1. **Стадия 1.** SegFormer-B2 с весами NVIDIA для Cityscapes: учится только последний 1×1 conv, лосс HyperUL, кадры \
train Cityscapes.
2. **Стадия 2.** Обученная B2 один раз считает свою неопределённость для каждого патча кадров val, карты сохраняются.
3. **Стадия 3.** SegFormer-B0 так же учит сегментацию на train, а отдельным выходом — неопределённость B2 на val.
4. **Стадия 4.** На val Cityscapes выбирается порог маршрутизации τ.
5. **Стадия 5.** Тест на всех размеченных кадрах ACDC (train и val) по условиям: туман, ночь, дождь, снег. Здесь \
ничего не подбирается."""

FORMULAS = r"""## Формулы

**Гиперболическое вложение.** Выход последнего 1×1 conv SegFormer, умноженный на `classifier_scale`, даёт для
пикселя вектор $v \in \mathbb{R}^K$, $K = 19$, по координате на класс. По умолчанию он отображается на гиперболоид
$\langle x, x \rangle_L = -1/c$ (модель Лоренца, `manifold: lorentz`): пространственные координаты точки служат
логитами, а расстояние до начала координат $o = (1/\sqrt{c}, 0, \dots, 0)$ — мерой уверенности:

$$
x = \exp_o^c(v) = \frac{1}{\sqrt{c}} \left(\cosh\left(\sqrt{c}\,\lVert v \rVert\right),\
\sinh\left(\sqrt{c}\,\lVert v \rVert\right) \frac{v}{\lVert v \rVert}\right), \qquad
p_k = \frac{e^{x_k}}{\sum_{j=1}^{K} e^{x_j}}, \qquad d_c(x, o) = \lVert v \rVert,
$$

где $x_1, \dots, x_K$ — пространственные координаты. В шаре Пуанкаре из статьи (`manifold: poincare`)
$x = \tanh(\sqrt{c}\,\lVert v \rVert)\, v / (\sqrt{c}\,\lVert v \rVert)$ и $d = 2 \lVert v \rVert$, но логиты там не
больше $1/\sqrt{c}$, и с замороженной сетью дообучение портит сегментацию. В обеих моделях $d \propto \lVert v \rVert$,
поэтому неопределённость ниже одна и та же.

**Неопределённость (Eq. 15).** $\bar d_P$ — среднее расстояние по патчу $P$ из `patch` × `patch` пикселей, максимум
берётся по патчам кадра:

$$
u_P = 1 - \frac{\bar d_P}{\max_{P'} \bar d_{P'}}.
$$

**Лосс HyperUL (Eq. 10–14).** $S$ — отбор OHUEM: доля `hr` размеченных пикселей каждого кадра с наибольшей
неопределённостью; $w_k$ — веса классов:

$$
\mathcal{L}_{\mathrm{HyperUL}} = \frac{1}{|S|} \sum_{i \in S}
\frac{-w_{y_i} \log p_{i, y_i}}{\ln\left(t + 1 - u_{P(i)}\right)}.
$$

**Прогноз неопределённости B2.** Дополнительный выход B0 $z$ усредняется по патчу, $\hat u_P = \sigma(\bar z_P)$. Он
учится на пикселях $V$ кадров val по картам $u^{B2}$ стадии 2, $\lambda$ = `large_unc_weight`:

$$
\mathcal{L} = \mathcal{L}_{\mathrm{HyperUL}}
+ \frac{\lambda}{|V|} \sum_{i \in V} \mathrm{BCE}\left(\hat u_{P(i)}, u^{B2}_i\right).
$$

**Маршрутизация.** Выигрыш патча $g_P$ и оценка кадра $s$ — среднее по $k = \max(1, \operatorname{round}(qN))$
патчам с наибольшим выигрышем из $N$ патчей кадра, $q$ = `top`. Если $s > \tau$, весь кадр обрабатывает B2:

$$
g_P = u^{B0}_P - \hat u_P, \qquad s = \frac{1}{k} \sum_{P \in \operatorname{top}_k(g)} g_P.
$$

В адаптивном инференсе и в решениях по патчам вместо $g$ берутся и другие сигналы: `own` — $u^{B0}$,
`entropy` — $H = -\sum_k p_k \log p_k$.

**Качество сегментации.** $K'$ — классы, которые есть в разметке или в предсказании:

$$
\mathrm{IoU}_k = \frac{TP_k}{TP_k + FP_k + FN_k}, \qquad \mathrm{mIoU} = \frac{1}{|K'|} \sum_{k \in K'} \mathrm{IoU}_k.
$$

Доверительные интервалы — перцентили 2.5 и 97.5 по бутстреп-выборкам кадров; p-значение разности по бутстрепу
$p = \min\left(1, 2 \min\left(P^*(\Delta \le 0), P^*(\Delta \ge 0)\right)\right)$; критерий Уилкоксона — по
разностям попиксельной точности кадров.

**Калибровка и прогноз.** Уверенность $1 - u$ делится на 15 равных корзин $B_b$; $N$ — размеченные пиксели:

$$
\mathrm{ECE} = \sum_{b=1}^{15} \frac{|B_b|}{N} \left| \mathrm{acc}(B_b) - \mathrm{conf}(B_b) \right|, \qquad
\mathrm{MAE} = \frac{1}{N} \sum_{i} \left| \hat u_{P(i)} - u^{B2}_{P(i)} \right|.
$$

**Маршрутизация и адаптивный инференс.** $\rho$ — доля кадров у B2, $r$ = `cost_ratio`; случайный выбор — ожидаемая
смесь матриц ошибок $(1 - \rho)\, C_{B0} + \rho\, C_{B2}$; оракул отдаёт кадры, где B2 права на большем числе пикселей:

$$
\mathrm{cost} = 1 + r \rho, \qquad
\mathrm{gap} = \frac{\mathrm{mIoU} - \mathrm{mIoU}_{B0}}{\mathrm{mIoU}_{B2} - \mathrm{mIoU}_{B0}}, \qquad
\mathrm{eff} = \frac{\mathrm{mIoU} - \mathrm{mIoU}_{\mathrm{rand}}}
{\mathrm{mIoU}_{\mathrm{oracle}} - \mathrm{mIoU}_{\mathrm{rand}}}.
$$

$F$ — пиксели, где B0 ошибается, а B2 права; $L$ — размеченные пиксели кадров, отданных B2:

$$
\mathrm{fix\ precision} = \frac{|F \cap L|}{|L|}, \qquad \mathrm{fix\ recall} = \frac{|F \cap L|}{|F|}.
$$

**Патчи.** Патчу нужна B2, если B2 права на большем числе его размеченных пикселей, чем B0. $e_P$ — доля неверных
размеченных пикселей B0 в патче; $y_{50} = [e_P \ge 0.5]$, $y_{\mathrm{any}} = [e_P \ge 0.1]$; recall@$b$ — доля всех
неверных пикселей B0 в доле $b$ самых неуверенных патчей:

$$
\mathrm{AUROC} = P\left(s^+ > s^-\right) + \tfrac{1}{2} P\left(s^+ = s^-\right), \qquad
\mathrm{AP} = \frac{1}{n_+} \sum_{j:\, y_{(j)} = 1} \mathrm{Precision@}j.
$$

AUROC $y_{50}$ в децилях энтропии — среднее AUROC внутри 10 корзин по средней энтропии патча, где есть оба класса."""


def load(path):
    """JSON -> its data; JSON lines -> rows, the last one of every step; None if the file is missing."""
    path = Path(path)
    if not path.is_file():
        return None
    if path.suffix != ".jsonl":
        return json.loads(path.read_text())
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row.get("step", len(rows))] = row
    return [rows[key] for key in sorted(rows)]


def fmt(x, digits=3, scale=1, sign=False):
    """A number for a table, a dash if it is missing or NaN."""
    if x is None or isinstance(x, float) and math.isnan(x):
        return "—"
    text = f"{x * scale:{'+' if sign else ''}.{digits}f}"
    return text.lstrip("+-") if float(text) == 0 else text  # no signed zeros


def pct(x, digits=1, sign=False):
    return fmt(x, digits, 100, sign)


def pvalue(x):
    return fmt(x) if x is None or math.isnan(x) or x >= 0.001 else "< 0.001"


def name(group):
    return NAMES.get(group, group)


def shown(cell):
    """Length of a table cell as displayed: without link targets and formatting characters."""
    return len(re.sub(r"\]\([^)]*\)|[`*\[]", "", cell))


def table(header, rows):
    """A Markdown table; the dashes under each header follow the column width, which pandoc uses for wide tables."""
    rows = [[str(cell) for cell in row] for row in rows]
    widths = [max(3, min(30, max(shown(row[k]) for row in [header, *rows]))) for k in range(len(header))]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


def details(summary, body):
    return f"<details><summary>{summary}</summary>\n\n{body}\n\n</details>"


def figure(fig, out, file, caption):
    """Saves a figure to out/figures/<file>.png -> Markdown with the image and its caption. No alt text: pandoc would
    make the image a floating figure, which leaves its section in a PDF."""
    fig.savefig(out / "figures" / f"{file}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f"![](figures/{file}.png)\n\n*{caption}*"


def git(*args):
    """Git of the repository with this code."""
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def header(cfg):
    commit, changes = git("rev-parse", "--short", "HEAD"), git("status", "--porcelain")
    version = (f"коммит `{commit}`" + (" с незакоммиченными изменениями" if changes else "")) if commit else "вне git"
    return ["# Отчёт: маршрутизация SegFormer B0 → B2 по гиперболической неопределённости",
            f"Сгенерирован {datetime.now():%Y-%m-%d %H:%M} на `{platform.node()}`, {version}. Источники: "
            f"`{cfg.large}`, `{cfg.export}`, `{cfg.small}`, `{cfg.val}`, `{cfg.test}`. Все метрики собраны в "
            "[metrics.json](metrics.json), логи — в [logs](logs)."]


def value(v):
    """A config value as plain text that can wrap: lists comma-separated, null as a dash."""
    if OmegaConf.is_list(v):
        return ", ".join(map(str, v))
    return "—" if v is None else str(v)


def settings(runs, keys):
    """{title: run dir with config.yaml} -> table of the main keys and the full configs."""
    configs = {t: OmegaConf.load(Path(d, "config.yaml")) for t, d in runs.items() if Path(d, "config.yaml").is_file()}
    if not configs:
        return ["Конфигов нет: " + ", ".join(f"`{d}`" for d in runs.values()) + "."]
    rows = [[key, *(value(OmegaConf.select(c, key)) for c in configs.values())] for key in keys]
    return [table(["ключ", *configs], rows),
            *(details(f"Полный конфиг: {t}", f"```yaml\n{OmegaConf.to_yaml(c)}```") for t, c in configs.items())]


def training(title, run, out, file):
    """Validation metrics, speed and curves of a src.train run."""
    vals, steps = load(Path(run, "metrics.jsonl")), load(Path(run, "train.jsonl"))
    if not vals and not steps:
        return [f"Нет `metrics.jsonl` и `train.jsonl` в `{run}`."]
    parts = []
    if vals:
        best, last = max(vals, key=lambda m: m["val/mIoU"]), vals[-1]
        keys = [("val/mIoU", "mIoU на val, %", lambda x: pct(x, 2)), ("val/hyperbolic_ECE", "ECE", lambda x: fmt(x, 4)),
                ("val/hyperbolic_uncertainty_correct", "неопределённость на верных пикселях", fmt),
                ("val/hyperbolic_uncertainty_wrong", "неопределённость на ошибочных пикселях", fmt)]
        parts.append(table(["метрика", f"best.pt, шаг {best['step']}", f"последняя валидация, шаг {last['step']}"],
                           [[label, f(best.get(key)), f(last.get(key))] for key, label, f in keys]))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    axes = axes.flatten()
    if steps:
        speed = np.mean([s["train/img_per_s"] for s in steps])
        memory = max((s["train/mem_gb"] for s in steps if "train/mem_gb" in s), default=None)
        parts.append(f"Итераций: {steps[-1]['step']}, средняя скорость {speed:.0f} кадр/с"
                     + (f", пиковая память GPU {memory:.1f} ГБ" if memory is not None else "")
                     + f", от первой до последней записи {(steps[-1]['time'] - steps[0]['time']) / 3600:.1f} ч.")
        x = [s["step"] for s in steps]
        axes[0].plot(x, [s["train/loss"] for s in steps], lw=1)
        axes[0].set(title="лосс обучения", xlabel="итерация")
        if "train/large_unc_mae" in steps[-1]:
            twin = axes[0].twinx()
            twin.plot(x, [s.get("train/large_unc_mae") for s in steps], lw=1, color="tab:orange")
            twin.set_ylabel("MAE прогноза u(B2)", color="tab:orange")
    if vals:
        x = [m["step"] for m in vals]
        axes[1].plot(x, [100 * m["val/mIoU"] for m in vals], "o-")
        axes[2].plot(x, [m["val/hyperbolic_ECE"] for m in vals], "o-")
        for key, label in (("correct", "верные"), ("wrong", "ошибочные")):
            axes[3].plot(x, [m[f"val/hyperbolic_uncertainty_{key}"] for m in vals], "o-", label=label)
        axes[3].legend()
    for ax, label in zip(axes[1:], ("mIoU на val, %", "ECE на val", "средняя неопределённость на val")):
        ax.set(title=label, xlabel="итерация")
    return [*parts, figure(fig, out, file, f"{title}: обучение и валидация")]


def export(path, out):
    summary = load(Path(path, "summary.json"))
    if summary is None:
        return [f"Нет `summary.json` в `{path}`."]
    counts = np.array(summary["histogram"], dtype=float)
    edges = np.linspace(0, 1, len(counts) + 1)
    fig, ax = plt.subplots(figsize=(6, 3), layout="constrained")
    ax.bar(edges[:-1], 100 * counts / max(counts.sum(), 1), width=np.diff(edges), align="edge", edgecolor="black")
    ax.set(xlabel="неопределённость B2", ylabel="пикселей карт, %")
    return [f"Неопределённость B2 посчитана на кадрах `{summary['split']}` (n = {summary['images']}), средняя "
            f"{summary['mean']:.3f}. Эти карты — цель, по которой B0 учится предсказывать неопределённость B2.",
            figure(fig, out, "export", "Распределение экспортированной неопределённости B2")]


def choose(bench, max_launch):
    """The routing row with the best mIoU among the thresholds that give B2 at most max_launch of the images."""
    rows = [r for r in bench["all"]["routing"]["thresholds"] if r["launch_rate"] <= max_launch]
    return max(rows, key=lambda r: r["mIoU"]) if rows else None


def models_table(bench):
    """mIoU with the 95% bootstrap interval: groups in rows, models in columns."""
    models = list(bench["all"]["models"])

    def cell(m):
        return f"{pct(m['mIoU'], 2)} ({pct(m['ci'][0])}–{pct(m['ci'][1])})"

    rows = [[f"{name(g)}, n = {bench[g]['images']}", *(cell(bench[g]["models"][m]) for m in models)] for g in bench]
    return table(["группа", *models], rows)


def accuracy_table(bench):
    models = list(bench["all"]["models"])
    return table(["группа", *models],
                 [[name(g), *(pct(bench[g]["models"][m]["pixel_accuracy"], 2) for m in models)] for g in bench])


def differences_table(bench):
    rows = []
    for pair in bench["all"]["differences"]:
        for g in bench:
            d = bench[g]["differences"][pair]
            rows.append([pair.replace("-", " − ") if g == "all" else "", name(g), pct(d["mIoU"], 2, True),
                         f"[{pct(d['ci'][0], 2, True)}, {pct(d['ci'][1], 2, True)}]", pvalue(d["p_bootstrap"]),
                         pvalue(d["p_wilcoxon"])])
    return table(["разность", "группа", "ΔmIoU, п. п.", "95% ДИ", "p, бутстреп", "p, Уилкоксон"], rows)


def classes_table(result):
    models = result["models"]
    classes = list(next(iter(models.values()))["iou"])
    return table(["класс", *models], [[c, *(pct(m["iou"][c]) for m in models.values())] for c in classes])


def uncertainty_tables(bench):
    """Calibration and the prediction error of B2's uncertainty; mean uncertainty on correct and wrong pixels."""
    u = {g: bench[g]["uncertainty"] for g in bench}
    calibration = table(["группа", *(f"ECE {m}" for m in FINE_TUNED), "MAE прогноза u(B2)"],
                        [[name(g), *(fmt(u[g][m]["ECE"], 4) for m in FINE_TUNED), fmt(u[g]["large_unc_mae"], 4)]
                         for g in bench])
    keys = (("uncertainty_correct", "верные"), ("uncertainty_wrong", "ошибки"))
    means = table(["группа", *(f"{m}, {label}" for m in FINE_TUNED for _, label in keys)],
                  [[name(g), *(fmt(u[g][m][key]) for m in FINE_TUNED for key, _ in keys)] for g in bench])
    return ["ECE гиперболической неопределённости и ошибка, с которой B0 предсказывает неопределённость B2:",
            calibration, "Средняя неопределённость на верных и ошибочных пикселях; на ошибочных она должна быть "
            "заметно больше:", means]


def reliability(result, out, file, caption):
    fig, axes = plt.subplots(1, len(FINE_TUNED), figsize=(12, 4.2), layout="constrained")
    for ax, model in zip(axes, FINE_TUNED):
        u = result["uncertainty"][model]
        if "reliability" in u:
            pixels = np.array(u["reliability"]["pixels"], dtype=float)
            accuracy = np.array(u["reliability"]["accuracy"], dtype=float)
            centers, filled = (np.arange(len(pixels)) + 0.5) / len(pixels), pixels > 0
            ax.bar(centers[filled], accuracy[filled], width=1 / len(pixels), edgecolor="black", alpha=0.8,
                   label="доля верных пикселей")
            ax.plot([0, 1], [0, 1], "--", color="gray", label="идеальная калибровка")
            twin = ax.twinx()
            twin.step(centers, 100 * pixels / max(pixels.sum(), 1), where="mid", color="tab:orange")
            twin.set_ylabel("пикселей в корзине, %", color="tab:orange")
        ax.set(title=f"{model}: ECE {fmt(u['ECE'], 4)}", xlabel="уверенность 1 − u", ylabel="доля верных пикселей",
               xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(loc="upper left")
    return figure(fig, out, file, caption)


ROUTING = ("Столбцы: доля кадров, отданных B2; mIoU маршрутизации; закрытая доля разрыва между B0_ft и B2_ft; "
           "стоимость в единицах B0; точность и полнота исправлений — пикселей, где B0 ошибалась, а B2 права.")


def routing_table(result, tau=None):
    def threshold(t):  # the chosen one in bold
        return f"**{t:g}**" if tau is not None and math.isclose(t, tau) else f"{t:g}"

    rows = [[threshold(r["threshold"]), pct(r["launch_rate"]), pct(r["mIoU"], 2), fmt(r["gap_closed"], 2),
             fmt(r["cost"], 2), pct(r["fix_precision"]), pct(r["fix_recall"])] for r in result["routing"]["thresholds"]]
    return table(["порог τ", "кадров B2, %", "mIoU, %", "разрыв", "стоимость", "точность, %", "полнота, %"], rows)


def routing_at(bench, tau):
    rows = []
    for g in bench:
        r = next((r for r in bench[g]["routing"]["thresholds"] if math.isclose(r["threshold"], tau)), None)
        if r is not None:
            rows.append([name(g), pct(r["launch_rate"]), pct(r["mIoU"], 2), fmt(r["gap_closed"], 2),
                         fmt(r["cost"], 2), pct(r["fix_precision"]), pct(r["fix_recall"])])
    if not rows:
        return f"Порога {tau:g} нет среди порогов теста: бенчмарки запускались с разными `thresholds`."
    return table(["группа", "кадров B2, %", "mIoU, %", "разрыв", "стоимость", "точность, %", "полнота, %"], rows)


def routing_figure(bench, tau, out, file, caption):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    for g in bench:
        rows = bench[g]["routing"]["thresholds"]
        x = [100 * r["launch_rate"] for r in rows]
        axes[0].plot(x, [100 * r["mIoU"] for r in rows], "o-", ms=3, label=name(g))
        axes[1].plot(x, [r["gap_closed"] for r in rows], "o-", ms=3, label=name(g))
        for r in rows:
            if tau is not None and math.isclose(r["threshold"], tau):
                axes[0].plot(100 * r["launch_rate"], 100 * r["mIoU"], "k*", ms=10)
                axes[1].plot(100 * r["launch_rate"], r["gap_closed"], "k*", ms=10)
    axes[0].set(title="mIoU маршрутизации", xlabel="кадров у B2, %", ylabel="mIoU, %")
    axes[1].set(title="закрытый разрыв", xlabel="кадров у B2, %", ylabel="(mIoU − B0) / (B2 − B0)")
    axes[0].legend()
    return figure(fig, out, file, caption + ("; звёздочка — порог τ" if tau is not None else ""))


def adaptive_figure(bench, out):
    cols = min(3, len(bench))
    rows = math.ceil(len(bench) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), squeeze=False, layout="constrained")
    for ax, g in zip(axes.flatten(), bench):
        for policy, curve in bench[g]["adaptive"].items():
            ax.plot([100 * r["launch_rate"] for r in curve], [100 * r["mIoU"] for r in curve],
                    "--" if policy in ("random", "oracle") else "o-", ms=3, label=policy)
        ax.set(title=name(g), xlabel="кадров у B2, %", ylabel="mIoU, %")
    for ax in axes.flatten()[len(bench):]:
        ax.set_visible(False)
    axes[0][0].legend()
    return figure(fig, out, "adaptive", "Адаптивный инференс: B2 получает долю кадров с наибольшей оценкой")


def adaptive_table(result):
    curves = result["adaptive"]
    budgets = [r["budget"] for r in next(iter(curves.values()))]
    keep = [k for k, b in enumerate(budgets) if 0 < b < 1]
    return ["mIoU на всех кадрах, %, когда B2 получает указанную долю кадров:",
            table(["политика", *(f"{100 * budgets[k]:g}%" for k in keep)],
                  [[policy, *(pct(rows[k]["mIoU"], 2) for k in keep)] for policy, rows in curves.items()])]


def frame_auroc_table(bench):
    signals = list(bench["all"]["frame_auroc"])
    return ["AUROC оценки кадра для события «B2 права на большем числе пикселей кадра, чем B0»:",
            table(["группа", *signals], [[name(g), *(fmt(bench[g]["frame_auroc"][s]) for s in signals)]
                                         for g in bench])]


def patches_table(bench):
    signals = [s for s in bench["all"]["patches"] if s not in ("patches", "base_rate")]
    rows = [[name(g), p["patches"], pct(p["base_rate"]),
             *(f"{fmt(p[s]['auroc'])} / {fmt(p[s]['ap'])}" for s in signals)]
            for g, p in ((g, bench[g]["patches"]) for g in bench)]
    return ["AUROC / AP сигналов для события «патчу нужна B2»; AP случайного сигнала равна доле таких патчей:",
            table(["группа", "патчей", "нужна B2, %", *signals], rows)]


def decisions_table(result):
    p = result["patches"]
    signals = [s for s in p if s not in ("patches", "base_rate")]

    def cell(row):
        return " / ".join(pct(row[key]) for key in ("precision", "recall", "f1"))

    rows = [[f"{100 * row['budget']:g}", *(cell(p[s]["budgets"][k]) for s in signals)]
            for k, row in enumerate(p[signals[0]]["budgets"]) if 0 < row["budget"] < 1]
    return ["Точность / полнота / F1 решения «нужна B2», %, когда отмечена доля самых неуверенных патчей, все кадры:",
            table(["отмечено, %", *signals], rows)]


def errors_tables(result):
    e = result["errors"]
    signals = [s for s in e if s != "rate"]
    budgets = [(k, r["budget"]) for k, r in enumerate(e[signals[0]]["budgets"]) if 0 < r["budget"] <= 0.3]
    recall = table(["сигнал", *(f"{100 * b:g}%" for _, b in budgets)],
                   [[s, *(pct(e[s]["budgets"][k]["recall"]) for k, _ in budgets)] for s in signals])
    ranking = table(["сигнал", "AUROC y50", "AUROC y_any", "AP y50", "AP y_any", "AUROC y50 в децилях"],
                    [[s, fmt(e[s]["auroc"]["y50"]), fmt(e[s]["auroc"]["y_any"]), fmt(e[s]["ap"]["y50"]),
                      fmt(e[s]["ap"]["y_any"]), fmt(e[s]["auroc_y50_entropy_deciles"])] for s in signals])
    return ["Полнота по ошибкам, все кадры: доля неверных пикселей B0, %, в отмеченной доле самых неуверенных патчей:",
            recall, f"Ранжирование патчей с ошибками, все кадры. Положительных патчей: y50 — {pct(e['rate']['y50'])}%, "
            f"y_any — {pct(e['rate']['y_any'])}%; такая же AP у случайного сигнала:", ranking]


def errors_groups(bench):
    signals = [s for s in bench["all"]["errors"] if s != "rate"]
    budgets = [r["budget"] for r in bench["all"]["errors"][signals[0]]["budgets"]]
    k = next((k for k, b in enumerate(budgets) if math.isclose(b, 0.1)), None)

    def cell(e, s):
        return f"{pct(e[s]['budgets'][k]['recall']) if k is not None else '—'} / {fmt(e[s]['auroc']['y50'])}"

    rows = [[name(g), pct(e["rate"]["y50"]), pct(e["rate"]["y_any"]), *(cell(e, s) for s in signals)]
            for g, e in ((g, bench[g]["errors"]) for g in bench)]
    return ["По условиям: доли положительных патчей, %, и для каждого сигнала recall@10%, % / AUROC y50:",
            table(["группа", "y50, %", "y_any, %", *signals], rows)]


def errors_figure(result, out):
    e = result["errors"]
    fig, ax = plt.subplots(figsize=(6, 4.2), layout="constrained")
    for s in (s for s in e if s != "rate"):
        rows = e[s]["budgets"]
        ax.plot([100 * r["budget"] for r in rows], [100 * r["recall"] for r in rows], "o-", ms=3, label=s)
    ax.plot([0, 100], [0, 100], "--", color="gray", label="случайный выбор")
    ax.set(xlabel="отмечено патчей, %", ylabel="неверных пикселей B0 в них, %")
    ax.legend()
    return figure(fig, out, "error_recall", "Полнота по ошибкам B0 на всех кадрах ACDC")


def panels(path, out):
    files = sorted(Path(path, "panels").glob("*.jpg"))
    if not files:
        return ["Панелей нет."]
    (out / "panels").mkdir(exist_ok=True)
    parts = ["Верхний ряд: кадр, разметка, предсказания B0_ft и B2_ft. Нижний: неопределённость B0_ft, её прогноз "
             "неопределённости B2, неопределённость B2_ft и выигрыш g (чёрный — 0, белый — 1)."]
    for file in files:
        shutil.copyfile(file, out / "panels" / file.name)
        folder, index = file.stem.rsplit("_", 1)
        parts.append(f"**{name(folder)}, пример {int(index) + 1}**\n\n![](panels/{file.name})")
    return parts


def threshold_section(val, chosen, cfg, out):
    if val is None:
        return [f"Нет `{Path(cfg.val, 'benchmark.json')}`."]
    tau, limit = chosen["threshold"] if chosen else None, pct(cfg.max_launch_rate, 0)
    choice = (f"**Выбран порог τ = {tau:g}**: лучшая mIoU маршрутизации, {pct(chosen['mIoU'], 2)}%, среди порогов, "
              f"при которых B2 получает не больше {limit}% кадров (здесь {pct(chosen['launch_rate'])}%)."
              if chosen else f"Ни один порог не отдаёт B2 не больше {limit}% кадров, порог не выбран.")
    return [f"Бенчмарк на кадрах val, n = {val['all']['images']}. {choice} Прогноз неопределённости B2 учился на "
            "этих же кадрах, поэтому маршрутизация здесь выглядит немного лучше, чем на новых данных.", ROUTING,
            routing_table(val["all"], tau),
            routing_figure(val, tau, out, "routing_val", "Маршрутизация на val Cityscapes"),
            "### Модели на val", "mIoU, %, в скобках 95% бутстреп-интервал:", models_table(val),
            "### Неопределённость на val", *uncertainty_tables(val)]


def test_section(test, chosen, cfg, out):
    if test is None:
        return [f"Нет `{Path(cfg.test, 'benchmark.json')}`."]
    result, tau = test["all"], chosen["threshold"] if chosen else None
    parts = ["### Модели", "B0 и B2 — исходные веса NVIDIA, `_ft` — после дообучения. mIoU, %, в скобках 95% "
             "бутстреп-интервал:", models_table(test), "Попиксельная точность, %:", accuracy_table(test),
             "Парные разности mIoU, процентные пункты:", differences_table(test),
             "### IoU по классам, %", classes_table(result),
             *(details(f"IoU по классам: {name(g)}", classes_table(test[g])) for g in test if g != "all"),
             "### Неопределённость дообученных моделей", *uncertainty_tables(test),
             reliability(result, out, "reliability", "Диаграммы надёжности на всех кадрах ACDC"),
             "### Маршрутизация по порогу", ROUTING]
    if tau is None:
        parts.append("Порог на val не выбран, ниже все пороги.")
    else:
        parts += [f"Порог τ = {tau:g} выбран на val Cityscapes и применён к ACDC без подгонки:", routing_at(test, tau)]
        best = choose(test, cfg.max_launch_rate)
        if best:
            parts.append(f"Для сравнения: лучший порог на самом ACDC при том же ограничении — "
                         f"τ = {best['threshold']:g}, mIoU {pct(best['mIoU'], 2)}% при {pct(best['launch_rate'])}% "
                         "кадров у B2. По нему порог не выбирают, это тест.")
    return [*parts, routing_figure(test, tau, out, "routing_test", "Маршрутизация на ACDC по условиям"),
            details("Все пороги на всех кадрах ACDC", routing_table(result, tau)),
            "### Адаптивный инференс по долям кадров", adaptive_figure(test, out), *adaptive_table(result),
            *frame_auroc_table(test),
            "### Решения по патчам: нужна ли B2", *patches_table(test), *decisions_table(result),
            "### Ошибки B0 по патчам", *errors_tables(result), *errors_groups(test), errors_figure(result, out),
            "### Примеры", *panels(cfg.test, out)]


def sweep(path):
    """The best trials of every stage, one column per searched key (values from the trial's config.yaml)."""
    parts = []
    for file in sorted(Path(path).glob("*/trials.jsonl")):
        trials = load(file)
        ranked = sorted((t for t in trials if t["score"] is not None), key=lambda t: -t["score"])
        keys = list(dict.fromkeys(k for t in trials for k in t["params"]))
        rows = []
        for t in ranked[:10]:
            config = file.parent / f"{t['trial']:03d}" / "config.yaml"
            values = OmegaConf.load(config) if config.is_file() else OmegaConf.create()
            rows.append([t["trial"], fmt(t["score"], 4),
                         *(value(OmegaConf.select(values, k, default=t["params"].get(k))) for k in keys), t["minutes"]])
        parts += [f"**Стадия `{file.parent.name}`**: успешных прогонов {len(ranked)} из {len(trials)}, батч "
                  f"{trials[0]['batch']}; ниже лучшие 10, время в минутах.",
                  table(["прогон", "оценка", *(k.rsplit(".", 1)[-1] for k in keys), "мин"], rows)]
    return parts or [f"Перебора нет в `{path}`."]


def plain(text, limit=160):
    """One line for a code span: whitespace collapsed, no backticks, at most limit characters."""
    text = " ".join(text.split()).replace("`", "'")
    return text if len(text) <= limit else text[:limit - 1] + "…"


def runs(cfg, out):
    """Runs of every stage from its log: start times (resumes marked), warnings, the error that stopped a run; the logs
    are copied. Their metrics and tables are already in the sections above."""
    sources = [("1_large", "стадия 1, обучение B2", Path(cfg.large, "train.log")),
               ("2_export", "стадия 2, экспорт", Path(cfg.export, "export.log")),
               ("3_small", "стадия 3, обучение B0", Path(cfg.small, "train.log")),
               ("4_val", "стадия 4, бенчмарк на val", Path(cfg.val, "benchmark.log")),
               ("5_test", "стадия 5, бенчмарк на ACDC", Path(cfg.test, "benchmark.log"))]
    (out / "logs").mkdir(exist_ok=True)
    rows, errors, warnings = [], [], []
    for slug, title, path in sources:
        if not path.is_file():
            rows.append([title, 0, "—", "—", "—", "нет лога"])
            continue
        shutil.copyfile(path, out / "logs" / f"{slug}.log")
        starts, resumed, warned, failed = [], 0, [], 0
        for run in path.read_text(errors="replace").split("\n=== ")[1:]:  # each run starts with a line of keep_log
            command, _, body = run.partition("\n")
            starts.append(command[:16])
            resumed += "--resume" in command
            warned += re.findall(r"^\S.*?:\d+: (\w*Warning: .*)$", body, re.M)
            if "Traceback (most recent call last):" in body:  # its last unindented line names the exception
                failed += 1
                exception = next(line for line in reversed(body.splitlines()) if line and not line[0].isspace())
                errors.append(f"- {title}, {command[:16]} — `{plain(exception)}`")
        launches = f"{len(starts)}" + (f", с `--resume` {resumed}" if resumed else "")
        rows.append([title, launches, starts[-1] if starts else "—", len(warned), failed,
                     f"[{slug}.log](logs/{slug}.log)"])
        warnings += [f"- {title} — `{plain(w)}`" for w in dict.fromkeys(warned)]
    parts = [table(["стадия", "запуски", "последний", "предупр.", "ошибки", "лог"], rows)]
    if errors:
        parts.append("Ошибки, на которых остановились запуски:\n\n" + "\n".join(errors))
    if warnings:
        parts.append("Предупреждения:\n\n" + "\n".join(warnings))
    return parts


def collect(cfg, chosen):
    """All metrics of all stages in one dict."""
    def run(path, results=None):
        config = Path(path, "config.yaml")
        out = {"config": OmegaConf.to_container(OmegaConf.load(config)) if config.is_file() else None}
        if results:
            return out | {"results": load(Path(path, results))}
        return out | {"validation": load(Path(path, "metrics.jsonl")), "training": load(Path(path, "train.jsonl"))}

    return {"stage1_large": run(cfg.large), "stage2_export": run(cfg.export, "summary.json"),
            "stage3_small": run(cfg.small), "stage4_val": run(cfg.val, "benchmark.json"),
            "stage5_test": run(cfg.test, "benchmark.json"), "threshold": chosen,
            "sweep": {file.parent.name: load(file) for file in sorted(Path(cfg.sweep).glob("*/trials.jsonl"))}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("overrides", nargs="*", help="dotted overrides, e.g. out=runs/report_v2")
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)
    plt.switch_backend("Agg")

    out = Path(cfg.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    val, test = load(Path(cfg.val, "benchmark.json")), load(Path(cfg.test, "benchmark.json"))
    chosen = choose(val, cfg.max_launch_rate) if val else None
    parts = [*header(cfg), SCHEME, FORMULAS,
             "## Настройки", *settings({"стадия 1, B2": cfg.large, "стадия 3, B0": cfg.small}, TRAIN_KEYS),
             *settings({"стадия 4, val Cityscapes": cfg.val, "стадия 5, ACDC": cfg.test}, BENCHMARK_KEYS),
             "## Стадия 1: обучение B2", *training("B2", cfg.large, out, "train_large"),
             "## Стадия 2: экспорт неопределённости B2", *export(cfg.export, out),
             "## Стадия 3: обучение B0", *training("B0", cfg.small, out, "train_small"),
             "## Стадия 4: выбор порога на val Cityscapes", *threshold_section(val, chosen, cfg, out),
             "## Стадия 5: тест на ACDC", *test_section(test, chosen, cfg, out),
             "## Перебор гиперпараметров", *sweep(cfg.sweep),
             "## Запуски", *runs(cfg, out)]
    (out / "report.md").write_text("\n\n".join(parts) + "\n")
    (out / "metrics.json").write_text(json.dumps(collect(cfg, chosen), indent=1))
    print(f"report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
