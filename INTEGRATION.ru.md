# Руководство по встраиванию: гиперболическая неопределённость B0 и B2 (голова HyperUL)

## 1. О чём документ

Здесь описано, как работает гиперболический метод этой ветки, как перенести его в общий проект, как заново пройти
его стадии после объединения и как подключить его к B0 и B2.

Параметры изображений — как в этой ветке:
- B0 `nvidia/segformer-b0-finetuned-cityscapes-1024-1024` и B2 `nvidia/segformer-b2-finetuned-cityscapes-1024-1024`;
- полный кадр: Cityscapes 2048 × 1024, ACDC 1920 × 1080;
- патчи 32 × 32 пикселя.

Метод добавляет к модели одну маленькую голову — свёртку 1 × 1. Она читает тот же тензор, что и последний слой
штатной сети, то есть вход `decode_head.classifier`. Сама сеть и её выходы не меняются.

Голова B0 выдаёт три карты, по одному значению на каждый патч 32 × 32:
- `own` — гиперболическую неопределённость B0;
- `large` — прогноз гиперболической неопределённости B2 на этом же кадре (B2 при этом не запускается);
- `gain` = `own` − `large`.

Голова B2 даёт гиперболическую неопределённость B2. Стадии используют её, чтобы получить обучающие цели для головы
B0.

## 2. Как работает метод

### 2.1 Идея

HyperUL (Chen, Peng, Cao, Röning, «Hyperbolic Uncertainty Aware Semantic Segmentation», IEEE T-ITS 2023) отображает
выходы классов сегментационной сети в гиперболическое пространство. Расстояние от точки до начала координат
показывает, насколько сеть уверена: чем оно меньше, тем выше неопределённость.

В этой ветке:
1. SegFormer заморожен. Учится только копия его последней свёртки 1 × 1, и стартует она со штатных весов.
2. B2 учится оценивать свою неопределённость.
3. B0 учится оценивать свою неопределённость, а через дополнительный выходной канал — неопределённость, которая
   будет у B2 на этом же кадре. Так B0 оценивает её без запуска B2.
4. Патч, где B0 не уверена, а B2, по прогнозу, будет уверена, получает высокий `gain`.

### 2.2 Формулы

Пусть f — вход `decode_head.classifier` в одной точке карты 1/4 разрешения (D = 256 каналов у B0, 768 у B2),
W и b — веса головы, s = 0.2 (`model.classifier_scale`). Тогда

$$o = W f + b, \qquad v = s \, o_{1 \ldots 19}$$

Экспоненциальное отображение в начале координат модели Лоренца с кривизной −c даёт точку

$$x = \frac{1}{\sqrt{c}} \left( \cosh(\sqrt{c} \lVert v \rVert), \; \sinh(\sqrt{c} \lVert v \rVert) \frac{v}{\lVert v \rVert} \right)$$

- 19 пространственных координат x — это логиты классов.
- Расстояние до начала координат d = ‖v‖; в коде √c ‖v‖ ограничено сверху числом 15.
- d билинейно поднимается до разрешения кадра и усредняется по патчам 32 × 32: получается d̄.
- Неопределённость патча (Eq. 15 статьи) считается по формуле ниже, максимум берётся по кадру. У самого
  уверенного патча кадра u = 0.

$$u = 1 - \frac{\bar{d}}{\max \bar{d}}$$

- 20-й выход B0 — прогноз неопределённости B2: û₂ = σ(среднее o₂₀ по патчу).
- Выигрыш патча g = u₀ − û₂.

Лосс обучения (HyperUL, Eq. 14) — кросс-энтропия гиперболических логитов, вес пикселя 1 / ln(t + d̄ / max d̄),
t = 1.869. OHUEM оставляет долю hr (1.0) наименее уверенных размеченных пикселей. 20-й выход B0 учится через
бинарную кросс-энтропию по экспортированной неопределённости B2.

### 2.3 Стадии

| Стадия | Модель | Что учится | Данные | Результат |
|---|---|---|---|---|
| 1 | B2 | Голова, лосс HyperUL | Cityscapes train (2975) | `runs/large/best.pt` |
| 2 | B2 | Ничего, экспорт u₂ | Cityscapes val (500), B2 их не видела | `runs/large/uncertainty/val/` |
| 3 | B0 | Голова: HyperUL на train, выход 20 на val | Cityscapes train и val | `runs/small/best.pt` |

```
STAGE 1  Cityscapes train ──> B2 (frozen) ──C──> head 768→19 ──Lorentz──> logits, d ──HyperUL──> runs/large/best.pt
STAGE 2  Cityscapes val ────> B2 + head ──> u₂ per patch ──PNG, 1/4 resolution──> runs/large/uncertainty/val/
STAGE 3  Cityscapes train ──> B0 (frozen) ──C──> head 256→20 ── outputs 1..19 ──HyperUL──┐
         Cityscapes val ────> B0 (frozen) ──C──> head ─────────── output 20 ──BCE(u₂)────┴──> runs/small/best.pt
```

На стадии 3 кадры train учат только сегментацию, а кадры val — только прогноз неопределённости B2. Метки классов
val в обучении не участвуют, поэтому mIoU на val по-прежнему честно выбирает чекпойнт.

`best.pt` — чекпойнт той валидации, где val mIoU лучшая. `src.sweep` умеет перебирать c, `classifier_scale`, t,
hr, weight decay и learning rate.

Ещё ветка оценивает головы (`src.benchmark`, `src.report`: mIoU, ECE, MAE прогноза û₂, метрики ошибок по патчам на
ACDC). Для встраивания эти скрипты не обязательны.

## 3. Куда голова встраивается в B0 и B2

| Модель | Штатный чекпойнт | Кадр на входе | Вход головы C | Голова |
|---|---|---|---|---|
| B0 | `nvidia/segformer-b0-finetuned-cityscapes-1024-1024` | Полный кадр: Cityscapes 2048 × 1024, ACDC 1920 × 1080 | (1, 256, H/4, W/4) | 256 → 20 |
| B2 | `nvidia/segformer-b2-finetuned-cityscapes-1024-1024` | Полный кадр | (1, 768, H/4, W/4) | 768 → 19 |

C — выход цепочки `linear_fuse → batch_norm → ReLU` в декодере; dropout в режиме eval ничего не делает. Хук
`forward_pre_hook` на `decode_head.classifier` перехватывает C во время обычного прохода штатной модели. Голова
ничего не пишет в сеть, штатные логиты не меняются (раздел 9).

`core.models.SegModel("b0")` общего проекта — это B0 `…-512-1024` на уменьшенном кадре, и её `features["C"]` голове
не подходит. Нужен проход B0 `…-1024-1024` по полному кадру (раздел 8).

ВНИМАНИЕ: голова подходит только к тому чекпойнту и размеру кадра, на которых её обучали. Нельзя подавать ей C от
B0 `…-512-1024` или от уменьшенного кадра.

## 4. Что получает голова

1. C из прохода штатной B0 `…-1024-1024` по одному полному кадру.
2. `size` = (W, H) этого кадра.
3. Чекпойнт стадии 3 (`runs/small/best.pt`). Голова берёт из него только `decode_head.classifier` и конфиг
   (`manifold`, `c`, `classifier_scale`, `patch`).

Разметка, B2 и штатные логиты голове не нужны.

## 5. Что выдаёт голова

`HyperbolicHead(чекпойнт)(C, size)` возвращает словарь:

| Ключ | Форма | Тип | Содержание |
|---|---|---|---|
| `own` | (Hp, Wp) | float32 | Гиперболическая неопределённость модели, 0…1; 0 — самый уверенный патч кадра |
| `large` | (Hp, Wp) | float32 | Только B0. Прогноз неопределённости B2, 0…1 |
| `gain` | (Hp, Wp) | float32 | Только B0. `own` − `large`; чем больше, тем увереннее, по прогнозу, будет B2 по сравнению с B0 |

Hp = ⌈H / 32⌉, Wp = ⌈W / 32⌉: 32 × 64 на Cityscapes и 34 × 60 на ACDC (последний ряд высотой 24 пикселя). Патчи
идут построчно: patch = row · Wp + col. Тензоры лежат на том же устройстве, что и голова.

На Cityscapes патчи покрывают те же участки кадра, что и патчи 16 × 16 ветки `SAE` на 1024 × 512. На ACDC сетки
разные: у `SAE` 36 × 64 на 1024 × 576.

## 6. Перенос в общий проект

### 6.1 Файлы

Из этой ветки в общий проект переносится следующее:

| Файл | Что сделать |
|---|---|
| `src/`, `configs/` | Скопировать без изменений: в базовом коде папок с такими именами нет |
| `src/head.py` | Новый файл, скопировать из этого руководства (ниже) |
| `README.md` | Положить как `docs/hyperbolic.md`: там подробности стадий — раскладка данных, две GPU, перебор, время |
| `requirements.txt`, `.gitignore` | Не копировать, а дополнить файлы общего проекта (раздел 6.2) |

`src/head.py`:

```python
"""Hyperbolic uncertainty of SegFormer next to its default classifier (branch double_hyperul_segformer)."""
import torch
import torch.nn.functional as F
from torch import nn

from src.hyperbolic import MANIFOLDS

NC = 19


class HyperbolicHead(nn.Module):
    """The trained last 1x1 conv of a src.train checkpoint, applied to the input of the default classifier.
    The default classifier and its outputs do not change. checkpoint: runs/small/best.pt (B0, 20 outputs) or
    runs/large/best.pt (B2, 19 outputs) of the branch, or a file with only their classifier tensors."""

    def __init__(self, checkpoint):
        super().__init__()
        ckpt = torch.load(checkpoint, map_location="cpu")
        m, state = ckpt["config"]["model"], ckpt["model"]
        weight, bias = state["net.decode_head.classifier.weight"], state["net.decode_head.classifier.bias"]
        self.conv = nn.Conv2d(weight.shape[1], weight.shape[0], 1)
        self.conv.load_state_dict({"weight": weight, "bias": bias})
        self.manifold = MANIFOLDS[m["manifold"]](m["c"])
        self.scale, self.patch = m["classifier_scale"], m["patch"]

    @torch.inference_mode()
    def forward(self, features, size):
        """features: (1, D, h, w), the input of decode_head.classifier (SegModel.features["C"]); size: (W, H) of the
        frame given to the model. Returns maps (Hp, Wp) per patch: own, the uncertainty of the model; for B0 also
        large, the predicted uncertainty of B2, and gain = own - large."""
        out = self.conv(features.float())
        z = self.manifold.expmap0(self.scale * out[:, :NC])
        hw = (size[1], size[0])
        dist = F.interpolate(self.manifold.dist0(z)[:, None], hw, mode="bilinear", align_corners=False)
        dist = F.avg_pool2d(dist, self.patch, ceil_mode=True)[0, 0]
        result = {"own": 1 - dist / dist.max().clamp_min(1e-5)}  # Eq. 15
        if out.shape[1] > NC:
            large = F.interpolate(out[:, NC:], hw, mode="bilinear", align_corners=False)
            result["large"] = F.avg_pool2d(large, self.patch, ceil_mode=True)[0, 0].sigmoid()
            result["gain"] = result["own"] - result["large"]
        return result
```

### 6.2 Пакеты и игнорируемые файлы

В `requirements.txt` общего проекта уже есть torch, torchvision, transformers, scipy и numpy. Для стадий нужно
добавить `omegaconf`, `tensorboard`, `tqdm`, `matplotlib`.

В `.gitignore` общего проекта нужно добавить `runs/`: туда стадии пишут все результаты.

## 7. Повторный прогон стадий после объединения

### 7.1 Данные

Стадии читают официальную раскладку Cityscapes из `data/cityscapes` — той же папки, что и в базовом коде:

```
data/cityscapes/
├── leftImg8bit/{train,val}/<city>/*_leftImg8bit.png
└── gtFine/{train,val}/<city>/*_gtFine_labelIds.png
```

ВНИМАНИЕ: базовый код по умолчанию скачивает только val, а стадиям нужен ещё и train с разметкой. В `data/cityscapes`
нужно распаковать `leftImg8bit_trainvaltest.zip` и `gtFine_trainvaltest.zip`.

ACDC нужен только для необязательной оценки. Базовый код хранит его в `data/acdc` (`rgb_anon/`, `gt/`). Раскладка
Cityscapes делается симлинками — скриптом из `docs/hyperbolic.md` (раздел «ACDC») с `ACDC=$PWD/data/acdc` и
`DST=$PWD/data/acdc_cityscapes`. Если у ACDC скачана только часть val, в цикле скрипта нужно оставить только `val`,
а бенчмарк запускать с `split=val`.

### 7.2 Команды

Все команды запускаются из корня общего проекта. Они даны для двух GPU с bf16. Для одной GPU вместо
`torchrun --standalone --nproc_per_node=2 -m src.train` пишется `python -m src.train`. `data.workers` — общее число
процессов загрузки.

0. Пробный запуск — проверяет окружение, данные и скорость:

   ```bash
   torchrun --standalone --nproc_per_node=2 -m src.train configs/train.yaml configs/large.yaml \
     data.root=data/cityscapes out=runs/try train.iters=200 log.every=20 log.val_every=200 train.amp=true
   ```

   На шаге 200 val mIoU должна быть около 0.8. После этого `runs/try` можно удалить.

1. Стадия 1, голова B2:

   ```bash
   torchrun --standalone --nproc_per_node=2 -m src.train configs/train.yaml configs/large.yaml \
     data.root=data/cityscapes train.amp=true
   ```

   Или перебор гиперпараметров с копированием лучшего прогона в `runs/large`:

   ```bash
   python -m src.sweep configs/sweep.yaml stage=large root=data/cityscapes 'overrides=["train.amp=true"]'
   mkdir -p runs/large && cp -r runs/sweep/large/<номер>/. runs/large/
   ```

2. Стадия 2, экспорт неопределённости B2:

   ```bash
   python -m src.export_uncertainty configs/export_uncertainty.yaml root=data/cityscapes
   ```

3. Стадия 3, голова B0:

   ```bash
   torchrun --standalone --nproc_per_node=2 -m src.train configs/train.yaml configs/small.yaml \
     data.root=data/cityscapes train.amp=true
   ```

   Или перебор гиперпараметров с копированием лучшего прогона в `runs/small`:

   ```bash
   python -m src.sweep configs/sweep.yaml stage=small root=data/cityscapes 'overrides=["train.amp=true"]'
   mkdir -p runs/small && cp -r runs/sweep/small/<номер>/. runs/small/
   ```

4. Необязательная оценка (подробности — в `docs/hyperbolic.md`):

   ```bash
   python -m src.benchmark configs/benchmark.yaml root=data/cityscapes split=val out=runs/benchmark_val
   python -m src.benchmark configs/benchmark.yaml root=data/acdc_cityscapes
   python -m src.report configs/report.yaml
   ```

## 8. Как подключить голову

Штатная B0 `…-1024-1024`, хук и голова создаются один раз на процесс. Затем на каждом кадре B0 один раз
прогоняется по полному кадру, и её C отдаётся голове:

```python
import torch
from transformers import SegformerForSemanticSegmentation

from src.head import HyperbolicHead

device = "cuda"
b0 = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
b0 = b0.to(device).eval()                                  # default B0 1024-1024
features = {}
b0.decode_head.classifier.register_forward_pre_hook(lambda m, a: features.__setitem__("C", a[0]))
head = HyperbolicHead("runs/small/best.pt").to(device)
mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1) * 255
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1) * 255

image = sample["image"]                                    # uint8 (3, H, W): the full frame
with torch.inference_mode():
    logits = b0(pixel_values=(image[None].to(device).float() - mean) / std).logits   # default output of B0
hyper = head(features["C"], (image.shape[2], image.shape[1]))                      # own, large, gain: (Hp, Wp)
```

`sample["image"]` — полный кадр из датасетов `core.data` (с `full=True`). Нормировка такая же, как в
`core.models.SegModel`.

Голову нужно вызывать до следующего прохода B0: `features` хранит только последний проход. Кадры подаются по одному.

Можно и иначе: добавить в `CHECKPOINTS` в `core/models.py` ключ
`"b0_1024": "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"`. Тогда `load_model("b0_1024")` поставит тот же
хук. Вызывать её нужно на полном кадре (`sample["image"]`), а не через `probs()`: `probs()` подаёт B0 уменьшенный
кадр.

B2 (по желанию): чтобы получить неопределённость B2 на полном кадре, то же самое делается с
`nvidia/segformer-b2-finetuned-cityscapes-1024-1024` и `HyperbolicHead("runs/large/best.pt")`. Голова B2 выдаёт
только `own`.

## 9. Приёмочный тест

Проводится после стадии 3. `path` — кадр Cityscapes:

```python
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation

from src.head import HyperbolicHead
from src.hyperbolic import hyperbolic_uncertainty
from src.model import load_model

device, ckpt = "cuda", "runs/small/best.pt"
b0 = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
b0 = b0.to(device).eval()                                  # default B0
head = HyperbolicHead(ckpt).to(device)
reference = load_model(ckpt, 19).to(device).eval()         # the model of stage 3
image = torch.from_numpy(np.asarray(Image.open(path).convert("RGB")).copy()).permute(2, 0, 1)   # full frame
mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1) * 255
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1) * 255
x = (image[None].to(device).float() - mean) / std

features = {}
with torch.inference_mode():
    before = b0(pixel_values=x).logits
    b0.decode_head.classifier.register_forward_pre_hook(lambda m, a: features.__setitem__("C", a[0]))
    after = b0(pixel_values=x).logits
    _, dist, large = reference(x)
out = head(features["C"], (image.shape[2], image.shape[1]))
grid = lambda m: F.avg_pool2d(m[None, None], head.patch, ceil_mode=True)[0, 0]
print("default logits", (after - before).abs().max().item())
print("own", (out["own"] - grid(hyperbolic_uncertainty(dist)[0])).abs().max().item())
print("large", (out["large"] - grid(large[0].sigmoid())).abs().max().item())
```

1. Разница штатных логитов должна быть 0: голова только читает C.
2. Разница `own` и `large` должна быть меньше 1e-4.
