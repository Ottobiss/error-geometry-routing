# error-geometry-routing

Геометрия ошибок нейросетей и адаптивный инференс в семантической сегментации.

Лёгкая модель (SegFormer-B0) сегментирует кадр; роутер по её признакам решает, на каких
регионах запустить тяжёлую модель (SegFormer-B2). Регион — компонента B0: 8-связная область
одного предсказанного класса. Цель роутера — выигрыш от замены
Δ = (пиксели, где права только B2) − (пиксели, где права только B0).

Главный вопрос: лучше ли геометрические признаки внутреннего представления B0 (Махаланобис,
плотность, kNN, прототипы) стандартных оценок уверенности (энтропия, предиктор провала p и
MetaSeg из Kang et al., arXiv 2610.16058), особенно при сдвиге домена Cityscapes → ACDC.
Результаты — в [EXPERIMENTS.md](EXPERIMENTS.md).

## Структура

```
core/        датасеты (torch Dataset), модели (nn.Module с хуками признаков), компоненты,
             оценка роутинга, раскладка results/
download/    скачивание Cityscapes и ACDC
features/    подготовка: маски и таблица компонент, логиты, признаки компонент (подгонка на P0)
routing/     методы роутера (реестр), запуск и оценка роутинга, попиксельная оценка
baseline/    воспроизведение Kang et al. на метке провала
splits/      списки кадров популяций: Cityscapes val случайно (сид 0) разбит на P0 — подгонка
             (300 кадров) и P1 — тест (200)
tests/
```

## Окружение

Python 3.12, GPU с CUDA (проверено на RTX 4060 Laptop 8 ГБ).

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Все команды ниже — из корня проекта через `.venv\Scripts\python.exe`. В консоли Windows с
кириллицей полезно `PYTHONIOENCODING=utf-8`.

## Данные

Модели — официальные чекпойнты NVIDIA на HuggingFace, скачиваются сами.

Cityscapes (нужен аккаунт на cityscapes-dataset.com): разметку `gtFine_trainvaltest.zip` скачать
вручную и распаковать `gtFine/val` в `data/cityscapes/`; кадры val (1.1 ГБ из 11 ГБ архива)
вытаскиваются из архива на сервере по частям:

```
python -m download.cityscapes --check
python -m download.cityscapes
```

ACDC (без аккаунта): кадры val (~0.8 ГБ) читаются потоком из архива на сервере (~15 ГБ трафика);
разметку `gt_trainval.zip` распаковать в `data/acdc/` (нужны `gt/<условие>/val`).

```
python -m download.acdc
```

## Пайплайн

```
python -m features.components                 # B0 и B2: маски и таблица компонент (~10 мин)
python -m features.logits                     # логиты B0 для md_output (~3 мин)
python -m features.build embeddings           # векторы декодера B0 по компонентам (~3 мин)
python -m features.build prototypes           # прототипы классов (~10 мин)
python -m features.build ddu                  # DDU / GDA (~7 мин)
python -m features.build knn                  # kNN / cDNP / CSLS (~16 мин)

python -m baseline.failure                    # сверка с таблицей 1 Kang et al.
python -m routing.run --method entropy --fields
python -m routing.run --method proto -p variant=B -p feature=spred_mean --name proto_B_spred_mean

python -m features.components --dataset acdc  # то же для ACDC (подгонка остаётся на CS P0)
python -m routing.summary                     # все методы на CS P1 и ACDC (по условиям и все)

python -m features.tiles --dataset cityscapes --input image   # B2 по патчам 256 + контекст 64, замеры времени
python -m routing.tiles --test cityscapes:p1 --tiles image_t256_m64   # каскад: mIoU против мс на кадр
python -m routing.cascade                     # каскад вживую: реальное время, fp32 / fp16
python -m features.tile_features --dataset cityscapes --tiles image_t256_m64   # признаки патчей
python -m routing.tile_router                 # обученный роутер на патчах против энтропии
```

Методы роутера — `routing/methods/`; список: `python -m routing.run --help`. Любой построитель
признаков можно проверить смоук-тестом: `python -m features.build knn --limit 3`.

## Тесты

```
python -m pytest tests
```
