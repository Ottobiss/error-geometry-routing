# error-geometry-routing

Маршрутизация между малой (SegFormer-B0) и большой (SegFormer-B2) моделями семантической сегментации по предсказанной неопределённости. Неопределённость гиперболическая, по HyperUL (Chen, Peng, Cao, Röning, «Hyperbolic Uncertainty Aware Semantic Segmentation», IEEE T-ITS 2023). Малая модель работает на каждом кадре и предсказывает свою неопределённость и неопределённость большой модели. Большая запускается, только если на кадре есть участки, где малая не уверена, а большая, по прогнозу, уверена.

## Как устроен пайплайн

Обе модели — SegFormer, уже обученные на Cityscapes (веса NVIDIA с Hugging Face). Сеть заморожена, обучаются только слои, на которые действует HyperUL: последний 1×1 conv декодера (евклидовы эмбеддинги, которые отображаются в шар Пуанкаре) и гиперболический MLR-классификатор. Неопределённость (Eq. 15 статьи) усредняется по патчам 32×32.

1. **Стадия 1, большая модель B2.** Обучение с лоссом HyperUL: кросс-энтропия, взвешенная гиперболической неопределённостью. Результат в `runs/large`.
2. **Стадия 2, экспорт.** Обученная B2 один раз считает неопределённость каждого патча каждого кадра train. Карты сохраняются в PNG и дальше не меняются. Результат в `runs/large/uncertainty`.
3. **Стадия 3, малая модель B0.** HyperUL по собственной неопределённости плюс BCE между предсказанием неопределённости B2 (отдельный выходной канал) и картами из стадии 2. Результат в `runs/small`.
4. **Стадия 4, оценка маршрутизации на val.** Обе модели прогоняются на каждом кадре. Для каждого сегмента (связной области одного класса в предсказании B0) считается score — среднее разности «своя неопределённость минус предсказанная неопределённость B2». Если score больше порога, сегмент берётся у B2. Перебирается несколько порогов. Результат в `runs/eval_router`.

На инференсе (`src.infer_router`) B0 работает всегда, а B2 — только если хотя бы один сегмент прошёл порог. Её предсказание подставляется только в эти сегменты.

Для сравнения можно обучить базовую B0 только с HyperUL, без предсказания неопределённости B2 (`runs/hyperul`).

## Требования

- **ОС и GPU.** Linux и NVIDIA GPU с памятью от 10 ГБ, RTX 3080 достаточно. Для `train.amp=true` (bf16) нужна архитектура Ampere или новее: RTX 30xx/40xx, A100 и т. п. На T4 и V100 bf16 нет, там оставьте `train.amp=false`.
- **CPU.** Желательно 8 ядер и больше. При замороженной сети скорость обычно упирается в загрузку данных: декодирование PNG 2048×1024.
- **Память и диск.** RAM 16–32 ГБ, ~12 ГБ диска под Cityscapes. Результаты занимают меньше 1 ГБ.

## Установка (uv)

Если [uv](https://docs.astral.sh/uv/) ещё не установлен: `curl -LsSf https://astral.sh/uv/install.sh | sh` (или `pip install uv`).

```bash
uv venv --python 3.12        # окружение .venv в корне репозитория; нужный Python uv скачает сам
source .venv/bin/activate
uv pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

На Linux колёса PyTorch с PyPI уже собраны с CUDA. Последняя команда должна вывести `True` и число GPU. Если выводит `False` или драйвер старый, поставьте сборку PyTorch под свой драйвер, а потом снова requirements:

```bash
uv pip install torch torchvision --torch-backend=auto   # uv сам выберет сборку CUDA под драйвер
uv pip install -r requirements.txt
```

Сборку можно указать и явно, например `--torch-backend=cu130`. Версию CUDA, которую поддерживает драйвер, показывает `nvidia-smi` в правом верхнем углу.

`transformers` нужен версии ниже 5 (проверено на 4.46.3). Это зафиксировано в `requirements.txt`.

Все команды ниже запускаются из корня репозитория в активированном окружении: код лежит в пакете `src`. Без активации окружения их можно запускать через `uv run`, например `uv run python -m src.train ...` или `uv run torchrun ...`.

## Данные

### Cityscapes

С [cityscapes-dataset.com](https://www.cityscapes-dataset.com) (нужна регистрация) понадобятся `leftImg8bit_trainvaltest.zip` и `gtFine_trainvaltest.zip`. После распаковки структура такая:

```
/data/cityscapes/
├── leftImg8bit/{train,val}/<город>/*_leftImg8bit.png
└── gtFine/{train,val}/<город>/*_gtFine_labelIds.png
```

Используются файлы `*_labelIds.png`. Перевод в 19 классов делается на лету, остальные метки игнорируются. Часть test не нужна.

### ACDC

Загрузчик понимает только раскладку Cityscapes. ACDC (папки `rgb_anon/` и `gt/` с [acdc.vision.ee.ethz.ch](https://acdc.vision.ee.ethz.ch)) переводится в неё симлинками, без копирования:

```bash
ACDC=/data/acdc                  # абсолютный путь к папке с rgb_anon/ и gt/
DST=/data/acdc_cityscapes
for split in train val; do
  for cond in fog night rain snow; do
    mkdir -p "$DST/leftImg8bit/$split/$cond" "$DST/gtFine/$split/$cond"
    for img in "$ACDC/rgb_anon/$cond/$split"/*/*_rgb_anon.png; do
      seq=$(basename "$(dirname "$img")"); name=$(basename "$img" _rgb_anon.png)
      ln -sf "$img" "$DST/leftImg8bit/$split/$cond/${name}_leftImg8bit.png"
      ln -sf "$ACDC/gt/$cond/$split/$seq/${name}_gt_labelIds.png" "$DST/gtFine/$split/$cond/${name}_gtFine_labelIds.png"
    done
  done
done
```

Дальше везде нужно указывать `data.root=/data/acdc_cityscapes` (в стадиях 2 и 4 — `root=`), а при обучении ещё `'data.base_size=[1920,1080]'`. Чтобы результаты не смешались с Cityscapes, задайте отдельные пути, например `out=runs/acdc/large`. Разметка test у ACDC закрыта, поэтому валидация идёт на val (406 кадров). Замороженные сети обучены на Cityscapes, и на ночи и тумане качество будет ограничено ими.

## Веса

По умолчанию берутся `nvidia/segformer-b0-finetuned-cityscapes-1024-1024` (малая модель) и `nvidia/segformer-b2-finetuned-cityscapes-1024-1024` (большая). При первом запуске они скачиваются в `~/.cache/huggingface`. Их классификатор на 19 классов заменяется новым слоем. Если на машине нет интернета, скачайте веса заранее и передавайте путь:

```bash
python -c "
from transformers import SegformerForSemanticSegmentation as S
for v in ('b0', 'b2'):
    S.from_pretrained(f'nvidia/segformer-{v}-finetuned-cityscapes-1024-1024').save_pretrained(f'weights/segformer-{v}')
"
# стадия 1: model.pretrained=weights/segformer-b2; стадия 3 и базовая B0: model.pretrained=weights/segformer-b0
```

## Запуск

### Пробный запуск

Перед долгим обучением стоит проверить память и скорость:

```bash
python -m src.train configs/train.yaml configs/large.yaml data.root=/data/cityscapes out=runs/try \
  train.iters=200 log.every=20 log.val_every=200
```

В TensorBoard смотрите `train/mem_gb` и `train/img_per_s`. Полное время обучения примерно равно `train.iters × data.batch_size / img_per_s` плюс валидации.

### Одна GPU

```bash
python -m src.train configs/train.yaml configs/large.yaml data.root=/data/cityscapes   # 1. B2 -> runs/large
python -m src.export_uncertainty configs/export_uncertainty.yaml root=/data/cityscapes  # 2. -> runs/large/uncertainty
python -m src.train configs/train.yaml configs/small.yaml data.root=/data/cityscapes   # 3. B0 -> runs/small
python -m src.eval_router configs/eval_router.yaml root=/data/cityscapes               # 4. -> runs/eval_router
python -m src.train configs/train.yaml data.root=/data/cityscapes                      # базовая B0 -> runs/hyperul
```

Стадия 2 берёт `runs/large/best.pt`, стадия 3 — карты из `runs/large/uncertainty`, стадия 4 — оба `best.pt`. Эти пути задаются в конфигах (см. «Настройки»). Конкретную карту выбирают так: `CUDA_VISIBLE_DEVICES=1 python -m ...`.

### Две GPU

Обучение (стадии 1 и 3, базовая B0) запускается через `torchrun`, по одному процессу на GPU (DDP):

```bash
torchrun --standalone --nproc_per_node=2 -m src.train configs/train.yaml configs/large.yaml data.root=/data/cityscapes
torchrun --standalone --nproc_per_node=2 -m src.train configs/train.yaml configs/small.yaml data.root=/data/cityscapes
```

- `data.batch_size` и `data.workers` задаются на все GPU сразу и делятся поровну, поэтому батч должен делиться на число GPU. Гиперпараметры получаются те же, что на одной GPU.
- Обучается всего 19–52 тыс. параметров, так что синхронизация между картами почти ничего не стоит.
- Валидирует, пишет в TensorBoard и сохраняет чекпойнты только первый процесс, остальные его ждут. Статистики `train/*` считаются по его части батча, а `train/img_per_s` показывает общую скорость.
- Экспорт, оценка и инференс работают на одной GPU и укладываются в 10 ГБ.
- Если `torchrun` зависает на старте (бывает на потребительских картах без NVLink), запускайте с переменной окружения: `NCCL_P2P_DISABLE=1 torchrun ...`.

### Продолжение и оценка чекпойнта

```bash
python -m src.train configs/train.yaml configs/large.yaml data.root=/data/cityscapes --resume runs/large/last.pt
python -m src.train configs/train.yaml configs/large.yaml data.root=/data/cityscapes --eval-only --resume runs/large/best.pt
```

Чекпойнты пишутся при каждой валидации: `last.pt` — последний, `best.pt` — лучший по `val/mIoU`. `--resume` восстанавливает модель, оптимизатор, расписание lr и номер итерации. Конфиг при этом должен быть тем же: `train.iters` задаёт ещё и длину расписания lr. `--eval-only` без `--resume` оценит необученную голову. Обе опции работают и под `torchrun`.

### Инференс

```bash
python -m src.infer_router configs/infer_router.yaml input=/path/to/images threshold=0.3    # каскад B0 -> B2
python -m src.infer configs/infer.yaml input=/path/to/images checkpoint=runs/small/best.pt  # одна модель
```

`input` — файл или папка; в папке рекурсивно берутся `.png`, `.jpg`, `.jpeg`.

- `src.infer_router` пишет в `runs/infer_router/`:
  - `pred/` — цветная сегментация;
  - `uncertainty/` — собственная неопределённость B0;
  - `large_uncertainty/` — предсказанная неопределённость B2;
  - `routed/` — маска сегментов, взятых у B2.
- `src.infer` пишет `pred/` и `uncertainty/` в `runs/infer/`.

Неопределённость сохраняется тепловой картой: чёрный цвет означает 0 (модель уверена), дальше красный и жёлтый, белый означает 1.

## Настройки

### Как передавать

Первый аргумент — базовый YAML. Дальше по порядку можно передать:
- другие YAML, которые перекрывают часть ключей (так устроены `large.yaml` и `small.yaml`);
- пары `ключ=значение` через точку.

Списки берите в кавычки: `'data.crop=[512,1024]'`. Неизвестный ключ вызывает ошибку, это защита от опечаток. `???` в конфиге означает обязательное значение. Итоговый конфиг сохраняется в `<out>/config.yaml` и внутрь чекпойнтов. Стадии 2–4 и инференс строят модель по нему, архитектуру отдельно указывать не нужно.

### Обучение: `configs/train.yaml`

`configs/large.yaml` (стадия 1) меняет `out: runs/large`, `model.variant: b2` и `model.pretrained` на веса B2. `configs/small.yaml` (стадия 3) меняет `out: runs/small` и задаёт `data.large_uncertainty: runs/large/uncertainty`.

| Ключ | По умолчанию | Что делает |
|---|---|---|
| `seed` | `0` | сид; при нескольких GPU у процесса `seed + rank` |
| `out` | `runs/hyperul` | папка запуска |
| `data.root` | обязательный | корень датасета в раскладке Cityscapes |
| `data.crop` | `[512, 1024]` | кроп при обучении, H × W |
| `data.base_size` | `[2048, 1024]` | базовый размер W × H для случайного масштабирования; для ACDC `[1920, 1080]` |
| `data.scale_range` | `[0.5, 2.0]` | диапазон случайного масштаба |
| `data.batch_size` | `8` | общий батч на все GPU |
| `data.workers` | `8` | общее число процессов загрузки данных на все GPU |
| `data.large_uncertainty` | `null` | папка стадии 2; включает выход неопределённости B2 (стадия 3) |
| `model.variant` | `b0` | энкодер: `b0` или `b2` |
| `model.pretrained` | `nvidia/segformer-b0-finetuned-cityscapes-1024-1024` | веса: id на Hugging Face или локальная папка; `null` — случайная инициализация |
| `model.freeze` | `true` | обучать только последний 1×1 conv и гиперболический MLR; `false` — всю сеть, как в статье (тогда `model.pretrained=nvidia/mit-b0` или `mit-b2`) |
| `model.embed_dim` | `64` | размерность гиперболических эмбеддингов |
| `model.manifold` | `poincare` | модель гиперболического пространства: `poincare` (шар, как в статье) или `lorentz` (гиперболоид) |
| `model.c` | `0.7` | пространство имеет кривизну −c |
| `model.patch` | `32` | сторона патча в пикселях для усреднения неопределённости; `1` — попиксельно, как в статье |
| `loss.t` | `1.869` | вес пикселя 1 / ln(t + d), Eq. 11 |
| `loss.hr` | `1.0` | порог отбора пикселей OHUEM, Eq. 13; `1.0` оставляет все |
| `loss.class_weights` | `null` | список весов 19 классов, Eq. 12 |
| `loss.large_unc_weight` | `1.0` | вес лосса неопределённости B2 (стадия 3) |
| `loss.large_unc_loss` | `bce` | лосс неопределённости B2: `bce`, `mse` или `l1` |
| `train.iters` | `80000` | число итераций, оно же длина расписания lr |
| `train.lr` | `1.5e-4` | lr энкодера (при `model.freeze` не используется) |
| `train.head_lr_mult` | `10` | множитель lr декодера и MLR; при `model.freeze` обучаемые слои идут с lr × 10 = 1.5e-3 |
| `train.weight_decay` | `0.01` | weight decay AdamW |
| `train.warmup` | `1000` | итерации линейного разогрева lr |
| `train.poly_power` | `0.9` | степень poly-спада lr |
| `train.amp` | `false` | bf16 autocast (Ampere и новее): меньше памяти, быстрее |
| `log.every` | `50` | шаг логирования `train/*` |
| `log.val_every` | `4000` | шаг валидации и сохранения чекпойнтов |
| `log.val_images` | `3` | число панелей валидации в TensorBoard |

c, t и hr — лучшие значения для SegFormer-B0 на Cityscapes из статьи (табл. XI–XII). Их подбирали для полного попиксельного обучения, так что с патчами и замороженной сетью их, возможно, стоит перебрать. 80k итераций тоже взяты из статьи. Для замороженной сети, скорее всего, хватит 10–20k; проверяйте по кривой `val/mIoU`.

### Стадия 2: `configs/export_uncertainty.yaml`

| Ключ | По умолчанию | Что делает |
|---|---|---|
| `checkpoint` | `runs/large/best.pt` | обученная B2 |
| `root` | обязательный | корень датасета |
| `split` | `train` | какой split экспортировать |
| `out` | `runs/large/uncertainty` | куда сохранять; должно совпадать с `data.large_uncertainty` стадии 3 |
| `stride` | `4` | карты сохраняются в 1/stride разрешения: патч 32 становится блоком 8×8 |
| `amp` | `false` | bf16 |
| `workers` | `4` | процессы загрузки данных |

### Стадия 4: `configs/eval_router.yaml`

| Ключ | По умолчанию | Что делает |
|---|---|---|
| `small`, `large` | `runs/small/best.pt`, `runs/large/best.pt` | чекпойнты моделей |
| `root` | обязательный | корень датасета, используется val |
| `out` | `runs/eval_router` | папка TensorBoard |
| `thresholds` | `[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]` | пороги: сегмент уходит к B2, если среднее «своя неопределённость − предсказанная неопределённость B2» больше порога |
| `stride` | `4` | сегменты ищутся в 1/stride разрешения |
| `amp` | `false` | bf16 |
| `images` | `3` | число панелей в TensorBoard |
| `workers` | `4` | процессы загрузки данных |

### Инференс: `configs/infer_router.yaml` и `configs/infer.yaml`

| Ключ | По умолчанию | Что делает |
|---|---|---|
| `small`, `large` | `runs/small/best.pt`, `runs/large/best.pt` | чекпойнты каскада (`infer_router`) |
| `checkpoint` | `runs/hyperul/best.pt` | чекпойнт одной модели (`infer`) |
| `input` | обязательный | файл или папка с изображениями |
| `out` | `runs/infer_router`, `runs/infer` | папка результатов |
| `threshold` | `0.2` | порог маршрутизации (`infer_router`) |
| `stride` | `4` | как в стадии 4 (`infer_router`) |
| `amp` | `false` | bf16 |
| `image_every` | `20` | панель в TensorBoard каждые N кадров |

## Перебор гиперпараметров

`src.sweep` запускает случайный поиск по главным гиперпараметрам одной стадии на всех видимых GPU:

```bash
python -m src.sweep configs/sweep.yaml stage=large root=/data/cityscapes   # стадия 1 (B2)
python -m src.sweep configs/sweep.yaml stage=small root=/data/cityscapes   # стадия 3 (B0) с оценкой маршрутизации
```

**Порядок работы:**
1. Перебор `stage=large`. Прогоны ранжируются по `metric`, по умолчанию по `val/hyperbolic_ECE`.
2. Лучшую B2 надо получить в `runs/large`: либо скопировать чекпойнт лучшего прогона (`runs/sweep/large/<номер>/last.pt`), либо обучить заново с напечатанными в конце параметрами (строка `best:`). Затем выполните экспорт (стадия 2) в `runs/large/uncertainty`.
3. Перебор `stage=small`. По умолчанию он берёт `runs/large/best.pt` и `runs/large/uncertainty`. Каждый прогон после обучения проходит через `eval_router`, а оценкой служит лучшая routed mIoU среди порогов, при которых B2 запускается не больше чем на `max_launch_rate` кадров.
4. Финальное обучение B0 с лучшими параметрами.

**Что перебирается** (`space` в [configs/sweep.yaml](configs/sweep.yaml)): `model.manifold`, `model.c`, `loss.t`, `loss.hr`, `model.embed_dim`, `train.weight_decay`, `train.lr`. Прогон 0 всегда идёт с параметрами по умолчанию, для сравнения. Пространство можно менять. Формат такой:
- `{key: ..., log: [low, high]}` — значение из лог-равномерного распределения;
- `{key: ..., choice: [...]}` — одно из перечисленных значений.

**Батч** подбирается автоматически: максимальный, при котором на каждой GPU остаётся `headroom_gb` свободной памяти.
- **Как подбирается.** В отдельных процессах замеряются шаги обучения и проход валидации на полном разрешении, затем результат проверяется реальным прогоном.
- **Сколько ждать на 24 ГБ.** В fp32 ориентировочно ~20 кадров на карту для B2 и ~60 для B0. С `overrides=["train.amp=true"]` для B2 примерно вдвое больше.
- **Для всех прогонов один.** Подобранный батч записывается и используется во всех прогонах этого перебора, чтобы их можно было сравнивать.

**Длительность прогона и lr:**
- Число итераций равно `samples / батч`. По умолчанию это столько же кадров, сколько 20k итераций при батче 8.
- Диапазон `train.lr` задан для батча 8 и масштабируется на √(батч / 8). В результатах указан фактический lr, поэтому финальное обучение запускайте с тем же `data.batch_size`, который напечатан рядом.

**Результаты** лежат в `runs/sweep/<стадия>/`:
- `trials.jsonl` — параметры и метрики всех прогонов;
- папки прогонов с `log.txt`, `metrics.jsonl`, чекпойнтами и TensorBoard (`tensorboard --logdir runs/sweep`).

В конце печатается таблица лучших прогонов.

**Повторный запуск** той же команды продолжает перебор: записанные прогоны пропускаются, прерванный продолжается со своего `last.pt`. Чтобы повторить упавший прогон, удалите его строку из `trials.jsonl`. Если меняете `space` или `seed`, задайте новый `out`.

**Размер патча** в перебор не входит, потому что он общий для всего пайплайна. Стадия `small` берёт `model.patch` из чекпойнта B2. Чтобы сравнить размеры патча, проводите обе стадии целиком с `overrides=["model.patch=16"]` и т. п.

**ACDC:** добавьте `root=/data/acdc_cityscapes` и `overrides=["data.base_size=[1920,1080]"]`. Несколько значений в `overrides` перечисляются через запятую, каждое в двойных кавычках, а весь аргумент — в одинарных: `'overrides=["data.base_size=[1920,1080]","train.amp=true"]'`.

**Время.** На двух картах по 24 ГБ один прогон длится ~30 мин: скорость упирается в загрузку данных, у `small` к этому добавляются ~5 мин на `eval_router`. 20 прогонов на стадию — около 10 ч.

| Ключ `configs/sweep.yaml` | По умолчанию | Что делает |
|---|---|---|
| `stage` | `large` | `large` — стадия 1 (B2), `small` — стадия 3 (B0) с `eval_router` |
| `root` | обязательный | корень датасета |
| `out` | `runs/sweep` | папка перебора |
| `trials` | `20` | число прогонов, включая прогон 0 с параметрами по умолчанию |
| `seed` | `0` | сид выборки параметров |
| `samples` | `160000` | обучающих кадров на прогон |
| `validations` | `4` | валидаций на прогон; оценку даёт последняя |
| `gpus` | `null` | число GPU; `null` — все видимые |
| `batch_per_gpu` | `null` | батч на GPU; `null` — подобрать максимальный |
| `headroom_gb` | `2.0` | сколько памяти GPU оставить свободной |
| `workers` | `8` | процессы загрузки данных, всего |
| `overrides` | `[]` | переопределения `train.yaml` для всех прогонов, применяются последними |
| `metric` | `val/hyperbolic_ECE` | оценка стадии `large`: `val/hyperbolic_ECE` (меньше — лучше), `val/mIoU` или `gap` (неопределённость на ошибочных минус на верных пикселях) |
| `large_checkpoint`, `large_uncertainty` | `runs/large/best.pt`, `runs/large/uncertainty` | B2 и её экспорт для стадии `small` |
| `max_launch_rate` | `0.5` | допустимая доля кадров с запуском B2 для оценки стадии `small` |
| `thresholds` | `0.0 … 0.6` с шагом 0.05 | пороги `eval_router` в стадии `small` |
| `space` | см. выше | пространство поиска |

## Мониторинг

```bash
tensorboard --logdir runs --samples_per_plugin images=100
```

| Теги | Где | Что показывают |
|---|---|---|
| `train/loss`, `train/large_unc`, `train/large_unc_mae` | обучение | общий лосс; лосс неопределённости B2 и её средняя ошибка (стадия 3). Минимум `train/large_unc` не равен нулю, поэтому запуски сравнивают по `train/large_unc_mae` |
| `train/weight`, `train/selected`, `train/dist` | обучение | средний вес HyperUL, доля пикселей после OHUEM, гиперболическое расстояние до начала координат (скаляр и гистограмма) |
| `train/lr`, `train/img_per_s`, `train/mem_gb` | обучение | lr; скорость в кадрах в секунду на все GPU; пиковая память GPU |
| `val/mIoU`, `val_iou/<класс>` | валидация | качество сегментации |
| `val/hyperbolic_ECE` | валидация | калибровка: ECE для 1 − неопределённость |
| `val/hyperbolic_uncertainty_correct`, `…_wrong` | валидация | средняя неопределённость на верных и ошибочных пикселях; на ошибочных она должна быть заметно больше |
| `small/*`, `large/*`, `small/large_unc_mae` | стадия 4 | те же метрики обеих моделей на val; ошибка предсказания неопределённости B2 |
| `route/fixable_pixels`, PR-кривая `route/fixable` | стадия 4 | доля пикселей, где B0 ошибается, а B2 прав; насколько score находит такие пиксели |
| `routed/mIoU`, `route/launch_rate`, `route/routed_pixels`, `route/fix_precision`, `route/fix_recall` | стадия 4, ось x — порог × 100 | качество каскада, доля кадров с запуском B2, доля переданных пикселей, точность и полнота исправлений |
| `infer/latency_ms`, `infer/large_launched`, `infer/routed_pixels` | инференс | задержка на кадр, был ли запуск B2, доля переданных пикселей |

Порог маршрутизации выбирают по графикам `route/*`: максимум `routed/mIoU` при допустимой `route/launch_rate`. Таблица по порогам печатается и в консоль.

## Выходные файлы

```
runs/
├── large/                          # стадия 1: events.*, config.yaml, metrics.jsonl, last.pt, best.pt
│   └── uncertainty/train/<город>/  # стадия 2: карты неопределённости B2
├── small/                          # стадия 3
├── hyperul/                        # базовая B0
├── eval_router/                    # стадия 4: TensorBoard и results.json
├── sweep/<стадия>/                 # перебор: trials.jsonl и папки прогонов
├── infer_router/                   # pred/, uncertainty/, large_uncertainty/, routed/
└── infer/                          # pred/, uncertainty/
```

`metrics.jsonl` (метрики каждой валидации) и `results.json` (итоги `eval_router` по всем порогам) нужны для скриптов, например для перебора.

## Время и память

Ниже ориентиры для Cityscapes при батче 8 в fp32. Точность около ±50%, поэтому сначала сделайте пробный запуск.

| Этап | 2× RTX 3080, 80k итераций | 2× RTX 3080, 20k итераций | Память на GPU |
|---|---|---|---|
| 1. B2 | ~2–2.5 ч | ~40 мин | ~5 ГБ, с `train.amp` ~3 ГБ |
| 2. Экспорт | ~5–10 мин | ~5–10 мин | ~4–5 ГБ |
| 3. B0 | ~2–2.5 ч | ~40 мин | ~2 ГБ |
| 4. Оценка | ~5 мин | ~5 мин | ~5 ГБ |

На одной RTX 3080 стадия 1 при батче 8 в fp32 занимает ~8–9 ГБ из 10, поэтому включайте `train.amp=true`. Полное обучение (`model.freeze=false`) требует для B2 порядка 25 ГБ при батче 8 и на 3080 не помещается.

## Частые проблемы

- **`CUDA out of memory`.** Поставьте `train.amp=true`, обучайте на двух GPU или уменьшите `data.batch_size`; последнее меняет гиперпараметры.
- **`torchrun` зависает на старте.** Запустите с `NCCL_P2P_DISABLE=1`.
- **`data.batch_size=... is not divisible by N GPUs`.** Батч должен делиться на число GPU.
- **`train/img_per_s` не растёт при второй GPU.** Скорость упирается в CPU. Если есть свободные ядра, увеличьте `data.workers`.
- **Нет интернета.** Скачайте веса заранее (см. «Веса»). Если кэш `~/.cache/huggingface` уже заполнен, достаточно `HF_HUB_OFFLINE=1`.
- **T4 или V100.** Оставьте `train.amp=false`: bf16 на них нет.
- **`transformers` 5.x.** Не поддерживается, нужна версия ниже 5.
