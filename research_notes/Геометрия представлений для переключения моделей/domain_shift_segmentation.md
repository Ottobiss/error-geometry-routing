# Оценки неопределённости и детекция ошибок сегментации под ковариатным сдвигом (Cityscapes → ACDC и др.): confidence vs. геометрия признаков

Контекст: SegFormer-B0 (Cityscapes) + роутер регионов в SegFormer-B2; тест на ACDC val (fog/night/rain/snow). Гипотеза: под сдвигом геометрия признаков (плотность, расстояние до прототипов, kNN) лучше энтропии предсказывает «B0 ошибся, а B2 был бы прав».

Пометка о проверке: источники с пометкой **[verified]** открыты/проверены в этой сессии (абстракт или текст). Пометка **[из базовых знаний]** — ссылка и суть известны мне по литературе, но в этой сессии не перепроверялись; писателю отчёта стоит относиться к деталям осторожно.

---

## 1. Калибровка и детекция ошибок сегментации под сдвигом: деградирует ли энтропия и деградируют ли меньше признаковые оценки?

### Takeaway
Калибровка софтмакса сегментационных сетей (включая SegFormer B0–B5) резко ухудшается при переходе Cityscapes → ACDC (относительный ECE растёт ~в 40 раз), но это НЕ означает, что ранжирование пикселей по энтропии/MSP разваливается: в большинстве бенчмарков детекции ошибок (failure detection) простой softmax-baseline остаётся лучшим или не хуже сложных методов, а признаковые (density/Mahalanobis) методы систематически сильнее на *детекции сдвига*, чем на *детекции ошибок*. Прямых работ «Mahalanobis/kNN vs. энтропия для детекции ошибок на ACDC» я не нашёл — это пробел, который проект может закрыть.

### Cited Findings
- de Jorge, Volpi, Torr, Rogez, «Reliability in Semantic Segmentation: Are We on the Right Track?», CVPR 2023, arXiv:2303.11298. Ключевая для проекта работа: ID = Cityscapes, OOD = IDD и ACDC (только 19 классов CS ⇒ чисто ковариатный сдвиг), упорядочение по силе сдвига CS << IDD << ACDC; модели: SETR, Segmenter, **SegFormer B0–B5**, Swin-L, ConvNeXt-L, ResNet-18/50/101 (DLV3+/UPerNet), Mask2Former. **[verified]** — [arXiv HTML](https://arxiv.org/html/2303.11298), [arXiv abs](https://arxiv.org/abs/2303.11298)
  - «all recent models are significantly more robust than well established baselines under natural shifts» — рост робастности (mIoU) у трансформеров реален. [arXiv](https://arxiv.org/html/2303.11298)
  - Но «OOD calibration error increases dramatically for all models»; относительный ECE растёт ≈40× от CS к ACDC; свежие модели «не значимо лучше откалиброваны», чем ResNet-бейзлайны; в аннотации: новые модели «not overall more reliable in terms of uncertainty estimation». [arXiv](https://arxiv.org/html/2303.11298)
  - Temperature scaling помогает in-domain, но даёт лишь маргинальный выигрыш OOD; **Local Temperature Scaling (LTS)** — «markedly the best calibration method out of domain, especially under stronger domain shifts», наибольший выигрыш на ACDC, и даёт резкий прирост PRR (детекция ошибок) на ACDC. [arXiv](https://arxiv.org/html/2303.11298)
  - Детекция ошибок (PRR): in-domain ResNet-модели лучше свежих; на ACDC свежие модели лучше и по mIoU, и по PRR; у ResNet-семейств вне домена корреляция mIoU–PRR отрицательная. [arXiv](https://arxiv.org/html/2303.11298)
  - Детекция OOD на уровне изображения: «OOD detection is inversely correlated with performance» — лучше всех маленький ResNet-18 (т.е. более робастная модель хуже «видит» сам сдвиг). [arXiv](https://arxiv.org/html/2303.11298)
  - Max prob vs. отрицательная энтропия как confidence — значимой разницы нет (Appendix C). MC dropout и ансамбли не оценивались; разбивки по условиям (ночь/туман) нет — только агрегат ACDC. [arXiv](https://arxiv.org/html/2303.11298)
- Cygert et al., «Closer Look at the Uncertainty Estimation in Semantic Segmentation under Distributional Shift», IJCNN 2021, arXiv:2106.00076: модель, почти идеально откалиброванная на источнике, под сдвигом резко теряет калибровку — ECE растёт с 1.49 до 9.55 для Cityscapes-модели на BDD; простые цветовые аугментации — сильный бейзлайн, сравнимый со style-transfer; ансамбль разных бэкбонов/аугментаций улучшает и точность, и неопределённость под сдвигом (ECE GTA→CS 4.05 → 1.1). **[verified: абстракт; число 1.49→9.55 — из поискового сниппета]** — [arXiv](https://arxiv.org/abs/2106.00076), [PDF](https://arxiv.org/pdf/2106.00076)
- Kahl et al., «ValUES: A Framework for Systematic Validation of Uncertainty Estimation in Semantic Segmentation», ICLR 2024 (oral), arXiv:2401.08501: разделение типов неопределённости работает на симулированных данных, но «не обязательно переносится» на реальные; **агрегация скоров (пиксель → изображение/регион) — ключевой и игнорируемый компонент**; ансамбли наиболее робастны по задачам, TTA — лёгкая альтернатива. Сдвиги в эксперименте: игрушечный датасет, LIDC-IDRI, GTA5/Cityscapes (не ACDC). **[verified: абстракт]** — [arXiv](https://arxiv.org/abs/2401.08501)
- Jaeger, Lüth, Klein, Bungert, «A Call to Reflect on Evaluation Practices for Failure Detection in Image Classification», ICLR 2023 (oral), arXiv:2211.15259: в широком бенчмарке confidence scoring functions по многим источникам ошибок (включая сдвиги) главный вывод — «simple softmax response baseline as the overall best performing method»; методы, проверяющие принадлежность к обучающему распределению, проигрывают на задаче детекции ошибок. (Классификация, не сегментация.) **[verified: абстракт]** — [arXiv](https://arxiv.org/abs/2211.15259)
- Сравнительный бенчмарк детекции ошибок в медицинской сегментации (arXiv:2406.03323, 2024): ID/OOD-разбиение само по себе плохо предсказывает, какие случаи провалятся — часть сдвинутых случаев сегментируется хорошо; роль агрегации confidence ключевая. **[verified: сниппет поиска]** — [arXiv](https://arxiv.org/pdf/2406.03323)
- Ovadia et al., «Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift», NeurIPS 2019, arXiv:1906.02530: качество неопределённости (калибровка) всех методов ухудшается с ростом силы сдвига; temperature scaling на ID не переносится на сдвиг; глубокие ансамбли наиболее устойчивы. (Классификация, ImageNet-C/CIFAR-C.) **[из базовых знаний]** — [arXiv](https://arxiv.org/abs/1906.02530)
- Minderer et al., «Revisiting the Calibration of Modern Neural Networks», NeurIPS 2021, arXiv:2106.07998: самые свежие модели, «notably those not using convolutions», — среди лучше откалиброванных; ухудшение калибровки под сдвигом и с размером модели у новых архитектур выражено слабее; архитектура — главный фактор. (Классификация.) **[verified: абстракт]** — [arXiv](https://arxiv.org/abs/2106.07998). Противоречит в части сегментации: de Jorge et al. 2023 не находят, что трансформеры-сегментаторы лучше откалиброваны под сдвигом — [arXiv](https://arxiv.org/html/2303.11298)
- Свежие работы по кросс-доменной калибровке сегментации: DA-Cal (arXiv:2602.20860, 2026) — попиксельные температуры, улучшения на SYNTHIA→CS и **CS→ACDC**; «Rethinking Post-Hoc Calibration in Semantic Segmentation» (arXiv:2607.01902, 2026) — эффекты сильнее под сильным сдвигом. Препринты, не проверены рецензированием. **[verified: сниппеты]** — [DA-Cal](https://arxiv.org/html/2602.20860), [Post-hoc](https://arxiv.org/html/2607.01902v1)
- Gamma-SSL (Williams et al., IEEE T-RO, arXiv:2402.17653): сегментационная сеть, обучаемая выдавать неопределённость через согласованность при аугментациях на неразмеченных данных целевых доменов, превосходит методы оценки неопределённости и OoD (до +10.7% AUROC, +19.2% AUPR в самом сложном сценарии) — на собственном бенчмарке SAX (городская → внедорожная среда), не на ACDC. **[verified: абстракт]** — [arXiv](https://arxiv.org/abs/2402.17653)
- MetaSeg (Rottmann et al., arXiv:1811.00648 и последующие): сегмент-уровневые метрики из дисперсии софтмакса (энтропия, margin, размер/форма сегмента) + мета-классификатор/регрессор IoU; на Cityscapes AUROC ≈89.9% для IoU=0 и R² ≈84.8% для регрессии IoU; в workshop-работе 2020 заявлено, что MetaSeg «надёжно» предсказывает IoU и для out-of-domain выборок. Проверки на туман/ночь/снег не найдено. **[verified: сниппеты]** — [arXiv 1811.00648](https://www.arxiv.org/pdf/1811.00648), [imacm 20-23](https://www.imacm.uni-wuppertal.de/fileadmin/imacm/preprints/2020/imacm_20_23.pdf), [arXiv 1904.04516](https://ar5iv.arxiv.org/html/1904.04516)
- MUAD (Franchi et al., «MUAD: Multiple Uncertainties for Autonomous Driving», BMVC 2022, arXiv:2203.01437): синтетический бенчмарк с туманом/дождём/ночью и OOD-объектами для оценки неопределённости сегментации; разделяет ковариатный и семантический сдвиг. **[из базовых знаний]** — [arXiv](https://arxiv.org/abs/2203.01437)

### Inferences
- Под сдвигом ломается в первую очередь **абсолютная калибровка** (уровень уверенности), а не обязательно **ранжирование** пикселей по энтропии. Для роутера с порогом/бюджетом (top-k% регионов) важно именно ранжирование; значит, «softmax overconfident» само по себе не гарантирует, что геометрия выиграет. Если порог фиксирован по Cityscapes — тогда калибровочный сдвиг бьёт по роутеру напрямую (рекомендация: бюджетный роутинг или перекалибровка порога, напр. LTS-подобная локальная температура).
- LTS (локальная, контент-зависимая температура) — это фактически признаковый сигнал, вмешивающийся в confidence; его выигрыш на ACDC — косвенное свидетельство, что локальный контекст/признаки несут информацию об ошибках, которой нет в глобальной энтропии. Это скорее аргумент за **гибрид** (энтропия + признаки), чем за замену.
- Результат de Jorge «детекция OOD обратно коррелирует с качеством» предупреждает: признаки, хорошо детектирующие *сдвиг*, могут плохо детектировать *ошибку* — особенно если B0 под сдвигом во многих пикселях всё ещё прав.

### Gaps
- Не найдено ни одной работы, сравнивающей энтропию с Mahalanobis/kNN/DDU-плотностью для **пиксельной детекции ошибок на ACDC** (по условиям). Также не найдено работ о роутинге «малая → большая модель» по признаковой геометрии под сдвигом.
- Нет разбивки калибровки/PRR по fog/night/rain/snow для SegFormer (de Jorge дают агрегат ACDC).
- Секция «uncertainty-aware semantic segmentation» (AUIoU) в статье ACDC не была дочитана — числа для бейзлайнов не получены.

---

## 2. Признаковые OOD-детекторы (плотность / Mahalanobis / kNN) при ковариатном сдвиге

### Takeaway
Доказательства в основном **против** гипотезы в её сильной форме: признаковые детекторы чувствительны к ковариатному сдвигу (хорошо *обнаруживают*, что вход сдвинут), но это не то же самое, что предсказать *ошибку*; в постановке failure detection/full-spectrum они проигрывают MSP или не лучше. Кроме того, Mahalanobis слаб на near-OOD и зависит от нормы признаков и «коллапса признаков». Почти все эти результаты — из классификации; в сегментации DDU-подобная плотность есть, но не проверена на погодном сдвиге.

### Cited Findings
- Yang, Zhang, Russakovsky, «ImageNet-OOD: Deciphering Modern Out-of-Distribution Detection Algorithms», ICLR 2024, arXiv:2310.01755: «OOD detectors are more sensitive to covariate shift than to semantic shift»; в постановке failure detection сложные детекторы «perform similarly to, or even worse than» MSP. **[verified: абстракт]** — [arXiv](https://arxiv.org/abs/2310.01755). По сниппету полного текста: при наличии ковариатного сдвига (ImageNet-C, -R, -Sketch) Mahalanobis и KNN опускаются до ≈31–44 AUROC (ниже случайного) в задаче отделения новых классов, т.е. ковариатный сдвиг «перебивает» сигнал; MSP лучше всех современных методов в failure-detection постановке. **[verified: сниппет]** — [PDF](https://arxiv.org/pdf/2310.01755)
- Zhang et al., «OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection», DMLR 2024 / NeurIPS 2023 workshop, arXiv:2306.09301: в full-spectrum постановке (ковариатно-сдвинутые ID-изображения считаются ID) все методы заметно хуже, чем в стандартной; ни один метод не лидирует стабильно. **[verified: сниппет/абстракт; конкретные числа по Mahalanobis/KNN не извлечены]** — [arXiv](https://arxiv.org/pdf/2306.09301)
- Yang et al., «Full-Spectrum Out-of-Distribution Detection», IJCV 2023, arXiv:2204.05306: вводит постановку, где ковариатный сдвиг должен *игнорироваться*; показывает, что большинство OOD-детекторов, включая признаковые, сильно деградируют, т.к. путают ковариатный сдвиг с семантическим. **[из базовых знаний; страница найдена в поиске]** — [DeepAI](https://deepai.org/publication/full-spectrum-out-of-distribution-detection), [HKBU](https://scholars.hkbu.edu.hk/en/publications/full-spectrum-out-of-distribution-detection/)
- Ren et al., «A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection», arXiv:2106.09022 (2021): ванильный Mahalanobis часто проваливается на near-OOD (CIFAR-100 vs CIFAR-10); Relative Mahalanobis (вычитание фоновой гауссианы) даёт до +15% AUROC. **[verified: сниппет]** — [arXiv](https://arxiv.org/pdf/2106.09022). Для проекта: погодный сдвиг ближе к «near»-режиму, Relative MD — дешёвая проверка.
- Mahalanobis++ (arXiv:2505.18032, 2025): норма признаков — сильный конфаундер, деградирующий Mahalanobis-детекторы; большая вариативность по моделям и схемам претрейна; ℓ2-нормализация признаков исправляет. **[verified: сниппет]** — [arXiv](https://arxiv.org/pdf/2505.18032)
- Mukhoti et al., «Deep Deterministic Uncertainty» (CVPR 2023; arXiv:2102.11582): плотность признаков (GDA) даёт хорошую эпистемическую неопределённость лишь при ограничениях на энкодер (residual + spectral normalization, чувствительность/гладкость); без них — **feature collapse** (разные входы схлопываются в одну точку признакового пространства), и методы Mahalanobis на обычном ResNet «уязвимы к коллапсу». **[verified: сниппет supplement; arXiv ID из базовых знаний]** — [CVPR supp](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Mukhoti_Deep_Deterministic_Uncertainty_CVPR_2023_supplemental.pdf)
- «Deep Deterministic Uncertainty for Semantic Segmentation» (Mukhoti, van Amersfoort, Torr, Gal; arXiv:2111.00079, UDL 2021): перенос DDU в сегментацию — GDA с одной гауссианой на класс по пикселям (пиксели как независимые выборки), плотность в один проход; оценка не на погодном сдвиге. **[verified: сниппет]** — [arXiv](https://arxiv.org/pdf/2111.00079)
- Kamoi & Kobayashi, «Why is the Mahalanobis Distance Effective for Anomaly Detection?» (arXiv:2003.00402): информация, критичная для классификации, не вносит вклад в качество детекции; одна общая гауссиана (без классов) почти так же эффективна. **[verified: сниппет]** — [arXiv](https://export.arxiv.org/abs/2003.00402v1). Для проекта: «расстояние до прототипа своего класса» может не давать больше, чем глобальная плотность.
- В CIFAR-10-C (классификация) Mahalanobis был сильнейшим бейзлайном детекции *коррумпированных входов*, но его AUROC растёт только с ростом severity (слабо на лёгких искажениях). (Lee & AlRegib, «Gradients as a Measure of Uncertainty», arXiv:2008.08030.) **[verified: сниппет]** — [arXiv](https://arxiv.org/pdf/2008.08030)
- SegmentMeIfYouCan (Chan et al., NeurIPS D&B 2021): MSP, ODIN, Mahalanobis, перенесённые из классификации, плохо обобщаются на аномальную/препятственную сегментацию — **это семантический OOD, не ковариатный**. **[verified: сниппет]** — [ResearchGate](https://www.researchgate.net/publication/351278473_SegmentMeIfYouCan_A_Benchmark_for_Anomaly_Segmentation)
- Медицинская сегментация (Covid-19 lesions, arXiv:2208.03217, 2022): лёгкий Mahalanobis в признаковом пространстве эффективно детектирует far- и near-OOD «silent failures» (уровень изображения). Позитивный пример, но другой домен и уровень изображения. **[verified: сниппет]** — [arXiv](https://arxiv.org/pdf/2208.03217)
- Lee et al. 2018 (Mahalanobis, arXiv:1807.03888) и Sun et al. 2022 (deep kNN, ICML, arXiv:2204.06507) — исходные методы; результаты в основном на семантическом OOD. **[из базовых знаний]** — [Mahalanobis](https://arxiv.org/abs/1807.03888), [kNN](https://arxiv.org/abs/2204.06507)

### Inferences
- Литература последовательно показывает: признаковые оценки — хорошие **детекторы сдвига**, но не лучшие **детекторы ошибок**. Под ACDC почти все пиксели «сдвинуты», поэтому сигнал «я далеко от Cityscapes» станет почти константой по кадру (особенно ночью) и потеряет дискриминативность внутри кадра. Важно оценивать AUROC/AURC *внутри условия* (ночь отдельно), а не на пуле CS+ACDC.
- Сигнал «B0 ошибся И B2 прав» ещё уже: если B2 использует похожие признаки, области, далёкие от CS в признаках B0, скорее всего далеки и для B2 — то есть геометрия может указывать на пиксели, где *обе* модели ошибаются (бесполезная маршрутизация). Это аргумент за обучение роутера на целевой метке «gain = 1[B2 прав ∧ B0 неправ]», а не на «B0 неправ».
- Если всё же пробовать геометрию: ℓ2-нормализация признаков, Relative Mahalanobis, нормализация по статистикам кадра (вычитание per-image среднего плотности) — минимальные меры против известных провалов.

### Gaps
- Нет найденных численных результатов Mahalanobis/kNN vs MSP на OpenOOD v1.5 full-spectrum (таблицы не извлечены).
- Нет работ, измеряющих feature collapse именно у SegFormer/MiT-энкодеров.

---

## 3. Робастность SegFormer (B0 vs B2/B5) в неблагоприятных условиях: где большая модель помогает?

### Takeaway
SegFormer известен высокой робастностью к Cityscapes-C, и трансформеры в целом заметно робастнее CNN на ACDC; самое тяжёлое условие для Cityscapes-моделей — ночь. Прямых опубликованных таблиц «B0 vs B2 zero-shot на ACDC по условиям и классам» я не нашёл; данные указывают, что выигрыш от масштаба есть, но малые модели теряют больше.

### Cited Findings
- Xie et al., «SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers», NeurIPS 2021, arXiv:2105.15203: B5 — 84.0% mIoU на Cityscapes val и «excellent zero-shot robustness on Cityscapes-C»; на 16 типах искажений SegFormer деградирует стабильно с ростом severity, в отличие от DeepLabV3+. Отчётные Cityscapes-C числа — для B5; данных для B0 в основном тексте нет. **[verified: сниппет]** — [arXiv](https://ar5iv.labs.arxiv.org/html/2105.15203), [NeurIPS](https://papers.nips.cc/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html)
- Sakaridis, Dai, Van Gool, «ACDC: The Adverse Conditions Dataset with Correspondences…», ICCV 2021, arXiv:2104.13395: Cityscapes-модели на ACDC (mIoU по fog/night/rain/snow): DeepLabv2 source-only 33.5 / 30.1 / 44.5 / 40.2; DeepLabv3+ (CS-pretrained) 45.7 / 25.0 / 50.0 / 42.0. Ночь — самый тяжёлый домен («the nighttime set of ACDC represents a harder domain»). Худшие классы в агрегате (DeepLabv2): wall 18.8, pole 19.7, rider 22.1, fence 22.5, sidewalk 26.2, motorcycle 26.2. **[verified]** — [arXiv HTML](https://arxiv.org/html/2104.13395v5)
- Обзор CTTA (arXiv:2607.08164, 2026), таблица Cityscapes→ACDC с SegFormer-B5: source-only 56.7 mIoU, BN Adapt 55.3, TENT 54.2, CoTTA 59.3, DAT 61.6; все методы сильнее всего деградируют на Night. **[verified: сниппет]** — [arXiv](https://arxiv.org/pdf/2607.08164)
- de Jorge et al. 2023: свежие модели (включая SegFormer) значимо робастнее ResNet-бейзлайнов на ACDC; у них же внутри семейства SegFormer B0–B5 оцениваются robust-метрики (агрегат). **[verified]** — [arXiv](https://arxiv.org/html/2303.11298)
- Global Average Feature Augmentation (arXiv:2412.01941, 2024): аугментация повышает робастность SegFormer, причём наибольший выигрыш — у малых/средних моделей B0–B3, что косвенно указывает на бо́льшую хрупкость малых вариантов. **[verified: сниппет]** — [arXiv](https://arxiv.org/pdf/2412.01941)

### Inferences
- Ожидаемо, разрыв B2−B0 будет максимален на ночи и на тонких/малых классах (pole, sign, rider, fence) и границах — это совпадает с классами, где Cityscapes-модели слабее всего на ACDC. Но на ночи и B2 часто будет неправ, поэтому «полезный» выигрыш маршрутизации может быть больше на fog/rain/snow, где B2 исправляет, а не тоже ошибается. Это надо измерять по oracle-gain картам в самом проекте.

### Gaps
- Не найдено опубликованных zero-shot ACDC mIoU для SegFormer-B0 и B2 по условиям и классам (только B5 source-only 56.7 в сводке CTTA). Требуется собственный замер.

---

## 4. Геометрия представлений под сдвигом: стиль vs. контент, нормализационные статистики, подпространства

### Takeaway
Есть сильная линия работ о том, что погодный/световой сдвиг в первую очередь проявляется в **статистиках первого/второго порядка признаков** (средние, дисперсии, ковариации каналов — «стиль»), а не в семантическом содержании. Это хорошо для *детекции сдвига на уровне кадра*, но говорит против того, что эти статистики локализуют *ошибочные пиксели*. TTA, основанная только на обновлении нормализационных статистик, в сегментации даёт мало или вредит.

### Cited Findings
- Обзор TTA-методов в сегментации («From Question to Exploration: Test-Time Adaptation in Semantic Segmentation?» / «A Critical Look at Classic TTA Methods in Semantic Segmentation», ACM MM 2024, arXiv:2310.05341): классическое обновление статистик нормализации даёт лишь небольшой прирост, а иногда вредит; batch renormalization проблему не решает; длиннохвостый дисбаланс классов — главный барьер. **[verified: сниппет]** — [arXiv](https://arxiv.org/pdf/2310.05341)
- Те же цифры с SegFormer-B5 CS→ACDC: BN Adapt (55.3) и TENT (54.2) хуже source-only (56.7); выигрывают CoTTA (59.3) и DAT (61.6), использующие weight-averaged teacher/аугментации. **[verified: сниппет]** — [arXiv:2607.08164](https://arxiv.org/pdf/2607.08164). Замечание: SegFormer использует LayerNorm, не BatchNorm, — статистики LN считаются по токену, поэтому «BN-статистики как индикатор сдвига» прямо не переносятся.
- Wang et al., CoTTA, «Continual Test-Time Domain Adaptation», CVPR 2022, arXiv:2203.13591: Cityscapes→ACDC (циклически повторяющиеся условия) с SegFormer-B5; использует усреднение по аугментациям и порог уверенности для решения, когда применять аугментационный teacher. **[из базовых знаний]** — [arXiv](https://arxiv.org/abs/2203.13591)
- Методы TTA на ACDC с попиксельной фильтрацией ненадёжных пикселей (PromptCAL, GaPaTTA, TU Graz) фильтруют по **энтропии Шеннона**, а не по признаковой геометрии. **[verified: сниппеты]** — [GaPaTTA](https://tugraz.elsevierpure.com/en/publications/gapatta-gaussian-entropy-guided-prompt-placement-for-test-time-ad/), [PromptCAL](https://tugraz.elsevierpure.com/en/publications/promptcal-entropy-calibrated-and-prompt-tunes-test-time-adaptatio/)
- ATASeg «Few Clicks Suffice: Active Test-Time Adaptation for Semantic Segmentation» (arXiv:2312.01835): SegFormer-B5 CS→ACDC, выбирает очень мало пикселей (b=16) для запроса меток — по сути задача «какие пиксели отправить оракулу», близкая к роутингу. **[verified: сниппет]** — [arXiv](https://arxiv.org/pdf/2312.01835)
- Pan et al., IBN-Net (ECCV 2018, arXiv:1807.09441) и Choi et al., RobustNet (CVPR 2021, arXiv:2103.15597): стиль/внешний вид (освещение, погода) кодируется в канальных средних/дисперсиях и ковариациях ранних слоёв; Instance Normalization и instance-selective whitening ковариаций удаляют доменно-специфичную информацию и улучшают обобщение на неизвестные домены (вкл. BDD, Mapillary). **[из базовых знаний]** — [IBN-Net](https://arxiv.org/abs/1807.09441), [RobustNet](https://arxiv.org/abs/2103.15597)
- WUDA (arXiv:2210.02088): мера «representation shift» между признаками источника и цели сильно коррелирует с mIoU на целевом домене в сегментации — признаковый сдвиг как предиктор *уровня датасета*. **[verified: сниппет]** — [arXiv](https://arxiv.org/pdf/2210.02088)
- Condition-Invariant Semantic Segmentation (CISS, arXiv:2305.17349): выравнивание внутренних признаков энкодера между исходным и стилизованным (под условие) изображением улучшает CS→ACDC/Dark Zurich — косвенно: доменный сдвиг сидит в признаках энкодера и отделим от контента. **[verified: только наличие в поиске; детали из базовых знаний]** — [arXiv](https://arxiv.org/html/2305.17349)

### Inferences
- Если сдвиг — в основном «стиль» (глобальные канальные статистики), то пиксельные геометрические скоры будут в основном двигаться **глобально по кадру**; дискриминативная внутри-кадровая информация может появиться лишь после **нормализации признаков по кадру** (instance-norm/whitening перед расстоянием до прототипов). Это конкретная, недорогая модификация для эксперимента.
- Признаковая геометрия, вероятно, полезнее как **сигнал уровня кадра** («этот кадр — сдвинутый, поднять бюджет B2»), чем как пиксельный селектор. Это согласуется с WUDA и с «OOD detectors sensitive to covariate shift».

### Gaps
- Не найдено работ об изменении внутренней размерности (intrinsic dimension) или доменных подпространств *в SegFormer/MiT* под погодным сдвигом.
- Не найдено работ, использующих LayerNorm-статистики трансформера как пиксельный индикатор ненадёжности.

---

## 5. Предсказание точности без меток (ATC, DoC, agreement-on-the-line) и регион-уровневые предикторы как сигнал роутера

### Takeaway
Методы оценки точности без меток (ATC, DoC, agreement-on-the-line) работают в основном на уровне датасета в классификации; переносов на сегментацию/ACDC почти нет. Наиболее близкие к роутеру регион-уровневые методы (MetaSeg) построены на признаках софтмакса, а не геометрии. Согласие двух моделей (B0 vs B2 или B0 vs аугментации) — теоретически обоснованный сигнал под сдвигом.

### Cited Findings
- Garg et al., «Leveraging Unlabeled Data to Predict Out-of-Distribution Performance» (ATC), ICLR 2022, arXiv:2201.04234: учится порог на confidence на источнике; точность цели = доля неразмеченных примеров с confidence выше порога; авторы подчёркивают, что без предположений о модели/характере сдвига никакой метод оценки точности не работает в общем случае. **[verified]** — [arXiv](https://arxiv.org/abs/2201.04234)
- Guillory et al., «Predicting with Confidence on Unseen Distributions» (DoC), ICCV 2021, arXiv:2107.03315: разница средней уверенности между доменами предсказывает падение точности. **[из базовых знаний]** — [arXiv](https://arxiv.org/abs/2107.03315)
- Baek et al., «Agreement-on-the-Line: Predicting the Performance of Neural Networks under Distribution Shift», NeurIPS 2022, arXiv:2206.13089: согласие между парами моделей ID vs OOD лежит на линии с тем же наклоном, что и точность ID vs OOD, что позволяет оценивать OOD-точность без меток. **[из базовых знаний]** — [arXiv](https://arxiv.org/abs/2206.13089)
- MetaSeg (Rottmann et al.): сегмент-уровневые признаки из софтмакса + регрессор IoU; заявлено надёжное предсказание IoU для out-of-domain выборок; расширение для false-negative reduction под доменным сдвигом с глубиной (arXiv:2207.03513). **[verified: сниппеты]** — [arXiv 1811.00648](https://www.arxiv.org/pdf/1811.00648), [arXiv 2207.03513](https://arxiv.org/pdf/2207.03513)
- ValUES (ICLR 2024): агрегация пиксельных скоров в изображение/регион — критичный, недооценённый компонент. **[verified]** — [arXiv](https://arxiv.org/abs/2401.08501)

### Inferences
- Для роутера естественный «ATC-аналог»: порог энтропии B0 калибруется на Cityscapes под целевую долю ошибок, а на ACDC доля пикселей выше порога даёт оценку ожидаемой ошибки кадра → адаптивный бюджет B2 на кадр. Это дешевле и обоснованнее, чем ждать, что геометрия выиграет пиксельно.
- Согласие B0 с дешёвыми вариантами самого себя (flip/scale-TTA) или с ранним выходом B2 — сигнал, теоретически связанный с ошибкой под сдвигом (agreement-on-the-line, ансамбли в Ovadia/ValUES), и часто сильнее софтмакса одной модели.
- Регион-уровневый мета-регрессор в стиле MetaSeg (энтропия + форма сегмента + признаковая плотность как доп. фичи) — наиболее реалистичный путь проверить, добавляет ли геометрия информацию *поверх* энтропии.

### Gaps
- Не найдено применения ATC/DoC/agreement-on-the-line к пиксельной/регион-уровневой сегментации на ACDC/Dark Zurich/BDD.
- Не найдено работ, предсказывающих именно «прирост от большей модели» (B2−B0) без меток — ближайшее — литература по каскадам/early-exit (не исследовалась в этом задании).

---

### Сводный вердикт по гипотезе (для писателя отчёта)
- **За:** калибровка софтмакса действительно сильно портится на ACDC (ECE ×~40, de Jorge 2023; ECE 1.49→9.55 на BDD, Cygert 2021); признаковые детекторы чувствительнее к ковариатному сдвигу, чем к семантическому (ImageNet-OOD, ICLR 2024); локальные/контент-зависимые поправки (LTS) существенно улучшают детекцию ошибок на ACDC.
- **Против:** в постановке *failure detection* (а не детекции сдвига) MSP/энтропия — лучший или неуступающий бейзлайн (Jaeger ICLR 2023; ImageNet-OOD ICLR 2024; OpenOOD v1.5 full-spectrum); Mahalanobis слаб на near-OOD и чувствителен к норме признаков/коллапсу; способность детектировать сдвиг обратно коррелирует с робастностью модели (de Jorge 2023); TTA-методы на ACDC фильтруют пиксели по энтропии, а не по геометрии.
- **Непроверено в литературе:** пиксельное сравнение энтропии и плотности/kNN для детекции ошибок SegFormer по условиям ACDC и критерий «B2 исправит». Это реальная новизна, но априорные ожидания скорее скептические; наиболее перспективно — геометрия как *дополнение* к энтропии и как сигнал уровня кадра.
