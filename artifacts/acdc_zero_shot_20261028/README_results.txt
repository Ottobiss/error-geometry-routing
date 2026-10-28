ACDC zero-shot: Cityscapes-trained SpatialRouter vs entropy

Основной вывод:
При бюджете 3 B5-патча роутер подтверждённо лучше entropy на ACDC all.

Строки domain=all, spatial B0 router:
domain  n_frames  B5_patches_per_frame            method      mIoU  delta_to_entropy     ci_lo    ci_hi
   all       406                     1 spatial B0 router 49.505143          0.420336 -0.578268 1.383862
   all       406                     2 spatial B0 router 52.456517          0.728134 -0.140275 1.690021
   all       406                     3 spatial B0 router 54.472683          0.966771  0.187824 1.835558

Правило применения:
B5 budget >= 3  -> spatial B0 router
B5 budget 1–2   -> entropy