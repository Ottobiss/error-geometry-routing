"""Средний вектор декодера B0 по компоненте: вход classifier (256 каналов), поднятый до рабочего
разрешения и усреднённый по пикселям компоненты. Колонки e0 … e255."""
import torch


class Embeddings:
    def fit(self, ctx):
        pass                                   # подгонки нет

    def apply(self, fd):
        x = fd.z.T                                                     # (H·W, 256)
        s = torch.zeros(fd.n, x.shape[1], device=x.device).index_add_(0, fd.comp_t, x)
        k = torch.bincount(fd.comp_t, minlength=fd.n).clamp(min=1)
        mean = (s / k[:, None]).half().float().cpu().numpy()          # хранится с точностью fp16
        return {f"e{i}": mean[:, i] for i in range(mean.shape[1])}, {}
