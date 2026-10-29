import torch
import torch.nn.functional as F

from features.base import Feature


class KNN(Feature):
    """Mean cosine distance of the decoder features (1/8 resolution) to their K nearest neighbours in a bank
    of features sampled from clean frames: high when the frame does not look like the fit data."""
    taps = ("decoder",)

    def __init__(self, bank_size=20000, per_frame=500, k=10, seed=0):
        super().__init__()
        self.bank_size, self.per_frame, self.k = bank_size, per_frame, k
        self.gen = torch.Generator().manual_seed(seed)
        self.samples = []
        self.register_buffer("bank", torch.empty(0))  # (bank_size, channels), fp16, unit norm; size set by the fit
        self.register_load_state_dict_pre_hook(
            lambda m, state, prefix, *_: setattr(m, "bank", torch.empty_like(state[prefix + "bank"], device=m.bank.device)))

    @staticmethod
    def vectors(out):
        f = F.avg_pool2d(out.taps["decoder"][0].float(), 2)  # 1/4 -> 1/8
        return F.normalize(f.flatten(1).T, dim=1)

    def update(self, out):
        v = self.vectors(out).cpu()
        self.samples.append(v[torch.randperm(len(v), generator=self.gen)[:self.per_frame]])

    def finish(self):
        v = torch.cat(self.samples)
        self.bank = v[torch.randperm(len(v), generator=self.gen)[:self.bank_size]].to(self.bank.device, torch.half)
        self.samples = []

    def forward(self, out):
        sim = (self.vectors(out).to(self.bank) @ self.bank.T).topk(self.k, dim=1).values.float()
        return (1 - sim.mean(1)).mean()

