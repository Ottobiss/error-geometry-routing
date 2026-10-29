from features.base import Feature


class Entropy(Feature):
    """Mean per-pixel entropy of the light model's softmax."""

    def forward(self, out):
        logp = out.logits[0].log_softmax(0)
        return -(logp.exp() * logp).sum(0).mean()

