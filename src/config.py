from omegaconf import OmegaConf


def load_config(path, overrides=()):
    """YAML config + dotted overrides (key.sub=value); unknown keys raise an error."""
    cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, True)
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
