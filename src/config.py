from omegaconf import OmegaConf


def load_config(path, overrides=()):
    """YAML config, then overrides in order: more YAML files (e.g. a stage on top of train.yaml) and dotted
    key=value pairs. Unknown keys raise an error."""
    cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, True)
    for item in overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([item]) if "=" in item else OmegaConf.load(item))
    return cfg
