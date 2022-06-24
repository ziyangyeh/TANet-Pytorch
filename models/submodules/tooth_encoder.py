from openpoints.models import build_model_from_cfg

def build_tooth_encoder(cfg, **kwargs):
    return build_model_from_cfg(cfg, **kwargs)
