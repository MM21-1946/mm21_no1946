from .tan import TAN
from .buttomup import LGI, CMIN, FIAN, CSMGAN
ARCHITECTURES = {
    "TAN": TAN,
    "LGI": LGI,
    "FIAN": FIAN,
    "CSMGAN": CSMGAN,
    "CMIN": CMIN,
}

def build_model(cfg, dataset_train=None):
    return ARCHITECTURES[cfg.MODEL.ARCHITECTURE](cfg, dataset_train)
