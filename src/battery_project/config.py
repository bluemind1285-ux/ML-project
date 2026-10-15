"""Project paths and settings loaded from config.yaml."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key, value in cfg["paths"].items():
        if key == "source_dirs":
            cfg["paths"][key] = [ROOT / p for p in value]
        else:
            cfg["paths"][key] = ROOT / value
    return cfg


def ensure_dirs(cfg):
    p = cfg["paths"]
    for d in [p["raw_dir"], p["processed_dir"], p["models_dir"] / "soh", p["models_dir"] / "soc",
              p["results_dir"] / "tables", p["results_dir"] / "figures"]:
        d.mkdir(parents=True, exist_ok=True)
