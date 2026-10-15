"""Save/load fitted pipelines and JSON metadata.

joblib files execute code when loaded: only load artifacts produced by this project.
"""
import json
import platform
from pathlib import Path

import joblib
import numpy as np


def library_versions():
    import pandas, scipy, sklearn
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pandas.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "joblib": joblib.__version__}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return o


def save_model(pipeline, metadata, model_dir):
    model_dir = Path(model_dir); model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / "model.joblib")
    metadata = dict(metadata); metadata["library_versions"] = library_versions()
    with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(metadata), f, indent=2)


def load_model(model_dir):
    model_dir = Path(model_dir)
    with open(model_dir / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    return joblib.load(model_dir / "model.joblib"), meta


def write_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(obj), f, indent=2, default=str)
