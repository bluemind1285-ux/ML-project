"""Step 3: SOH selection, training, held-out evaluation, figures and saved model."""
import _bootstrap  # noqa: F401
import pandas as pd

from battery_project import soh_pipeline
from battery_project.config import ensure_dirs, load_config

if __name__ == "__main__":
    cfg = load_config(); ensure_dirs(cfg)
    comp, best, selected = soh_pipeline.run(cfg)
    pd.set_option("display.width", 200)
    print(f"Selected features ({len(selected)}): {selected}")
    print(comp[["model", "selected_by_cv", "cv_MAE_mean", "test_MAE", "test_RMSE", "test_R2", "test_n"]].to_string(index=False))
    print(f"Saved best model ({best}) to models/soh/")
