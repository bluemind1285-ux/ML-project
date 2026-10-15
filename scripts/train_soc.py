"""Step 4: SOC reference assessment; trains SOC models only if the reference is defensible."""
import _bootstrap  # noqa: F401
import pandas as pd

from battery_project import soc_pipeline
from battery_project.config import ensure_dirs, load_config

if __name__ == "__main__":
    cfg = load_config(); ensure_dirs(cfg)
    out = soc_pipeline.run(cfg)
    if out is not None:
        comp, best, selected = out
        pd.set_option("display.width", 200)
        print(f"Selected features ({len(selected)}): {selected}")
        print(comp[["model", "selected_by_cv", "cv_MAE_mean", "test_MAE", "test_RMSE", "test_R2", "test_n"]].to_string(index=False))
        print(f"Saved best model ({best}) to models/soc/; assessment in results/soc_assessment.md")
