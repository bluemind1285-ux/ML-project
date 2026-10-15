"""Metrics and grouped cross-validation shared by the SOH and SOC pipelines."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    n = len(y_true)
    r2 = r2_score(y_true, y_pred) if n >= 2 and np.var(y_true) > 0 else np.nan
    return {
        "n": n,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2,
    }


def grouped_cv(pipeline, X, y, groups):
    """Leave-one-group-out CV on training data. Returns per-fold metrics."""
    rows = []
    for tr, va in LeaveOneGroupOut().split(X, y, groups):
        p = clone(pipeline).fit(X.iloc[tr], y.iloc[tr])
        m = regression_metrics(y.iloc[va], p.predict(X.iloc[va]))
        m["held_out_group"] = str(pd.Series(groups).iloc[va].iloc[0])
        rows.append(m)
    return pd.DataFrame(rows)


def comparison_table(cv_results, test_results, unit, best_name, train_groups, test_groups):
    rows = []
    for name in cv_results:
        cv = cv_results[name]; te = test_results[name]
        rows.append({
            "model": name,
            "selected_by_cv": name == best_name,
            "cv_MAE_mean": cv["MAE"].mean(), "cv_RMSE_mean": cv["RMSE"].mean(), "cv_R2_mean": cv["R2"].mean(),
            "test_MAE": te["MAE"], "test_RMSE": te["RMSE"], "test_R2": te["R2"], "test_n": te["n"],
            "unit": unit,
            "train_groups": ",".join(train_groups), "test_groups": ",".join(test_groups),
        })
    return pd.DataFrame(rows)
