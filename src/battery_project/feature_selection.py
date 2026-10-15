"""Leakage-safe RFECV feature selection, run separately for SOH and SOC on training data only."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline


def drop_uninformative(X_train):
    """Drop all-missing and constant columns, judged on training rows only."""
    keep, dropped = [], {}
    for col in X_train.columns:
        s = X_train[col]
        if s.notna().sum() == 0:
            dropped[col] = "all missing in training data"
        elif s.nunique(dropna=True) <= 1:
            dropped[col] = "constant in training data"
        else:
            keep.append(col)
    return keep, dropped


def _run_rfecv(X_train, y_train, groups, estimator):
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("model", estimator)])
    rfecv = RFECV(pipe, step=1, cv=LeaveOneGroupOut(), scoring="neg_mean_absolute_error",
                  min_features_to_select=1, importance_getter="named_steps.model.feature_importances_",
                  n_jobs=1)
    rfecv.fit(X_train, y_train, groups=groups)
    cols = list(X_train.columns)
    ranking = pd.DataFrame({"feature": cols, "rfecv_rank": rfecv.ranking_, "selected": rfecv.support_})
    ranking = ranking.sort_values(["rfecv_rank", "feature"]).reset_index(drop=True)
    cv_curve = pd.DataFrame({
        "n_features": np.arange(1, len(rfecv.cv_results_["mean_test_score"]) + 1),
        "cv_mae_mean": -rfecv.cv_results_["mean_test_score"],
        "cv_mae_std": rfecv.cv_results_["std_test_score"],
    })
    selected = [c for c, s in zip(cols, rfecv.support_) if s]
    return selected, ranking, cv_curve


def select_soh_features(X_train, y_train, groups, random_state):
    """RFECV for SOH: modest random forest, leave-one-training-battery-out CV."""
    est = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=random_state, n_jobs=-1)
    return _run_rfecv(X_train, y_train, groups, est)


def select_soc_features(X_train, y_train, groups, random_state):
    """RFECV for SOC: smaller forest (many samples), leave-one-training-battery-out CV.
    Groups are batteries, so samples of one discharge segment never straddle a fold."""
    est = RandomForestRegressor(n_estimators=40, max_depth=14, min_samples_leaf=20,
                                random_state=random_state, n_jobs=-1)
    return _run_rfecv(X_train, y_train, groups, est)
