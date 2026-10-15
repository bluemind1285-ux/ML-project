"""SOH path only: split, RFECV, three-model comparison, held-out test, figures, saved model."""
import numpy as np
import pandas as pd

from . import artifacts, evaluation, feature_selection, models, visualization
from .feature_engineering import SOH_FEATURES, SOH_TARGET

UNIT = "SOH percentage points"


def run(cfg):
    p = cfg["paths"]; rs = cfg["random_state"]
    tables, figs = p["results_dir"] / "tables", p["results_dir"] / "figures"
    path = p["processed_dir"] / "soh_features.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python scripts/prepare_data.py")
    df = pd.read_csv(path)

    n0 = len(df)
    df = df[np.isfinite(df[SOH_TARGET])].reset_index(drop=True)
    excluded_target = n0 - len(df)

    batteries = sorted(df["battery_id"].unique())
    test_b = cfg["soh"]["test_battery"]
    if test_b not in batteries:
        raise SystemExit(f"Configured SOH test battery {test_b} is not in the data ({batteries}). Edit config.yaml soh.test_battery.")
    train_b = [b for b in batteries if b != test_b]
    if len(train_b) < 2:
        raise SystemExit("Need at least two training batteries for grouped CV; only found " + str(train_b))

    train = df[df["battery_id"].isin(train_b)].reset_index(drop=True)
    test = df[df["battery_id"] == test_b].reset_index(drop=True)
    y_tr, y_te = train[SOH_TARGET], test[SOH_TARGET]
    g_tr = train["battery_id"]

    candidates, dropped = feature_selection.drop_uninformative(train[SOH_FEATURES])
    selected, ranking, cv_curve = feature_selection.select_soh_features(train[candidates], y_tr, g_tr, rs)
    ranking.to_csv(tables / "soh_rfecv_ranking.csv", index=False)
    cv_curve.to_csv(tables / "soh_rfecv_cv_curve.csv", index=False)
    pd.DataFrame({"selected_feature": selected}).to_csv(tables / "soh_selected_features.csv", index=False)
    visualization.rfecv_plot(cv_curve, ranking, "SOH feature selection (RFECV)", UNIT, figs / "soh_rfecv_selection.png")

    # Model choice from grouped CV on training batteries only.
    cv_results = {}
    for name in models.MODEL_NAMES:
        folds = evaluation.grouped_cv(models.make_pipeline(name, rs), train[selected], y_tr, g_tr)
        folds.insert(0, "model", name)
        cv_results[name] = folds
    pd.concat(cv_results.values()).to_csv(tables / "soh_cv_folds.csv", index=False)
    best = min(cv_results, key=lambda n: cv_results[n]["MAE"].mean())

    # Fixed protocol: fit each on all training batteries, evaluate once on the held-out battery.
    fitted, test_metrics, preds = {}, {}, test[["battery_id", "discharge_number", SOH_TARGET, "capacity_Ah"]].copy()
    for name in models.MODEL_NAMES:
        pipe = models.make_pipeline(name, rs).fit(train[selected], y_tr)
        fitted[name] = pipe
        yhat = pipe.predict(test[selected])
        preds[f"pred_{name}"] = yhat
        test_metrics[name] = evaluation.regression_metrics(y_te, yhat)
    preds.to_csv(tables / "soh_test_predictions.csv", index=False)
    comp = evaluation.comparison_table(cv_results, test_metrics, UNIT, best, train_b, [test_b])
    comp.to_csv(tables / "soh_model_comparison.csv", index=False)

    # Feature coverage (fraction of rows with a value) by battery.
    cov = df.groupby("battery_id")[SOH_FEATURES].apply(lambda g: g.notna().mean()).T
    cov.index.name = "feature"
    cov.to_csv(tables / "soh_feature_coverage.csv")

    # Figures
    pc = f"pred_{best}"
    visualization.actual_vs_predicted(
        preds[SOH_TARGET], preds[pc], preds["battery_id"],
        f"SOH held-out battery {test_b}: {best}", "SOH %", figs / "soh_actual_vs_predicted.png",
        caption=f"Model trained on {', '.join(train_b)}; {len(test)} cycles of {test_b} never used in selection or training.")
    visualization.actual_vs_predicted_all(
        preds, SOH_TARGET, models.MODEL_NAMES, best, test_metrics,
        f"SOH held-out battery {test_b}: all three models", "SOH %", figs / "soh_actual_vs_predicted_all_models.png",
        caption=f"Same {len(test)} held-out cycles for each model; units SOH percentage points. Model choice was made by grouped CV, not by these test results.")
    visualization.residual_plots(
        preds, SOH_TARGET, pc, "discharge_number", "Discharge cycle number", "reference - predicted", "SOH %",
        f"SOH residuals on held-out battery {test_b} ({best})", figs / "soh_residuals.png")
    visualization.model_comparison(comp, "SOH %", "SOH model comparison", figs / "soh_model_comparison.png")

    ranges = {c: {"min": float(train[c].min()), "max": float(train[c].max())} for c in selected}
    meta = {
        "target": SOH_TARGET,
        "target_definition": f"100 * measured discharge Capacity (Ah) / {cfg['rated_capacity_Ah']} Ah rated capacity (supplied README)",
        "unit": UNIT,
        "model_name": best,
        "feature_order": selected,
        "selected_features": selected,
        "candidate_features": SOH_FEATURES,
        "dropped_before_rfecv": dropped,
        "training_feature_ranges": ranges,
        "split": {"type": "battery holdout", "train_batteries": train_b, "test_batteries": [test_b],
                  "train_rows": len(train), "test_rows": len(test),
                  "train_cycles": {b: [int(train.loc[train.battery_id == b, "discharge_number"].min()),
                                       int(train.loc[train.battery_id == b, "discharge_number"].max())] for b in train_b},
                  "rows_excluded_nonfinite_target": excluded_target},
        "cv": "LeaveOneGroupOut over training batteries (used for RFECV and model choice)",
        "rfecv": {"estimator": "RandomForestRegressor(n_estimators=200, min_samples_leaf=2)", "scoring": "neg MAE", "step": 1},
        "selection_rule": "lowest mean grouped-CV MAE",
        "cv_metrics": {n: cv_results[n][["MAE", "RMSE", "R2"]].mean().to_dict() for n in cv_results},
        "test_metrics": test_metrics,
        "random_state": rs,
        "input_files": [f"{b}.mat" for b in batteries],
    }
    artifacts.save_model(fitted[best], meta, p["models_dir"] / "soh")
    return comp, best, selected
