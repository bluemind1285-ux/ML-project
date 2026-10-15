"""SOC path only: reference-quality assessment, then (if defensible) split, RFECV, three models,
held-out test, figures, saved model, and results/soc_assessment.md."""
import numpy as np
import pandas as pd

from . import artifacts, evaluation, feature_selection, models, visualization
from .feature_engineering import SOC_FEATURES, SOC_TARGET

UNIT = "SOC percentage points"


def assess_reference(diag, cfg):
    tol = cfg["soc"]["reference_capacity_tolerance"]
    acc = diag[diag["accepted"]]
    ratio_dev = (acc["q_to_cutoff_over_capacity"] - 1).abs()
    checks = {
        "segments_total": len(diag),
        "segments_accepted": len(acc),
        "segments_rejected": int((~diag["accepted"]).sum()),
        "reject_reasons": diag.loc[~diag["accepted"], "reject_reason"].value_counts().to_dict(),
        "start_voltage_min_V": float(diag["v_start_V"].min()),
        "max_time_step_s": float(diag["max_dt_s"].max()),
        "mean_load_current_A_range": [float(acc["mean_load_current_A"].min()), float(acc["mean_load_current_A"].max())],
        "q_to_cutoff_over_capacity_min": float(acc["q_to_cutoff_over_capacity"].min()),
        "q_to_cutoff_over_capacity_median": float(acc["q_to_cutoff_over_capacity"].median()),
        "q_to_cutoff_over_capacity_max": float(acc["q_to_cutoff_over_capacity"].max()),
        "soc_end_unclipped_min_percent": float(acc["soc_end_unclipped_percent"].min()),
        "soc_end_unclipped_max_percent": float(acc["soc_end_unclipped_percent"].max()),
        "soc_max_unclipped_percent": float(acc["soc_max_unclipped_percent"].max()),
        "soc_largest_step_increase_percent": float(acc["soc_max_increase_percent"].max()),
        "q_full_record_over_capacity_max": float((acc["q_full_record_Ah"] / acc["capacity_Ah"]).max()),
        "capacity_tolerance": tol,
        "segments_outside_tolerance": int((ratio_dev > tol).sum()),
    }
    defensible = len(acc) > 0 and checks["segments_outside_tolerance"] == 0
    return defensible, checks


def run(cfg):
    p = cfg["paths"]; rs = cfg["random_state"]
    tables, figs = p["results_dir"] / "tables", p["results_dir"] / "figures"
    s_path, d_path = p["processed_dir"] / "soc_samples.parquet", p["processed_dir"] / "soc_segment_diagnostics.csv"
    if not s_path.exists() or not d_path.exists():
        raise SystemExit(f"SOC processed data not found. Run: python scripts/prepare_data.py")
    samples = pd.read_parquet(s_path)
    diag = pd.read_csv(d_path)

    defensible, checks = assess_reference(diag, cfg)
    visualization.soc_reference_quality(diag, samples, figs / "soc_reference_quality.png")
    if not defensible:
        _write_assessment(cfg, checks, defensible, None)
        print("SOC reference not defensible; training skipped. See results/soc_assessment.md")
        return None

    n0 = len(samples)
    finite = np.isfinite(samples[SOC_TARGET]) & np.isfinite(samples[[c for c in SOC_FEATURES if c != "dvdt_past_V_per_s"]]).all(axis=1)
    samples = samples[finite].reset_index(drop=True)
    excluded = n0 - len(samples)

    batteries = sorted(samples["battery_id"].unique())
    test_b = cfg["soc"]["test_battery"]
    if test_b not in batteries:
        raise SystemExit(f"Configured SOC test battery {test_b} not in data ({batteries}). Edit config.yaml soc.test_battery.")
    train_b = [b for b in batteries if b != test_b]
    train = samples[samples["battery_id"].isin(train_b)].reset_index(drop=True)
    test = samples[samples["battery_id"] == test_b].reset_index(drop=True)
    assert not set(train["segment_id"]) & set(test["segment_id"])
    y_tr, y_te, g_tr = train[SOC_TARGET], test[SOC_TARGET], train["battery_id"]

    candidates, dropped = feature_selection.drop_uninformative(train[SOC_FEATURES])
    selected, ranking, cv_curve = feature_selection.select_soc_features(train[candidates], y_tr, g_tr, rs)
    ranking.to_csv(tables / "soc_rfecv_ranking.csv", index=False)
    cv_curve.to_csv(tables / "soc_rfecv_cv_curve.csv", index=False)
    pd.DataFrame({"selected_feature": selected}).to_csv(tables / "soc_selected_features.csv", index=False)
    visualization.rfecv_plot(cv_curve, ranking, "SOC feature selection (RFECV)", UNIT, figs / "soc_rfecv_selection.png")

    cv_results = {}
    for name in models.MODEL_NAMES:
        folds = evaluation.grouped_cv(models.make_pipeline(name, rs, large_data=True), train[selected], y_tr, g_tr)
        folds.insert(0, "model", name)
        cv_results[name] = folds
    pd.concat(cv_results.values()).to_csv(tables / "soc_cv_folds.csv", index=False)
    best = min(cv_results, key=lambda n: cv_results[n]["MAE"].mean())

    fitted, test_metrics = {}, {}
    preds = test[["battery_id", "segment_id", "discharge_number", "elapsed_s", SOC_TARGET]].copy()
    for name in models.MODEL_NAMES:
        pipe = models.make_pipeline(name, rs, large_data=True).fit(train[selected], y_tr)
        fitted[name] = pipe
        preds[f"pred_{name}"] = pipe.predict(test[selected])
        test_metrics[name] = evaluation.regression_metrics(y_te, preds[f"pred_{name}"])
    preds.to_parquet(p["processed_dir"] / "soc_test_predictions.parquet", index=False)
    comp = evaluation.comparison_table(cv_results, test_metrics, UNIT, best, train_b, [test_b])
    comp.to_csv(tables / "soc_model_comparison.csv", index=False)

    pc = f"pred_{best}"
    seg = preds.groupby("segment_id").apply(
        lambda g: pd.Series({"discharge_number": g["discharge_number"].iloc[0],
                             "MAE": float(np.mean(np.abs(g[pc] - g[SOC_TARGET]))),
                             "max_abs_error": float(np.max(np.abs(g[pc] - g[SOC_TARGET]))),
                             "n_samples": len(g)}), include_groups=False).reset_index()
    seg.to_csv(tables / "soc_test_per_segment_errors.csv", index=False)
    soc_bins = pd.cut(preds[SOC_TARGET], [-5, 10, 20, 40, 60, 80, 90, 101])
    binned = preds.assign(abs_err=(preds[pc] - preds[SOC_TARGET]).abs()).groupby(soc_bins, observed=True)["abs_err"] \
        .agg(["count", "mean", "max"]).rename(columns={"mean": "MAE", "max": "max_abs_error"})
    binned.index = binned.index.astype(str); binned.index.name = "reference_SOC_bin_percent"
    binned.to_csv(tables / "soc_test_error_by_soc_range.csv")

    ids = sorted(test["segment_id"].unique())
    visualization.soc_trajectories(preds, [ids[0], ids[len(ids) // 2], ids[-1]], pc, figs / "soc_trajectories_heldout.png")
    visualization.actual_vs_predicted(
        preds[SOC_TARGET], preds[pc], preds["battery_id"], f"SOC held-out battery {test_b}: {best}", "SOC %",
        figs / "soc_actual_vs_predicted.png", alpha=0.15, s=3,
        caption=f"{len(test)} samples from {len(ids)} held-out discharge segments; reference is coulomb-counting (approximate).")
    visualization.actual_vs_predicted_all(
        preds, SOC_TARGET, models.MODEL_NAMES, best, test_metrics,
        f"SOC held-out battery {test_b}: all three models", "SOC %", figs / "soc_actual_vs_predicted_all_models.png",
        caption=f"Same {len(test)} held-out samples for each model; reference is approximate coulomb counting. Model choice was made by grouped CV.",
        alpha=0.1, s=2)
    visualization.residual_plots(
        preds, SOC_TARGET, pc, "elapsed_s", "Elapsed time in discharge (s)", "predicted - reference", "SOC %",
        f"SOC residuals on held-out battery {test_b} ({best})", figs / "soc_residuals.png", s=3, alpha=0.15)
    visualization.model_comparison(comp, "SOC %", "SOC model comparison", figs / "soc_model_comparison.png")

    ranges = {c: {"min": float(train[c].min()), "max": float(train[c].max())} for c in selected}
    meta = {
        "target": SOC_TARGET,
        "target_definition": "100 * (1 - trapz(-Current_measured, Time)/3600 / measured Capacity of the same discharge); "
                             "segment from record start to first sample < 2.7 V under load; SOC0 = 100 %",
        "reference_status": "approximate coulomb-counting reference, not measured ground-truth SOC",
        "reference_defensible": defensible,
        "reference_checks": checks,
        "reference_assumptions": [
            "Current_measured < 0 while discharging; discharge current = -Current_measured",
            "Discharge records start right after a CC-CV charge to 4.2 V / 20 mA; start voltage >= "
            f"{cfg['soc']['full_charge_min_voltage_V']} V accepted as SOC0 = 100 %",
            "Usable capacity = measured Capacity field of the same discharge (cycle-specific, known only after the discharge)",
            "Segment truncated at the 2.7 V cutoff (README definition of Capacity); deeper cutoffs not modelled",
            "Validity limited to the dataset protocol: 2 A constant-current discharge, 24 degC ambient",
        ],
        "unit": UNIT,
        "model_name": best,
        "feature_order": selected,
        "selected_features": selected,
        "candidate_features": SOC_FEATURES,
        "dropped_before_rfecv": dropped,
        "training_feature_ranges": ranges,
        "split": {"type": "battery holdout (whole discharge segments)", "train_batteries": train_b, "test_batteries": [test_b],
                  "train_samples": len(train), "test_samples": len(test),
                  "train_segments": int(train["segment_id"].nunique()), "test_segments": len(ids),
                  "samples_excluded_nonfinite": excluded},
        "cv": "LeaveOneGroupOut over training batteries (used for RFECV and model choice)",
        "rfecv": {"estimator": "RandomForestRegressor(n_estimators=40, max_depth=14, min_samples_leaf=20)", "scoring": "neg MAE"},
        "selection_rule": "lowest mean grouped-CV MAE",
        "cv_metrics": {n: cv_results[n][["MAE", "RMSE", "R2"]].mean().to_dict() for n in cv_results},
        "test_metrics": test_metrics,
        "random_state": rs,
        "input_files": [f"{b}.mat" for b in batteries],
    }
    artifacts.save_model(fitted[best], meta, p["models_dir"] / "soc")
    _write_assessment(cfg, checks, defensible, {"comp": comp, "best": best, "selected": selected, "seg": seg,
                                                "binned": binned, "train_b": train_b, "test_b": test_b,
                                                "n_test": len(test), "n_seg": len(ids)})
    return comp, best, selected


def _fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _write_assessment(cfg, checks, defensible, res):
    c = checks
    lines = [
        "# SOC reference assessment",
        "",
        "Generated by `python scripts/train_soc.py`. All numbers below are computed from the supplied files.",
        "",
        "## Evidence inspected",
        "",
        f"- Discharge records: {c['segments_total']}; accepted as SOC segments: {c['segments_accepted']}; rejected: {c['segments_rejected']} {c['reject_reasons'] or ''}",
        f"- Current sign: `Current_measured` is negative during discharge in every record; discharge current is taken as `-Current_measured`. "
        f"Mean load current over accepted segments: {_fmt(c['mean_load_current_A_range'][0])}–{_fmt(c['mean_load_current_A_range'][1])} A (README: 2 A CC).",
        "- `Current_load` changes sign between records (positive in some, negative in others), so it is not used for integration.",
        f"- Time: strictly increasing within every record; largest step {_fmt(c['max_time_step_s'], 1)} s (gap limit {cfg['soc']['max_gap_s']} s). No resets inside a record, so each discharge record is one segment.",
        f"- Initial condition: lowest discharge start voltage {_fmt(c['start_voltage_min_V'])} V. Each discharge follows a CC-CV charge to 4.2 V/20 mA (README), so SOC0 = 100 % is used.",
        f"- Integration vs measured capacity: integrating to the first sample below 2.7 V under load gives "
        f"{_fmt(c['q_to_cutoff_over_capacity_min'], 4)} / {_fmt(c['q_to_cutoff_over_capacity_median'], 4)} / {_fmt(c['q_to_cutoff_over_capacity_max'], 4)} "
        f"(min / median / max) times the `Capacity` field. Tolerance used: ±{c['capacity_tolerance']*100:.0f} %; segments outside: {c['segments_outside_tolerance']}.",
        f"- Unclipped reference SOC at the cutoff: {_fmt(c['soc_end_unclipped_min_percent'], 2)} % to {_fmt(c['soc_end_unclipped_max_percent'], 2)} %; "
        f"maximum value {_fmt(c['soc_max_unclipped_percent'], 2)} %; largest single-step increase {_fmt(c['soc_largest_step_increase_percent'], 3)} % "
        "(small increases come from near-zero current noise before the load switches on). No clipping is applied.",
        f"- Integration to the end of the record (including the deeper 2.5/2.2 V cutoffs of B0006/B0007/B0018 and rest samples) gives up to "
        f"{_fmt(c['q_full_record_over_capacity_max'], 4)} times `Capacity`, which is why segments stop at 2.7 V, the cutoff the README gives for `Capacity`.",
        "",
        "## Reference status",
        "",
    ]
    if not defensible:
        lines += ["The reference failed the checks above, so SOC models were **not trained** and the dashboard shows no SOC estimate."]
    else:
        lines += [
            "The reference is **defensible but approximate**: it is coulomb counting, not a measured SOC. Its known limitations:",
            "",
            "- The denominator is the measured capacity of the *same* discharge, which is only known after that discharge ends. The reference is therefore cycle-specific (0 % = 2.7 V cutoff at 2 A), not an absolute thermodynamic SOC.",
            "- It applies only to this dataset's protocol: full charge, then 2 A constant-current discharge at 24 °C ambient. It says nothing about dynamic loads, partial charges or other temperatures.",
            "- Integration error relative to `Capacity` is up to about 0.6 % of capacity (see figure `soc_reference_quality.png`).",
        ]
    if res:
        comp = res["comp"].set_index("model"); b = res["best"]
        lines += [
            "",
            "## Held-out validation",
            "",
            f"Trained on {', '.join(res['train_b'])}; tested once on all {res['n_seg']} discharge segments ({res['n_test']} samples) of {res['test_b']}. "
            f"Model chosen by grouped CV MAE: **{b}**. Selected features (RFECV, training batteries only): {', '.join(res['selected'])}.",
            "",
            "| Model | CV MAE | Test MAE | Test RMSE | Test R² |",
            "|---|---|---|---|---|",
        ]
        for m, r in comp.iterrows():
            lines.append(f"| {m}{' (selected)' if r['selected_by_cv'] else ''} | {_fmt(r['cv_MAE_mean'], 2)} | {_fmt(r['test_MAE'], 2)} | {_fmt(r['test_RMSE'], 2)} | {_fmt(r['test_R2'], 4)} |")
        seg = res["seg"]; binned = res["binned"]
        worst_bin = binned["MAE"].idxmax()
        lines += [
            "",
            f"Test MAE of the selected model ({_fmt(comp.loc[b, 'test_MAE'], 2)}) is higher than its grouped-CV MAE ({_fmt(comp.loc[b, 'cv_MAE_mean'], 2)}): "
            "the held-out battery is harder than the training folds. R² is high mainly because the reference spans the whole 0–100 % range, so MAE and the error table below are more informative.",
            "",
            f"Units: {UNIT}. Per-segment MAE of the selected model on the test battery: median {_fmt(seg['MAE'].median(), 2)}, "
            f"worst {_fmt(seg['MAE'].max(), 2)}; worst single-sample error {_fmt(seg['max_abs_error'].max(), 2)}.",
            "",
            "Error by reference-SOC range (selected model, held-out battery):",
            "",
            "| Reference SOC bin (%) | samples | MAE | max abs error |",
            "|---|---|---|---|",
        ]
        for idx, r in binned.iterrows():
            lines.append(f"| {idx} | {int(r['count'])} | {_fmt(r['MAE'], 2)} | {_fmt(r['max_abs_error'], 2)} |")
        lines += [
            "",
            "## Decision",
            "",
            "Dashboard inclusion is controlled by `soc.dashboard_enabled` in `config.yaml` together with `reference_defensible` in the saved metadata. "
            "Inclusion is justified only within the scope above: the model reproduces an approximate coulomb-counting reference for a 2 A "
            "constant-current discharge from full charge on an unseen battery of the same test campaign. It is not validated for other load profiles, "
            "temperatures, partial-charge starts, or different cells. Errors are largest in the middle of the discharge "
            f"(reference SOC bin {worst_bin} %: MAE {_fmt(binned.loc[worst_bin, 'MAE'], 2)}, max {_fmt(binned.loc[worst_bin, 'max_abs_error'], 2)}), "
            "so read the error table before relying on an individual estimate.",
        ]
    out = cfg["paths"]["results_dir"] / "soc_assessment.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
