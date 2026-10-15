"""Figures for data understanding and validation. All saved as PNG under results/figures."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"B0005": "#1f77b4", "B0006": "#ff7f0e", "B0007": "#2ca02c", "B0018": "#d62728"}


def _color(bid):
    return COLORS.get(bid, "#555555")


def _save(fig, path, caption=None):
    if caption:
        fig.text(0.01, 0.005, caption, fontsize=8, color="#444444", ha="left", va="bottom", wrap=True)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def capacity_vs_cycle(soh, rated, eol, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for bid, g in soh.groupby("battery_id"):
        ax.plot(g["discharge_number"], g["capacity_Ah"], ".-", ms=3, lw=0.8, color=_color(bid), label=bid)
    ax.axhline(rated, color="k", ls=":", lw=1, label=f"Rated {rated} Ah (README)")
    ax.axhline(eol, color="grey", ls="--", lw=1, label=f"EOL {eol} Ah (README criterion)")
    ax.set(xlabel="Discharge cycle number", ylabel="Measured discharge capacity (Ah)",
           title="Measured discharge capacity vs cycle")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, path, "Measured values from the Capacity field of each discharge record.")


def soh_vs_cycle(soh, eol_pct, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for bid, g in soh.groupby("battery_id"):
        ax.plot(g["discharge_number"], g["SOH_percent"], ".-", ms=3, lw=0.8, color=_color(bid), label=bid)
    ax.axhline(100, color="k", ls=":", lw=1, label="100 %")
    ax.axhline(eol_pct, color="grey", ls="--", lw=1, label=f"EOL {eol_pct:.0f} % (README criterion)")
    ax.set(xlabel="Discharge cycle number", ylabel="SOH (%)", title="Reference SOH vs cycle (derived from measured capacity)")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, path, "SOH = 100 x capacity / 2.0 Ah. Unclipped: values above 100 % are shown as measured.")


def actual_vs_predicted(y, yhat, labels, title, unit, path, caption=None, alpha=0.8, s=12):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for lab in sorted(set(labels)):
        m = np.asarray(labels) == lab
        ax.scatter(np.asarray(y)[m], np.asarray(yhat)[m], s=s, alpha=alpha, color=_color(lab), label=lab)
    lo = float(min(np.min(y), np.min(yhat))); hi = float(max(np.max(y), np.max(yhat)))
    pad = 0.03 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, label="Identity (y = x)")
    ax.set(xlabel=f"Reference ({unit})", ylabel=f"Predicted ({unit})", title=title)
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, path, caption)


def actual_vs_predicted_all(preds, ref_col, model_names, best, metrics, title, unit, path, caption=None, alpha=0.8, s=12):
    """One panel per model on the same held-out data and the same axes."""
    y = preds[ref_col].to_numpy()
    allv = np.concatenate([y] + [preds[f"pred_{m}"].to_numpy() for m in model_names])
    lo, hi = float(allv.min()), float(allv.max()); pad = 0.03 * (hi - lo)
    fig, axes = plt.subplots(1, len(model_names), figsize=(5.2 * len(model_names), 5.2), sharex=True, sharey=True)
    for ax, m in zip(np.atleast_1d(axes), model_names):
        for lab in sorted(preds["battery_id"].unique()):
            g = preds[preds["battery_id"] == lab]
            ax.scatter(g[ref_col], g[f"pred_{m}"], s=s, alpha=alpha, color=_color(lab), label=lab)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, label="Identity (y = x)")
        mt = metrics[m]
        ax.set_title(f"{m}{' (selected by CV)' if m == best else ''}\n"
                     f"MAE {mt['MAE']:.2f}, RMSE {mt['RMSE']:.2f}, R² {mt['R2']:.3f}", fontsize=10)
        ax.set_xlabel(f"Reference ({unit})"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel(f"Predicted ({unit})")
    fig.suptitle(title)
    _save(fig, path, caption)


def residual_plots(pred, ref_col, pred_col, x2_col, x2_label, sign_label, unit, title, path, caption=None, s=12, alpha=0.8):
    """Residual = reference - predicted (SOH) or as given by sign_label."""
    res = pred[ref_col] - pred[pred_col] if sign_label.startswith("reference") else pred[pred_col] - pred[ref_col]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for bid, g in pred.assign(res=res).groupby("battery_id"):
        axes[0].scatter(g[pred_col], g["res"], s=s, alpha=alpha, color=_color(bid), label=bid)
        axes[1].scatter(g[x2_col], g["res"], s=s, alpha=alpha, color=_color(bid), label=bid)
    for ax, xl in zip(axes, [f"Predicted ({unit})", x2_label]):
        ax.axhline(0, color="k", lw=1)
        ax.set(xlabel=xl, ylabel=f"Residual = {sign_label} ({unit})")
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle(title)
    _save(fig, path, caption)


def model_comparison(table, unit, title, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    x = np.arange(len(table)); w = 0.38
    for ax, metric, better in zip(axes, ["MAE", "RMSE", "R2"], ["lower is better", "lower is better", "higher is better"]):
        ax.bar(x - w / 2, table[f"cv_{metric}_mean"], w, label="Grouped CV mean (training batteries)", color="#9ecae1")
        ax.bar(x + w / 2, table[f"test_{metric}"], w, label="Held-out test battery", color="#3182bd")
        ax.set_xticks(x); ax.set_xticklabels(table["model"], rotation=10)
        ax.set_title(f"{metric} ({better})")
        ax.set_ylabel(metric if metric == "R2" else f"{metric} ({unit})")
        ax.axhline(0, color="k", lw=0.6); ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    _save(fig, path, "Model choice was made from the CV columns only; test metrics are reported once for comparison.")


def rfecv_plot(cv_curve, ranking, title, unit, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.2]})
    axes[0].errorbar(cv_curve["n_features"], cv_curve["cv_mae_mean"], yerr=cv_curve["cv_mae_std"], marker="o", capsize=3)
    n_sel = int(ranking["selected"].sum())
    axes[0].axvline(n_sel, color="r", ls="--", label=f"Selected: {n_sel} features")
    axes[0].set(xlabel="Number of features", ylabel=f"Grouped CV MAE ({unit})", title="RFECV curve (training batteries only)")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    r = ranking.iloc[::-1]
    axes[1].barh(r["feature"], r["rfecv_rank"], color=np.where(r["selected"], "#2ca02c", "#bbbbbb"))
    axes[1].set(xlabel="RFECV elimination rank (1 = selected; larger = eliminated earlier)", title="RFECV ranking")
    fig.suptitle(title)
    _save(fig, path, "RFECV rank reflects elimination order with a random-forest selector; it is not a generic feature-importance measure.")


def soc_trajectories(pred, segment_ids, pred_col, path):
    n = len(segment_ids)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.2), squeeze=False)
    for ax, sid in zip(axes[0], segment_ids):
        g = pred[pred["segment_id"] == sid]
        ax.plot(g["elapsed_s"], g["SOC_reference_percent"], "k-", lw=2, label="Coulomb-counting reference")
        ax.plot(g["elapsed_s"], g[pred_col], "-", color="#d62728", lw=1.2, label="Model prediction")
        ax.set(xlabel="Elapsed time in discharge (s)", ylabel="SOC (%)", title=f"{sid} (held-out)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("SOC reference and prediction vs time, representative held-out discharge segments")
    _save(fig, path, "Reference = 100 x (1 - integrated discharge Ah / measured capacity), unclipped; segments end at the 2.7 V cutoff.")


def soc_reference_quality(diag, samples, path):
    acc = diag[diag["accepted"]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for bid, g in acc.groupby("battery_id"):
        axes[0].scatter(g["capacity_Ah"], g["q_to_cutoff_Ah"], s=8, color=_color(bid), label=bid)
        axes[1].plot(g["discharge_number"], g["soc_end_unclipped_percent"], ".", ms=4, color=_color(bid), label=bid)
    lo, hi = acc["capacity_Ah"].min(), acc["capacity_Ah"].max()
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=1, label="Identity")
    axes[0].set(xlabel="Measured Capacity field (Ah)", ylabel="Integrated discharge charge to 2.7 V (Ah)",
                title="Integration vs measured capacity")
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set(xlabel="Discharge cycle number", ylabel="Unclipped reference SOC at segment end (%)",
                title="Reference SOC at 2.7 V cutoff")
    for ax in axes[:2]:
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ex = samples[samples["battery_id"] == samples["battery_id"].iloc[0]]
    ids = ex["segment_id"].unique()
    for sid in [ids[0], ids[len(ids) // 2], ids[-1]]:
        g = ex[ex["segment_id"] == sid]
        axes[2].plot(g["elapsed_s"], g["q_discharged_Ah"], label=f"{sid}: integrated Ah")
        axes[2].axhline(g["usable_capacity_Ah"].iloc[0], ls=":", lw=1, color=axes[2].lines[-1].get_color())
    axes[2].set(xlabel="Elapsed time (s)", ylabel="Cumulative discharged charge (Ah)",
                title="Integration (dotted = measured capacity)")
    axes[2].grid(alpha=0.3); axes[2].legend(fontsize=7)
    fig.suptitle("SOC reference quality diagnostics (derived values)")
    _save(fig, path, "Trapezoidal integration of -Current_measured over actual time steps. Deviation from identity shows reference error.")


def signal_plots(charge_ts, discharge_ts, bid, path):
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex="col")
    for col, (ts, op) in enumerate([(charge_ts, "charge"), (discharge_ts, "discharge")]):
        g = ts[ts["battery_id"] == bid]
        nums = sorted(g["op_number"].unique())
        if not nums:
            continue
        lengths = g.groupby("op_number").size()
        nums = [n for n in nums if lengths[n] >= 50]   # skip truncated records for display
        picks = [nums[0], nums[len(nums) // 2], nums[-1]]
        for n in picks:
            r = g[g["op_number"] == n]
            for row, (c, lab) in enumerate([("voltage_measured_V", "Voltage (V)"), ("current_measured_A", "Current (A)"),
                                            ("temperature_measured_C", "Temperature (°C)")]):
                axes[row, col].plot(r["time_s"], r[c], lw=1, label=f"{op} #{n}")
                axes[row, col].set_ylabel(f"Measured {lab}")
        axes[0, col].set_title(f"{bid} {op} records (measured)")
        axes[2, col].set_xlabel("Time within record (s)")
        for row in range(3):
            axes[row, col].grid(alpha=0.3)
        axes[0, col].legend(fontsize=8)
    _save(fig, path, "Data-understanding plot of measured signals (early / middle / late records); not model evidence. "
                     "Current sign: positive = charging, negative = discharging (Current_measured).")
