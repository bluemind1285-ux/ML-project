"""Streamlit BMS dashboard: presentation/inference layer over the saved SOH and SOC pipelines.

Run: streamlit run dashboard.py
It never trains models. SOH and SOC use their own artifacts and are only shown side by side here.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from battery_project.artifacts import load_model  # noqa: E402
from battery_project.config import load_config  # noqa: E402

st.set_page_config(page_title="Battery BMS dashboard", layout="wide")
cfg = load_config()
P = cfg["paths"]


@st.cache_resource
def get_model(target):
    d = P["models_dir"] / target
    if not (d / "model.joblib").exists() or not (d / "metadata.json").exists():
        return None, None
    return load_model(d)


@st.cache_data
def get_soh_table():
    path = P["processed_dir"] / "soh_features.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def get_soc_samples():
    path = P["processed_dir"] / "soc_samples.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data
def get_discharge_ts():
    path = P["processed_dir"] / "discharge_timeseries.parquet"
    return pd.read_parquet(path) if path.exists() else None


def split_label(meta, battery):
    if battery in meta["split"]["test_batteries"]:
        return "HELD-OUT test battery (never used for selection or training)"
    if battery in meta["split"]["train_batteries"]:
        return "TRAINING battery (in-sample, optimistic)"
    return "not used by this model"


def manual_inputs(meta, key):
    """Number inputs for the model's selected features, with training ranges. Returns (row or None, warnings)."""
    vals, out_of_range = {}, []
    cols = st.columns(2)
    for i, f in enumerate(meta["feature_order"]):
        r = meta["training_feature_ranges"][f]
        mid = (r["min"] + r["max"]) / 2
        txt = cols[i % 2].text_input(f"{f}  (training range {r['min']:.4g} … {r['max']:.4g})", value=f"{mid:.6g}", key=f"{key}_{f}")
        try:
            v = float(txt)
        except ValueError:
            v = np.nan
        vals[f] = v
        if np.isfinite(v) and not (r["min"] <= v <= r["max"]):
            out_of_range.append(f)
    bad = [f for f, v in vals.items() if not np.isfinite(v)]
    if bad:
        st.error(f"Missing or non-numeric input for: {', '.join(bad)}. No prediction made.")
        return None, out_of_range
    return pd.DataFrame([vals])[meta["feature_order"]], out_of_range


st.title("Battery management dashboard (NASA Li-ion aging data)")
st.caption("Predictions come from saved models. Measured and reference values from the dataset are shown separately and labelled. "
           "Results apply to the dataset's test protocol (24 °C, 1.5 A CC-CV charge, 2 A CC discharge).")

soh_model, soh_meta = get_model("soh")
soc_model, soc_meta = get_model("soc")
soh_df = get_soh_table()
if soh_df is None:
    st.error("Processed data not found. Run `python scripts/prepare_data.py` first.")
    st.stop()

mode = st.sidebar.radio("Mode", ["Dataset demonstration", "Manual input"])

# ------------------------------------------------------------------------------------ demo mode
if mode == "Dataset demonstration":
    batteries = sorted(soh_df["battery_id"].unique())
    default_b = soh_meta["split"]["test_batteries"][0] if soh_meta else batteries[0]
    battery = st.sidebar.selectbox("Battery", batteries, index=batteries.index(default_b))
    bdf = soh_df[soh_df["battery_id"] == battery].sort_values("discharge_number")
    cycle = st.sidebar.select_slider("Discharge cycle", options=bdf["discharge_number"].tolist(),
                                     value=bdf["discharge_number"].iloc[len(bdf) // 2])
    row = bdf[bdf["discharge_number"] == cycle].iloc[0]

    col_soh, col_soc = st.columns(2)

    # ---------------- SOH
    with col_soh:
        st.header("State of health (SOH)")
        if soh_model is None:
            st.error("SOH model artifacts missing in models/soh/. Run `python scripts/train_soh.py`.")
        else:
            st.markdown(f"**Data split:** {split_label(soh_meta, battery)}")
            pred = float(soh_model.predict(row[soh_meta["feature_order"]].to_frame().T.astype(float))[0])
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted SOH (model)", f"{pred:.1f} %")
            m2.metric("Reference SOH (measured capacity)", f"{row['SOH_percent']:.1f} %")
            m3.metric("Measured capacity", f"{row['capacity_Ah']:.3f} Ah")
            tm = soh_meta["test_metrics"][soh_meta["model_name"]]
            st.caption(f"Model: {soh_meta['model_name']} with {len(soh_meta['feature_order'])} RFECV-selected features. "
                       f"Held-out ({', '.join(soh_meta['split']['test_batteries'])}) MAE {tm['MAE']:.2f}, RMSE {tm['RMSE']:.2f} SOH points, R² {tm['R2']:.3f}; "
                       "it over-predicted the held-out battery on average (see results/summary.md). Treat ± a few points as the uncertainty scale. "
                       f"Reference SOH = 100 × capacity / {cfg['rated_capacity_Ah']} Ah rated (README).")
            missing = [f for f in soh_meta["feature_order"] if not np.isfinite(row[f])]
            if missing:
                st.warning(f"This cycle has no measured value for {', '.join(missing)}; the model imputed the training median.")

            preds_all = soh_model.predict(bdf[soh_meta["feature_order"]].astype(float))
            fig, ax = plt.subplots(figsize=(6, 3.4))
            ax.plot(bdf["discharge_number"], bdf["SOH_percent"], "k.-", ms=3, lw=0.8, label="Reference SOH (measured capacity)")
            ax.plot(bdf["discharge_number"], preds_all, "-", color="#d62728", lw=1.2, label="Predicted SOH (model)")
            ax.axvline(cycle, color="grey", ls=":", lw=1)
            ax.axhline(100 * cfg["eol_capacity_Ah"] / cfg["rated_capacity_Ah"], color="grey", ls="--", lw=0.8, label="EOL 70 % (README)")
            ax.set(xlabel="Discharge cycle", ylabel="SOH (%)", title=f"{battery}: SOH trend")
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
            st.pyplot(fig); plt.close(fig)

    # ---------------- SOC
    with col_soc:
        st.header("State of charge (SOC)")
        soc_ok = (soc_model is not None and soc_meta.get("reference_defensible") and cfg["soc"].get("dashboard_enabled"))
        if soc_model is None:
            st.info("No SOC model artifacts in models/soc/. Run `python scripts/train_soc.py`. "
                    "If the reference was judged not defensible, see results/soc_assessment.md.")
        elif not soc_ok:
            st.info("SOC is not displayed as a validated estimate: the reference or review did not approve it. See results/soc_assessment.md.")
        else:
            samples = get_soc_samples()
            seg = samples[(samples["battery_id"] == battery) & (samples["discharge_number"] == cycle)].sort_values("sample")
            if seg.empty:
                st.warning("No SOC segment for this cycle.")
            else:
                st.markdown(f"**Data split:** {split_label(soc_meta, battery)}")
                seg_pred = soc_model.predict(seg[soc_meta["feature_order"]])
                t_idx = st.slider("Time within discharge (sample)", 0, len(seg) - 1, len(seg) // 2)
                s = seg.iloc[t_idx]
                m1, m2, m3 = st.columns(3)
                m1.metric("Predicted SOC (model)", f"{seg_pred[t_idx]:.1f} %")
                m2.metric("Reference SOC (coulomb counting)", f"{s['SOC_reference_percent']:.1f} %")
                m3.metric("Elapsed / voltage", f"{s['elapsed_s']:.0f} s / {s['voltage_measured_V']:.3f} V")
                tm = soc_meta["test_metrics"][soc_meta["model_name"]]
                st.caption(f"Model: {soc_meta['model_name']}. Held-out ({', '.join(soc_meta['split']['test_batteries'])}) MAE {tm['MAE']:.2f}, "
                           f"RMSE {tm['RMSE']:.2f} SOC points; errors up to ~8 points occurred mid-discharge. The reference is an approximate "
                           "coulomb-counting SOC (0 % = 2.7 V cutoff at 2 A), not measured ground truth. Valid only for 2 A CC discharge from full charge at 24 °C.")
                fig, ax = plt.subplots(figsize=(6, 3.4))
                ax.plot(seg["elapsed_s"], seg["SOC_reference_percent"], "k-", lw=2, label="Reference SOC (coulomb counting)")
                ax.plot(seg["elapsed_s"], seg_pred, "-", color="#d62728", lw=1.2, label="Predicted SOC (model)")
                ax.axvline(s["elapsed_s"], color="grey", ls=":", lw=1)
                ax.set(xlabel="Elapsed time (s)", ylabel="SOC (%)", title=f"{battery} discharge #{cycle}")
                ax.legend(fontsize=7); ax.grid(alpha=0.3)
                st.pyplot(fig); plt.close(fig)

    # ---------------- measured signals
    st.header("Measured signals for this discharge")
    ts = get_discharge_ts()
    rec = ts[(ts["battery_id"] == battery) & (ts["record_index"] == row["record_index"])]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.2))
    for ax, (c, lab) in zip(axes, [("voltage_measured_V", "Voltage (V)"), ("current_measured_A", "Current (A)"),
                                   ("temperature_measured_C", "Temperature (°C)")]):
        ax.plot(rec["time_s"], rec[c], lw=1)
        ax.set(xlabel="Time (s)", ylabel=f"Measured {lab}"); ax.grid(alpha=0.3)
    fig.suptitle(f"{battery} discharge #{cycle}: measured data (negative current = discharging)")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with st.expander("Capacity fade across all batteries (measured)"):
        fig, ax = plt.subplots(figsize=(8, 3.5))
        for b, g in soh_df.groupby("battery_id"):
            ax.plot(g["discharge_number"], g["capacity_Ah"], ".-", ms=2, lw=0.7, label=b)
        ax.axhline(cfg["eol_capacity_Ah"], color="grey", ls="--", lw=0.8, label="EOL 1.4 Ah (README)")
        ax.set(xlabel="Discharge cycle", ylabel="Measured capacity (Ah)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close(fig)

# ------------------------------------------------------------------------------------ manual mode
else:
    st.warning("Manual mode: predictions are only meaningful for measurements taken under the dataset's protocol and within the "
               "training ranges shown. Inputs outside those ranges are flagged; the model has no evidence there.")
    st.header("SOH from manual inputs")
    if soh_model is None:
        st.error("SOH model artifacts missing. Run `python scripts/train_soh.py`.")
    else:
        X, oor = manual_inputs(soh_meta, "soh")
        if X is not None:
            if oor:
                st.warning(f"Outside training range: {', '.join(oor)}. Prediction is an extrapolation and not trustworthy.")
            st.metric("Predicted SOH (model)", f"{float(soh_model.predict(X)[0]):.1f} %")
            st.caption("Feature definitions: data/processed/soh_data_dictionary.csv")

    st.header("SOC from manual inputs")
    if soc_model is None or not (soc_meta.get("reference_defensible") and cfg["soc"].get("dashboard_enabled")):
        st.info("SOC is not available as a validated estimate (see results/soc_assessment.md).")
    else:
        X, oor = manual_inputs(soc_meta, "soc")
        if X is not None:
            if oor:
                st.warning(f"Outside training range: {', '.join(oor)}. Prediction is an extrapolation and not trustworthy.")
            st.metric("Predicted SOC (model)", f"{float(soc_model.predict(X)[0]):.1f} %")
            st.caption("Inputs must come from one sample of a 2 A CC discharge that started from full charge. "
                       "Feature definitions: data/processed/soc_data_dictionary.csv")
