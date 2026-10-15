"""Cycle-level SOH features and sample-level SOC reference/features.

SOH and SOC are built by separate functions with separate feature lists; nothing here is shared
between the two targets except reading the same processed time-series tables.
"""
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

# --------------------------------------------------------------------------------------------
# SOH: one row per discharge cycle
# --------------------------------------------------------------------------------------------
SOH_FEATURES = [
    "discharge_number",
    # preceding charge record (measured before this discharge)
    "chg_cc_duration_s", "chg_cv_duration_s", "chg_v_start_V", "chg_temp_max_C", "chg_temp_rise_C",
    # impedance record measured between the previous discharge and this one
    "imp_Re_ohm", "imp_Rct_ohm",
    # first `early_window_s` seconds of this discharge only (no full-discharge duration/energy)
    "dis_v_start_V", "dis_v_60s_V", "dis_v_300s_V", "dis_v_600s_V", "dis_vdrop_60_600s_V",
    "dis_temp_start_C", "dis_temp_rise_600s_C",
]
SOH_TARGET = "SOH_percent"


def _charge_features(ts, min_samples, max_voltage):
    """Returns (features, flag). Truncated or physically implausible records give missing features."""
    nan = dict(chg_cc_duration_s=np.nan, chg_cv_duration_s=np.nan, chg_v_start_V=np.nan,
               chg_temp_max_C=np.nan, chg_temp_rise_C=np.nan)
    if ts is None:
        return nan, "no charge record"
    if len(ts) < min_samples:
        return nan, "truncated charge record"
    if np.nanmax(ts["voltage_measured_V"]) > max_voltage:
        return nan, f"charge voltage above {max_voltage} V (implausible for 4.2 V cell)"
    t = ts["time_s"].to_numpy(); i = ts["current_measured_A"].to_numpy()
    v = ts["voltage_measured_V"].to_numpy(); temp = ts["temperature_measured_C"].to_numpy()
    on = np.where(i > 1.0)[0]
    out = dict(nan)
    out["chg_v_start_V"] = v[0]
    out["chg_temp_max_C"] = np.nanmax(temp)
    out["chg_temp_rise_C"] = np.nanmax(temp) - temp[0]
    if len(on):
        after = np.where((np.arange(len(i)) > on[0]) & (i < 1.3))[0]   # CC (≈1.5 A) has ended
        if len(after):
            cc_end = after[0]
            out["chg_cc_duration_s"] = t[cc_end] - t[on[0]]
            low = np.where((np.arange(len(i)) > cc_end) & (i < 0.1))[0]  # CV taper reached 0.1 A
            if len(low):
                out["chg_cv_duration_s"] = t[low[0]] - t[cc_end]
    return out, ""


def _discharge_early_features(ts, window_s):
    t = ts["time_s"].to_numpy(); v = ts["voltage_measured_V"].to_numpy()
    temp = ts["temperature_measured_C"].to_numpy()
    if t[-1] < window_s:
        return {k: np.nan for k in SOH_FEATURES if k.startswith("dis_")}
    vi = lambda s: float(np.interp(s, t, v))
    ti = lambda s: float(np.interp(s, t, temp))
    return {
        "dis_v_start_V": v[0],
        "dis_v_60s_V": vi(60), "dis_v_300s_V": vi(300), "dis_v_600s_V": vi(window_s),
        "dis_vdrop_60_600s_V": vi(60) - vi(window_s),
        "dis_temp_start_C": temp[0],
        "dis_temp_rise_600s_C": ti(window_s) - temp[0],
    }


def build_soh_table(index, charge_ts, discharge_ts, cfg):
    rated = cfg["rated_capacity_Ah"]
    window = cfg["soh"]["early_window_s"]
    min_samples = cfg["soh"]["min_charge_samples"]
    ch_groups = {k: g for k, g in charge_ts.groupby(["battery_id", "record_index"])}
    dis_groups = {k: g for k, g in discharge_ts.groupby(["battery_id", "record_index"])}
    rows = []
    for bid, recs in index.groupby("battery_id"):
        recs = recs.sort_values("record_index")
        prev_dis_idx = -1
        for _, r in recs.iterrows():
            if r["op_type"] != "discharge":
                continue
            between = recs[(recs["record_index"] > prev_dis_idx) & (recs["record_index"] < r["record_index"])]
            ch = between[between["op_type"] == "charge"]
            imp = between[between["op_type"] == "impedance"]
            ch_rec = ch.iloc[-1]["record_index"] if len(ch) else None
            imp_row = imp.iloc[-1] if len(imp) else None
            row = {
                "battery_id": bid,
                "discharge_number": int(r["op_number"]),
                "record_index": int(r["record_index"]),
                "start_time": r["start_time"],
                "matched_charge_record": ch_rec,
                "matched_impedance_record": imp_row["record_index"] if imp_row is not None else None,
                "capacity_Ah": r["capacity_Ah"],
                SOH_TARGET: 100.0 * r["capacity_Ah"] / rated,
            }
            feats, flag = _charge_features(ch_groups.get((bid, ch_rec)), min_samples, cfg["soh"]["max_plausible_voltage_V"])
            row.update(feats)
            row["charge_feature_flag"] = flag
            row["imp_Re_ohm"] = imp_row["Re_ohm"] if imp_row is not None else np.nan
            row["imp_Rct_ohm"] = imp_row["Rct_ohm"] if imp_row is not None else np.nan
            row.update(_discharge_early_features(dis_groups[(bid, r["record_index"])], window))
            rows.append(row)
            prev_dis_idx = r["record_index"]
    return pd.DataFrame(rows)


SOH_DICTIONARY = [
    ("battery_id", "-", "Battery identifier; grouping key only, not a feature"),
    ("discharge_number", "count", "1-based index of the discharge operation within the battery file"),
    ("record_index", "-", "0-based position of the record in the MATLAB cycle array"),
    ("start_time", "datetime", "Record start time from the MATLAB date vector"),
    ("matched_charge_record", "-", "record_index of the last charge between the previous discharge and this one"),
    ("matched_impedance_record", "-", "record_index of the last impedance test between the previous discharge and this one"),
    ("capacity_Ah", "Ah", "Measured discharge capacity (file field Capacity). TARGET SOURCE - never a predictor"),
    ("SOH_percent", "%", "TARGET: 100 * capacity_Ah / rated capacity (2.0 Ah from supplied README)"),
    ("charge_feature_flag", "-", "Why chg_* features are missing (no/truncated/implausible charge record); empty if valid"),
    ("chg_cc_duration_s", "s", "Preceding charge: time from current >1.0 A until current first falls below 1.3 A (end of 1.5 A CC phase)"),
    ("chg_cv_duration_s", "s", "Preceding charge: time from end of CC until current falls below 0.1 A"),
    ("chg_v_start_V", "V", "Preceding charge: measured terminal voltage at first sample"),
    ("chg_temp_max_C", "degC", "Preceding charge: maximum measured cell temperature"),
    ("chg_temp_rise_C", "degC", "Preceding charge: max minus initial measured temperature"),
    ("imp_Re_ohm", "ohm", "Re (electrolyte resistance) from impedance test between previous discharge and this one"),
    ("imp_Rct_ohm", "ohm", "Rct (charge-transfer resistance) from the same impedance test"),
    ("dis_v_start_V", "V", "This discharge: terminal voltage at t=0 (before load current rises)"),
    ("dis_v_60s_V", "V", "This discharge: terminal voltage at 60 s (linear interpolation)"),
    ("dis_v_300s_V", "V", "This discharge: terminal voltage at 300 s"),
    ("dis_v_600s_V", "V", "This discharge: terminal voltage at 600 s"),
    ("dis_vdrop_60_600s_V", "V", "This discharge: voltage at 60 s minus voltage at 600 s"),
    ("dis_temp_start_C", "degC", "This discharge: measured temperature at t=0"),
    ("dis_temp_rise_600s_C", "degC", "This discharge: temperature at 600 s minus temperature at t=0"),
]

# --------------------------------------------------------------------------------------------
# SOC: one row per time sample inside a discharge segment
# --------------------------------------------------------------------------------------------
SOC_FEATURES = [
    "voltage_measured_V", "discharge_current_A", "temperature_measured_C", "voltage_load_V",
    "elapsed_s", "temp_rise_C", "voltage_drop_V", "dvdt_past_V_per_s",
]
SOC_TARGET = "SOC_reference_percent"


def build_soc_table(index, discharge_ts, cfg):
    """Coulomb-counting SOC reference per discharge segment.

    Rules (from inspection):
      * Current_measured is negative while discharging -> discharge_current_A = -Current_measured.
      * Segment = samples from the start of the discharge record up to (excluding) the first
        sample where voltage < cutoff while the load is on. Rest samples after load-off and the
        deeper battery-specific cutoffs (2.5/2.2 V) are excluded so every segment matches the
        README definition of Capacity ("discharge till 2.7V").
      * SOC0 = 100 % only if the record starts at >= full_charge_min_voltage_V.
      * usable capacity = measured Capacity of that discharge (cycle-specific reference).
    Returns (samples DataFrame, per-segment diagnostics DataFrame).
    """
    c = cfg["soc"]
    caps = index.set_index(["battery_id", "record_index"])["capacity_Ah"]
    samples, diag = [], []
    for (bid, rec), g in discharge_ts.groupby(["battery_id", "record_index"], sort=True):
        g = g.sort_values("sample")
        t = g["time_s"].to_numpy(); v = g["voltage_measured_V"].to_numpy()
        i_dis = -g["current_measured_A"].to_numpy()
        temp = g["temperature_measured_C"].to_numpy()
        cap = caps.loc[(bid, rec)]
        d = {"battery_id": bid, "record_index": rec, "discharge_number": int(g["op_number"].iloc[0]),
             "capacity_Ah": cap, "v_start_V": v[0], "n_samples_record": len(g)}
        reasons = []
        if np.any(np.diff(t) <= 0):
            reasons.append("time not increasing")
        if np.max(np.diff(t)) > c["max_gap_s"]:
            reasons.append(f"time gap > {c['max_gap_s']} s")
        if v[0] < c["full_charge_min_voltage_V"]:
            reasons.append("start voltage below full-charge threshold")
        if not np.isfinite(cap) or cap <= 0:
            reasons.append("no valid measured capacity")
        cross = np.where((v < c["cutoff_voltage_V"]) & (i_dis > c["load_on_current_A"]))[0]
        if len(cross) == 0:
            reasons.append("never reaches cutoff under load")
        end = cross[0] if len(cross) else len(t)
        q_full = cumulative_trapezoid(i_dis, t, initial=0) / 3600.0
        d["q_full_record_Ah"] = q_full[-1]
        d["q_to_cutoff_Ah"] = q_full[end - 1] if end > 0 else np.nan
        d["q_to_cutoff_over_capacity"] = d["q_to_cutoff_Ah"] / cap if cap else np.nan
        d["segment_samples"] = int(end)
        d["segment_duration_s"] = t[end - 1] - t[0] if end > 0 else np.nan
        d["max_dt_s"] = float(np.max(np.diff(t)))
        d["mean_load_current_A"] = float(np.mean(i_dis[:end][i_dis[:end] > c["load_on_current_A"]])) if end else np.nan
        if reasons:
            d["accepted"] = False; d["reject_reason"] = "; ".join(reasons)
            diag.append(d)
            continue
        s = slice(0, end)
        q = q_full[s]
        soc = 100.0 * (1.0 - q / cap)
        d["soc_end_unclipped_percent"] = soc[-1]
        d["soc_min_unclipped_percent"] = soc.min()
        d["soc_max_unclipped_percent"] = soc.max()
        d["soc_max_increase_percent"] = float(np.max(np.diff(soc), initial=0.0))
        d["accepted"] = True; d["reject_reason"] = ""
        diag.append(d)

        k = c["slope_window_samples"]
        tt, vv = t[s], v[s]
        back = np.maximum(np.arange(end) - k, 0)
        dt = tt - tt[back]
        dvdt = np.where(dt > 0, (vv - vv[back]) / np.where(dt > 0, dt, 1), np.nan)
        samples.append(pd.DataFrame({
            "battery_id": bid, "record_index": rec, "discharge_number": d["discharge_number"],
            "segment_id": f"{bid}_d{d['discharge_number']:03d}",
            "sample": np.arange(end),
            "voltage_measured_V": vv,
            "discharge_current_A": i_dis[s],
            "temperature_measured_C": temp[s],
            "voltage_load_V": g["voltage_load_V"].to_numpy()[s],
            "elapsed_s": tt - tt[0],
            "temp_rise_C": temp[s] - temp[0],
            "voltage_drop_V": v[0] - vv,
            "dvdt_past_V_per_s": dvdt,
            "q_discharged_Ah": q,
            "usable_capacity_Ah": cap,
            SOC_TARGET: soc,
        }))
    return (pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()), pd.DataFrame(diag)


SOC_DICTIONARY = [
    ("battery_id", "-", "Battery identifier; grouping key only"),
    ("record_index", "-", "0-based position of the discharge record in the MATLAB cycle array"),
    ("discharge_number", "count", "1-based discharge index within the battery"),
    ("segment_id", "-", "Discharge segment key (battery + discharge number); grouping key"),
    ("sample", "count", "Sample index within the segment"),
    ("voltage_measured_V", "V", "Measured terminal voltage at this sample"),
    ("discharge_current_A", "A", "-Current_measured (positive = discharging) at this sample"),
    ("temperature_measured_C", "degC", "Measured cell temperature at this sample"),
    ("voltage_load_V", "V", "Voltage measured at load at this sample"),
    ("elapsed_s", "s", "Time since segment start"),
    ("temp_rise_C", "degC", "Temperature now minus temperature at segment start (past-only)"),
    ("voltage_drop_V", "V", "Voltage at segment start minus voltage now (past-only)"),
    ("dvdt_past_V_per_s", "V/s", "Backward voltage slope over the previous 3 samples (past-only)"),
    ("q_discharged_Ah", "Ah", "Trapezoidal integral of discharge current up to this sample. Used for the reference only, NOT a predictor"),
    ("usable_capacity_Ah", "Ah", "Measured Capacity of this discharge. Reference denominator only, NOT a predictor"),
    ("SOC_reference_percent", "%", "TARGET (approximate reference): 100 * (1 - q_discharged_Ah / usable_capacity_Ah); unclipped"),
]
