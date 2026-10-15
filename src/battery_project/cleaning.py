"""Validate parsed records and write normalized processed tables."""
import numpy as np
import pandas as pd

# Signal names in the files -> stable processed column names with units.
SIGNAL_COLUMNS = {
    "Time": "time_s",
    "Voltage_measured": "voltage_measured_V",
    "Current_measured": "current_measured_A",
    "Temperature_measured": "temperature_measured_C",
    "Current_charge": "current_charger_A",
    "Voltage_charge": "voltage_charger_V",
    "Current_load": "current_load_A",
    "Voltage_load": "voltage_load_V",
}


def _to_float(arr):
    out = pd.to_numeric(pd.Series(np.asarray(arr).ravel()), errors="coerce").to_numpy(dtype=float)
    return out


def validate_records(records):
    """Coerce signals to float and attach quality flags. Returns (records, warnings)."""
    warnings = []
    for r in records:
        flags = []
        sig = {k: _to_float(v) for k, v in r["signals"].items()}
        lengths = {len(v) for v in sig.values()}
        if len(lengths) > 1:
            flags.append("signal_length_mismatch")
        for name, v in sig.items():
            if not np.all(np.isfinite(v)):
                flags.append(f"nonfinite_{name}")
        t = sig.get("Time")
        if t is not None and len(t) > 1 and np.any(np.diff(t) <= 0):
            flags.append("time_not_increasing")
        r["signals"] = sig
        r["n_samples"] = max(lengths) if lengths else 0
        r["flags"] = flags
        if flags:
            warnings.append(f"{r['battery_id']} record {r['record_index']} ({r['op_type']}): {', '.join(flags)}")
    return records, warnings


def remove_exact_duplicates(records):
    """Drop records whose (battery, op_type, start_time) key and all values are identical."""
    seen, kept, removed = {}, [], 0
    for r in records:
        key = (r["battery_id"], r["op_type"], r["start_time"])
        if key in seen:
            o = seen[key]
            same = (o["signals"].keys() == r["signals"].keys()
                    and all(np.array_equal(o["signals"][k], r["signals"][k], equal_nan=True) for k in r["signals"])
                    and o["scalars"] == r["scalars"])
            if same:
                removed += 1
                continue
        seen[key] = r
        kept.append(r)
    return kept, removed


def number_operations(records):
    """Add op_number: 1-based count of each operation type within a battery, in file order."""
    counters = {}
    for r in records:
        key = (r["battery_id"], r["op_type"])
        counters[key] = counters.get(key, 0) + 1
        r["op_number"] = counters[key]
    return records


def records_index_table(records):
    rows = []
    for r in records:
        t = r["signals"].get("Time")
        rows.append({
            "battery_id": r["battery_id"],
            "record_index": r["record_index"],
            "op_type": r["op_type"],
            "op_number": r["op_number"],
            "start_time": r["start_time"],
            "ambient_temperature_C": r["ambient_temperature_C"],
            "n_samples": r["n_samples"],
            "duration_s": float(t[-1] - t[0]) if t is not None and len(t) > 1 else np.nan,
            "capacity_Ah": r["scalars"].get("Capacity", np.nan),
            "Re_ohm": r["scalars"].get("Re", np.nan),
            "Rct_ohm": r["scalars"].get("Rct", np.nan),
            "flags": ";".join(r["flags"]),
        })
    return pd.DataFrame(rows)


def timeseries_table(records, op_type):
    frames = []
    for r in records:
        if r["op_type"] != op_type or not r["signals"]:
            continue
        df = pd.DataFrame({SIGNAL_COLUMNS.get(k, k): v for k, v in r["signals"].items()})
        df.insert(0, "sample", np.arange(len(df)))
        df.insert(0, "op_number", r["op_number"])
        df.insert(0, "record_index", r["record_index"])
        df.insert(0, "battery_id", r["battery_id"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
