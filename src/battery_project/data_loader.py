"""Locate the NASA .mat files, stage them unchanged in data/raw, and parse cycle records.

The parser reads whatever fields each record's `data` struct actually contains; it does not
assume a fixed field list. Observed layout (MATLAB v5 files):
    <BatteryID>[0,0].cycle[0, k] -> struct(type, ambient_temperature, time, data)
"""
import hashlib
import shutil
from datetime import datetime, timedelta

import numpy as np
import scipy.io as sio


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_raw_files(cfg):
    """Copy supplied .mat/README files into raw_dir byte-for-byte (originals untouched).

    Returns (found: {battery_id: path}, missing: [battery_id], notes: [str]).
    """
    raw_dir = cfg["paths"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    wanted = [f"{b}.mat" for b in cfg["expected_batteries"]] + cfg.get("documentation_files", [])
    notes = []
    for name in wanted:
        dest = raw_dir / name
        if dest.exists():
            continue
        for src_dir in cfg["paths"]["source_dirs"]:
            src = src_dir / name
            if src.exists():
                shutil.copy2(src, dest)
                if sha256(src) != sha256(dest):
                    raise RuntimeError(f"Copy of {src} is not byte-identical")
                notes.append(f"Copied {src} -> {dest} (sha256 verified)")
                break
    found, missing = {}, []
    for b in cfg["expected_batteries"]:
        p = raw_dir / f"{b}.mat"
        if p.exists():
            found[b] = p
        else:
            missing.append(b)
    return found, missing, notes


def _datevec_to_datetime(vec):
    v = np.asarray(vec, dtype=float).ravel()
    if v.size != 6 or not np.all(np.isfinite(v)):
        return None
    y, mo, d, h, mi, s = v
    return datetime(int(y), int(mo), int(d), int(h), int(mi)) + timedelta(seconds=float(s))


def load_battery(path, battery_id):
    """Return a list of normalized records for one battery file.

    Each record: battery_id, record_index (position in the file), op_type,
    ambient_temperature_C, start_time, signals {name: 1-D array} for vector fields,
    scalars {name: float} for 1x1 fields, complex_fields {name: array}.
    """
    try:
        mat = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
    except NotImplementedError as exc:  # MATLAB v7.3 (HDF5); not present in the supplied files
        raise RuntimeError(f"{path} looks like MATLAB v7.3/HDF5, which this parser does not handle: {exc}")
    keys = [k for k in mat if not k.startswith("__")]
    if battery_id in keys:
        key = battery_id
    elif len(keys) == 1:
        key = keys[0]
    else:
        raise RuntimeError(f"{path}: cannot identify battery struct among keys {keys}")
    top = mat[key][0, 0]
    if "cycle" not in top._fieldnames:
        raise RuntimeError(f"{path}: struct has no 'cycle' field (fields: {top._fieldnames})")

    records = []
    for k, c in enumerate(top.cycle.ravel()):
        data = c.data[0, 0]
        rec = {
            "battery_id": battery_id,
            "record_index": k,
            "op_type": str(np.asarray(c.type).ravel()[0]),
            "ambient_temperature_C": float(np.asarray(c.ambient_temperature).ravel()[0]),
            "start_time": _datevec_to_datetime(c.time),
            "signals": {},
            "scalars": {},
            "complex_fields": {},
        }
        for name in data._fieldnames:
            arr = np.asarray(getattr(data, name))
            if np.iscomplexobj(arr):
                rec["complex_fields"][name] = arr.ravel()
            elif arr.size == 1:
                rec["scalars"][name] = float(arr.ravel()[0])
            elif arr.size == 0:
                rec["scalars"][name] = np.nan
            else:
                rec["signals"][name] = arr.ravel()
        records.append(rec)
    return records
