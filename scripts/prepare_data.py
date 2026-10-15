"""Step 2: parse and clean all records, write processed tables, SOH feature table and SOC sample table."""
import _bootstrap  # noqa: F401
import pandas as pd

from battery_project import cleaning, data_loader, feature_engineering as fe
from battery_project.config import ensure_dirs, load_config


def main():
    cfg = load_config(); ensure_dirs(cfg)
    found, missing, _ = data_loader.stage_raw_files(cfg)
    if missing:
        print(f"MISSING battery files, continuing without them: {missing}")
    if not found:
        raise SystemExit("No battery .mat files found in data/raw. Run scripts/inspect_data.py after placing them.")
    out = cfg["paths"]["processed_dir"]

    records, removed, warnings = [], 0, []
    for bid, path in found.items():
        recs = data_loader.load_battery(path, bid)
        recs, w = cleaning.validate_records(recs)
        recs, r = cleaning.remove_exact_duplicates(recs)
        records += cleaning.number_operations(recs); removed += r; warnings += w
    print(f"Parsed {len(records)} records; exact duplicates removed: {removed}; records with quality flags: {len(warnings)}")

    index = cleaning.records_index_table(records)
    index.to_csv(out / "records_index.csv", index=False)
    charge_ts = cleaning.timeseries_table(records, "charge")
    discharge_ts = cleaning.timeseries_table(records, "discharge")
    charge_ts.to_parquet(out / "charge_timeseries.parquet", index=False)
    discharge_ts.to_parquet(out / "discharge_timeseries.parquet", index=False)
    print(f"Time series rows: charge {len(charge_ts)}, discharge {len(discharge_ts)}")

    # SOH table (one row per battery discharge cycle)
    soh = fe.build_soh_table(index, charge_ts, discharge_ts, cfg)
    soh.to_csv(out / "soh_features.csv", index=False)
    pd.DataFrame(fe.SOH_DICTIONARY, columns=["column", "unit", "description"]).to_csv(out / "soh_data_dictionary.csv", index=False)
    cov = soh.groupby("battery_id").agg(cycles=("discharge_number", "size"),
                                         charge_matched=("chg_cc_duration_s", lambda s: s.notna().mean()),
                                         impedance_matched=("imp_Re_ohm", lambda s: s.notna().mean()))
    print("SOH rows per battery and match coverage:\n" + cov.to_string())
    cov.to_csv(cfg["paths"]["results_dir"] / "tables" / "soh_match_coverage.csv")

    # SOC table (one row per time sample inside a discharge segment) - built independently
    soc, diag = fe.build_soc_table(index, discharge_ts, cfg)
    soc.to_parquet(out / "soc_samples.parquet", index=False)
    diag.to_csv(out / "soc_segment_diagnostics.csv", index=False)
    pd.DataFrame(fe.SOC_DICTIONARY, columns=["column", "unit", "description"]).to_csv(out / "soc_data_dictionary.csv", index=False)
    print(f"SOC: {int(diag['accepted'].sum())}/{len(diag)} discharge segments accepted, {len(soc)} samples")
    with open(out / "prepare_log.txt", "w", encoding="utf-8") as f:
        f.write(f"missing_batteries: {missing}\nduplicates_removed: {removed}\n")
        f.write("quality_warnings:\n" + "\n".join(warnings) + "\n")


if __name__ == "__main__":
    main()
