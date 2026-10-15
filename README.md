# NASA Battery SOH/SOC Estimation and BMS Dashboard

This project estimates the state of health (SOH) of the NASA Li-ion aging batteries B0005, B0006, B0007 and B0018. It also builds an approximate state-of-charge (SOC) reference and model, and shows both in a Streamlit dashboard. The implementation contract is [PROJECT_SPECIFICATION.md](PROJECT_SPECIFICATION.md).

SOH and SOC are independent pipelines from feature construction through to saved models. They share only the parsed time-series tables, and their results first appear together in the dashboard.

## Setup

Python 3.14 was used. Run these from the project root:

```text
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

Inputs: `inspect_data.py` copies `B0005.mat`, `B0006.mat`, `B0007.mat`, `B0018.mat` and `README.txt` byte-for-byte from `1. BatteryAgingARC-FY08Q4/` into `data/raw/` and verifies them by sha256. The originals are never modified. You can also place the files in `data/raw/` yourself. Missing batteries are reported, and the scripts continue with the files they find.

## Run

```text
python scripts/inspect_data.py     # inventory -> results/data_inventory.json, results/data_inspection.md
python scripts/prepare_data.py     # parse/clean -> data/processed/ (records, time series, SOH table, SOC samples)
python scripts/train_soh.py        # SOH: RFECV + 3 models + held-out test -> models/soh/, results/
python scripts/train_soc.py        # SOC: reference checks, then (if defensible) RFECV + 3 models -> models/soc/, results/soc_assessment.md
python scripts/make_report.py      # context figures + results/summary.md
streamlit run dashboard.py
```

The full sequence took about 2.5 minutes on the development laptop; SOC training is the slowest step. The dashboard only loads saved artifacts and never retrains. If an artifact is missing, it tells you which command to run.

**Security note:** `models/*/model.joblib` files are loaded with joblib (pickle). Only load artifacts produced by this project.

## Layout

| Path | Contents |
|---|---|
| `config.yaml` | Paths, rated capacity, test battery, and SOC reference settings, each with its justification |
| `src/battery_project/` | `data_loader` (MATLAB parsing), `cleaning`, `feature_engineering` (separate SOH and SOC builders), `feature_selection` (separate SOH/SOC RFECV), `models`, `evaluation`, `visualization`, `soh_pipeline`, `soc_pipeline`, `artifacts` |
| `scripts/` | Thin command-line entry points |
| `data/processed/` | `records_index.csv`, `charge_timeseries.parquet`, `discharge_timeseries.parquet`, `soh_features.csv` + `soh_data_dictionary.csv`, `soc_samples.parquet` + `soc_data_dictionary.csv`, `soc_segment_diagnostics.csv` |
| `models/soh`, `models/soc` | `model.joblib` (imputer + scaler + regressor on selected features) and `metadata.json` (features, ranges, split, metrics, versions, assumptions) |
| `results/` | `data_inspection.md`, `soc_assessment.md`, `summary.md`, `tables/*.csv`, `figures/*.png` |

## Method

**Data.** Each file holds a `cycle` struct array of charge, discharge and impedance records. Inspection findings are in [results/data_inspection.md](results/data_inspection.md). The main ones:

- Discharge records use `Current_load`/`Voltage_load`, not the field names the README lists.
- `Current_load` changes sign between records.
- `Current_measured` is negative while discharging.
- Some charge records are truncated or physically implausible (>8 V). Their features are set to missing, not deleted.
- Impedance tests do not precede every discharge.

**SOH target.** `SOH_percent = 100 × Capacity / 2.0 Ah`. The 2 Ah rated capacity comes from the supplied README ("from 2Ahr to 1.4Ahr"). Values are not clipped: B0006 starts at 101.8 %.

**SOH predictors.** All are measured before or early in the discharge, so none is an alias of capacity:

- From the preceding charge: CC and CV durations, start voltage, and temperature.
- `Re`/`Rct` from an impedance test between the previous discharge and this one. These are missing when no test exists; nothing is forward-filled.
- Voltage and temperature features from only the first 600 s of the discharge.
- The discharge cycle number.

Full-discharge duration, energy and integrated Ah are excluded because under a 2 A constant current they are proxies of the target.

**SOC reference.** Coulomb counting per discharge record:

```text
Q(t)   = trapz(-Current_measured, Time) / 3600
SOC(t) = 100 × (1 − Q(t) / Capacity)
```

- SOC starts at 100 %. This is justified because each discharge follows a CC-CV charge to 4.2 V, and every discharge starts at ≥ 4.158 V.
- The segment ends at the first sample below 2.7 V under load, because the README defines `Capacity` to 2.7 V.
- Integrated charge matches `Capacity` to within 0.62 % for all 636 segments.

The reference is approximate and cycle-specific: its denominator is known only after the discharge ends. It is not measured ground-truth SOC. Details are in [results/soc_assessment.md](results/soc_assessment.md).

**SOC predictors.** These are causal per-sample signals, using only past samples:

- voltage, discharge current, temperature, and load voltage;
- elapsed time;
- temperature rise and voltage drop since the segment start;
- backward dV/dt over the last 3 samples.

The integrated charge and the capacity are not predictors.

**Split and selection** (identical procedure, run separately for each target):

1. Battery **B0018 is held out** for the final test. This was fixed in `config.yaml` before training.
2. RFECV (random-forest selector, leave-one-battery-out CV on B0005–B0007, MAE scoring) runs on training batteries only.
3. Linear Regression, Decision Tree and Random Forest, with fixed modest settings, are compared by grouped CV.
4. The model with the lowest CV MAE is chosen.
5. All three are then evaluated once on B0018.

Imputation and scaling are fitted inside each pipeline on training data. For SOC, whole discharge segments stay on one side of the split.

## Results (held-out battery B0018)

### SOH

132 discharge cycles. RFECV kept 8 features: discharge_number, chg_cc_duration_s, chg_v_start_V, dis_v_60s_V, dis_v_300s_V, dis_v_600s_V, dis_vdrop_60_600s_V, dis_temp_rise_600s_C.

| Model | CV MAE | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|
| Linear Regression | 4.00 | 4.74 | 4.93 | 0.592 |
| Decision Tree | 2.85 | 2.33 | 2.87 | 0.862 |
| **Random Forest (selected by CV)** | 2.62 | **2.30** | **2.66** | **0.881** |

Units are SOH percentage points. The selected model **over-predicts B0018 by 2.2 points on average**, across the whole SOH range; see `figures/soh_residuals.png`. The Re/Rct features were not selected. Their coverage is limited, and B0018 has impedance values for only 37 % of its cycles.

### SOC

132 held-out discharge segments, 31,921 samples. RFECV kept 5 features: voltage_measured_V, elapsed_s, temp_rise_C, voltage_drop_V, dvdt_past_V_per_s.

| Model | CV MAE | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|
| Linear Regression | 2.25 | 3.81 | 4.48 | 0.976 |
| Decision Tree | 1.21 | 2.24 | 3.14 | 0.988 |
| **Random Forest (selected by CV)** | 0.84 | **1.74** | **2.29** | **0.994** |

Units are SOC percentage points.

- Test error is about twice the CV error.
- Errors are largest at 20–40 % reference SOC (MAE 3.2, maximum 7.6 points).
- R² is high mainly because the reference spans 0–100 %.

SOC is shown in the dashboard only within the protocol scope stated in the assessment. The switch is `soc.dashboard_enabled` in `config.yaml`, and the saved metadata must also mark the reference as defensible.

The full tables are in [results/summary.md](results/summary.md) and `results/tables/`. The figures, listed with captions in `summary.md`, are:

- capacity and SOH vs cycle;
- measured V/I/T signals per battery;
- RFECV curves and rankings;
- actual vs predicted: the selected model, and all three models side by side (`*_actual_vs_predicted_all_models.png`);
- residuals;
- model comparisons;
- SOC trajectories;
- SOC reference-quality diagnostics.

## Dashboard

- **Dataset demonstration:** choose a battery and discharge cycle. The SOH panel shows the predicted SOH next to the reference SOH from measured capacity, plus a trend chart. The SOC panel shows predicted and coulomb-counting reference SOC at a chosen instant, plus the whole trajectory. Each panel states whether the battery is the held-out test battery or a training battery; training batteries give in-sample, optimistic results. Below the panels are the measured V/I/T signals for the selected discharge.
- **Manual input:** fields are generated from each model's selected features and show their training ranges. Missing or non-numeric values are rejected, and out-of-range values are flagged as extrapolation.

## Limitations

- **Four cells, one test cell.** All four batteries came from one test campaign at 24 °C with identical charge/discharge currents. The held-out result describes generalisation to one more cell under the same protocol, not to other chemistries, temperatures or load profiles. Grouped CV has only 3 folds.
- **Selection bias.** RFECV used all three training batteries, so the per-model CV scores computed on the selected features are slightly optimistic. The B0018 test is untouched.
- **SOH.** The model shows a systematic offset on the held-out cell. The 2 Ah rated capacity comes from the README; B0006 measures above it early on. Impedance is sparse, and B0018 runs a different operation sequence.
- **SOC.** The reference is approximate coulomb counting:
  - It normalises by the same cycle's measured capacity.
  - Its 0 % is defined at the 2.7 V cutoff, so the deeper cutoffs of B0006, B0007 and B0018 are not modelled.
  - It assumes a full charge at the start.
  - It applies only to 2 A constant-current discharges.
  - It does not cover dynamic loads or partial-charge starts.
- **Data issues.** These are kept and reported, not hidden:
  - Charge records 84 in B0005, B0006 and B0007 read > 8 V; their features are set to missing.
  - Truncated charge records and one B0018 charge record with non-finite samples have missing features.
  - The sign of `Current_load` is inconsistent between records.
- **Presentation.** The PowerPoint deck was used only for objectives and terminology; it contains no results.
