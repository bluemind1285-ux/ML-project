# NASA Battery SOH/SOC Estimation and BMS Dashboard

## Purpose and implementation contract

Build a small, reproducible Python project using the NASA battery aging dataset to estimate state of health (SOH), investigate state of charge (SOC), and present validated results in a Streamlit battery management system (BMS) dashboard.

Implement this specification as written. Inspect the actual files before choosing parsing details or making claims about results. The referenced conversation identifies these attachments: `B0005.mat`, `B0006.mat`, `B0007.mat`, `B0018.mat`, `README.txt`, and `[24BEE0104],[24BEE0027](1).pptx`. The available preview of the attachments does not guarantee they are in this project directory. Search the supplied workspace/input folders first. Preserve original `.mat` files byte-for-byte under `data/raw/`; if they are not present or accessible, do not recreate or substitute them. Record which inputs are missing and implement discovery/configuration so the user can place them in `data/raw/`.

Read the supplied README and presentation, if present, and use their dataset description, project objectives, and terminology as context. Do not treat presentation claims as measured results. Do not invent dataset fields, evaluation results, thresholds, or performance claims. Record inspection findings in a short `results/data_inspection.md`.

Use Python, not MATLAB. Keep code in focused modules rather than one large script. Keep SOH and SOC data preparation, targets, feature selection, training, evaluation, and saved artifacts separate. Join them only in the final dashboard. Do not create Model A/Model B comparisons or add unnecessary model families.

## Project objectives

1. Estimate SOH using measured discharge capacity as the reference target.
2. Investigate SOC only after creating a clearly documented coulomb-counting reference from time-series current and time. SOC is a conditional outcome: include SOC prediction in the BMS only if the reference is defensible and held-out validation supports its use. Otherwise present the limitation plainly and omit an SOC prediction card.
3. Compare Linear Regression, Decision Tree Regressor, and Random Forest Regressor separately for SOH and, if viable, SOC.
4. Produce graphs and tables that let a teacher inspect the data, method, and validation without implying stronger evidence than the dataset supports.
5. Deliver an easy-to-run project suitable for a one-day implementation. Prefer a complete, honest baseline over optional complexity.

## Recommended repository layout

```text
project/
├── PROJECT_SPECIFICATION.md
├── README.md
├── requirements.txt
├── config.yaml                    # paths and small documented settings
├── data/
│   ├── raw/                       # original NASA .mat and supplied documentation
│   └── processed/                 # generated CSV/parquet; never overwrite raw
├── models/
│   ├── soh/                       # fitted pipeline and metadata
│   └── soc/                       # fitted pipeline and metadata, only if viable
├── results/
│   ├── data_inspection.md
│   ├── tables/
│   └── figures/
├── src/
│   └── battery_project/
│       ├── __init__.py
│       ├── config.py
│       ├── data_loader.py
│       ├── cleaning.py
│       ├── feature_engineering.py
│       ├── feature_selection.py
│       ├── models.py
│       ├── evaluation.py
│       ├── visualization.py
│       ├── soh_pipeline.py
│       ├── soc_pipeline.py
│       └── artifacts.py
├── scripts/
│   ├── inspect_data.py
│   ├── prepare_data.py
│   ├── train_soh.py
│   ├── train_soc.py
│   └── make_report.py
└── dashboard.py
```

Modules should have clear responsibilities. `data_loader.py` handles MATLAB file parsing and normalizes records; `cleaning.py` validates and cleans records; `feature_engineering.py` creates cycle-level and time-series features; `feature_selection.py` owns separate leakage-safe SOH/SOC selection routines; `models.py` defines the three estimators; `evaluation.py` calculates metrics and writes comparison tables; `visualization.py` generates figures; each pipeline module orchestrates only its own target path; `artifacts.py` saves/loads fitted sklearn pipelines and metadata. Scripts are thin command-line entry points. Avoid duplicated parsing or metric logic.

## Data discovery and MATLAB structure

1. Locate supplied `.mat` files and documentation. Use the actual files found; do not assume all four battery IDs are present.
2. Inspect MATLAB keys, nested structs, field names, data types, array shapes, and operation types before extraction. NASA files commonly store nested cycle records, but write a parser based on observed structure, not a hard-coded assumed layout.
3. Support MATLAB v7.3/HDF5 if encountered (for example, use `h5py` when `scipy.io.loadmat` cannot read the file). Keep this fallback only as needed by actual inputs.
4. Normalize each cycle into a record with battery ID, cycle index/number, operation type, cycle metadata, and the available measurements. Preserve raw signal arrays in a structured intermediate representation long enough to derive features; do not flatten nested objects into misleading strings.
5. Extract documented fields when present, for example charge/discharge voltage, current, temperature, time, load voltage/current, discharge capacity, and impedance quantities such as `Re` and `Rct`. Field names vary; map based on inspection and record missing fields. An absent measurement is not zero.
6. Match charge, discharge, and impedance records by battery and cycle only where the observed cycle structure justifies a match. For unmatched records retain missing values and record match coverage; do not silently shift cycles or forward-fill measurements across arbitrary gaps.
7. Save a machine-readable inspection inventory: source files, battery IDs, cycle counts by operation, field availability, units when documented, time ranges, missingness, and parser warnings.

## Cleaning and preprocessing

- Preserve source files unchanged. Write normalized and feature tables under `data/processed/` with stable column names and explicit units in names or a data dictionary.
- Keep one row per battery-cycle in the SOH feature table. Keep SOC at the time-sample level or a clearly defined window/sample level; do not reduce it to a single cycle summary if that destroys the target trajectory.
- Coerce numeric fields safely; mark invalid conversions missing. Remove duplicate records only when an explicit key and equality check establish they are duplicates; report removed counts.
- Exclude non-finite inputs/targets from training rows and report counts. Do not impute a missing target. For features, fit any imputer on training folds only, preferably in an sklearn `Pipeline`.
- Remove constant/all-missing features using training data only. Correlation pruning is optional and only use it if the feature set makes it useful; fit it on training data and document its threshold/configuration. Do not remove features based on target/test-set inspection.
- Preserve plausible unusual measurements for inspection. Flag physically suspect values and investigate units/sign conventions; do not delete outliers merely because they reduce model performance. Any explicit range filtering must be justified from supplied metadata and recorded.
- Make feature and target units explicit. Do not use cycle ID strings as numeric magnitudes. Battery ID is a grouping key, not a numeric feature.

## Feature engineering

### SOH features and reference target

Primary target: discharge capacity (Ah) converted to SOH percentage:

`SOH_percent = 100 * discharge_capacity_Ah / rated_capacity_Ah`

Determine rated capacity from reliable supplied dataset documentation or project materials and verify it against observed metadata. Do not blindly assume a nominal value. If no defensible rated capacity is documented, report that and use capacity in Ah as the directly supported target; do not label a guessed ratio SOH. Do not clip measured SOH to 0–100% to make results look better; investigate and report unusual values.

Candidate predictors may use useful charge, discharge, and impedance measurements, plus cycle number where appropriate. Extract only fields actually present. For charge and discharge time-series, compute defensible summaries such as mean/min/max/std, initial/final values, duration, temperature rise, and current/voltage summaries; use discharge capacity only as the target, never as an input or as a transformed proxy. For impedance use available measured impedance, `Re`, `Rct`, or other inspected quantities. Avoid creating redundant aliases of the target.

Merge cycle-level summaries by battery and cycle after extraction. If charge/impedance records are sparse, retain rows with missing predictors and impute within training folds or limit training to adequately observed rows with the coverage stated. Never use future cycles to fill earlier-cycle features.

### SOC reference and predictors

The dataset does not provide a directly measured SOC label in the described context. Build a reference per continuous discharge time series from current and elapsed time only after inspecting current sign, units, time ordering, resets, gaps, and capacity consistency. Use trapezoidal integration over actual time deltas:

`Q_discharged_Ah(t) = cumulative_integral(discharge_current_A, time_seconds) / 3600`

Set a documented initial SOC of 100% at the start of a discharge segment only when the data and task interpretation support a full-charge start. Set usable capacity from a defensible source for that battery/cycle (prefer measured discharge capacity where available, with the limitation that this makes the reference cycle-specific). Then:

`SOC_reference_percent(t) = 100 * (1 - Q_discharged_Ah(t) / usable_capacity_Ah)`

Do not silently assume current polarity; normalize direction after inspecting it and record the rule. Split at time resets, large gaps, or operation changes rather than integrating across them. Check monotonicity, range, and integrated charge against available discharge capacity. If initial SOC, capacity, or timing cannot be justified, label SOC reference as approximate and do not claim true SOC ground truth. Any clipping to 0–100 for visualization must be separately documented; preserve and inspect unclipped values for validation.

SOC predictors should be contemporaneous/causal voltage, current, temperature, load signals, and elapsed/time-derived features from the same time sample or a past-only window. Do not use future samples, cycle discharge capacity, cumulative future charge, or the SOC target itself as predictors. If cumulative discharged charge is included, it must be computed only from measurements available up to that timestamp. Keep SOC features and selection entirely separate from SOH.

## Splitting, leakage prevention, and feature selection

Use deterministic splits and write the exact battery/cycle/sample grouping to metadata. The primary generalization check should hold out whole battery IDs when there are enough batteries for a meaningful split. With four listed batteries, use a documented battery-group holdout (e.g., leave one battery out for final testing, if feasible) and use grouped cross-validation on training batteries. Do not randomly split cycle rows across train and test as the only result: adjacent cycles from one battery are highly related. If a battery-level split leaves too little data, state the limitation and provide a clearly labeled secondary within-battery chronological evaluation, with earlier cycles for training and later cycles for test. Never let samples from one discharge trajectory appear in both sides of the SOC split; group by battery and discharge cycle/segment.

Fit preprocessing, correlation filtering if used, and RFECV using training data only. RFECV is the main feature-selection method, run separately for SOH and SOC. Choose an estimator with `coef_` or `feature_importances_` (use a modest tree-based estimator for selection if needed) and a CV splitter consistent with groups. Pass groups through selection CV where supported. Keep a final untouched test set out of RFECV, hyperparameter decisions, and model selection. If grouped RFECV is infeasible for the actual number of groups, document the constraint and use the simplest valid training-only fallback; do not leak test data. Save the selected feature names and CV configuration.

Use modest, fixed model settings to fit the one-day scope. If tuning is included, it must be small, grouped, and confined to training data; it is not required. Choose the best model using training-only cross-validation, then evaluate that chosen pipeline once on the final test group. Report all three model test metrics for comparison only if the comparison protocol is fixed in advance; do not repeatedly tune based on test results. Include `random_state` for stochastic models.

## Model training and metrics

Train three regressors for each viable target: `LinearRegression`, `DecisionTreeRegressor`, and `RandomForestRegressor`. Fit all preprocessing and RFECV independently inside each target pipeline. Do not share selected features, fitted imputers, or target logic across SOH and SOC.

For both targets report MAE, RMSE, and R² on the held-out test set, plus training/CV scores where useful and clearly labeled. Include units (`SOH percentage points`; `SOC percentage points`). State sample counts and held-out group IDs beside metrics. R² can be negative and should be shown as calculated. If a test set is too small or R² undefined, report that honestly. Do not establish arbitrary acceptance thresholds in advance. Judge SOC viability from actual reference quality, held-out errors, residual behavior, and whether the result is useful relative to the 0–100% scale; explain the reasoning and do not claim an SOC model is validated solely because a metric exists.

## Required outputs: tables and figures

Save figures as readable PNG files (and optionally PDF) under `results/figures/`, with titles, axes, units, legends, and captions or a report index. Save tables as CSV under `results/tables/` and provide a concise summary in `README.md` or `results/`.

### Required SOH outputs

- Cleaned per-cycle SOH feature table and data dictionary.
- Capacity (Ah) vs cycle, grouped/colored by battery.
- SOH (%) vs cycle, grouped/colored by battery.
- Actual vs predicted SOH on held-out data, with identity line and battery/group labels where legible.
- SOH residuals (actual minus predicted) vs predicted value and/or cycle; include a zero reference line.
- SOH model comparison table and a clear MAE/RMSE/R² comparison plot. Lower MAE/RMSE is better; higher R² is better.
- Selected SOH feature list and a readable selected-feature visualization or table. Do not call RFECV selection scores generic feature importance.

### Required SOC outputs, if SOC reference can be constructed

- SOC reference vs elapsed time for representative held-out discharge segments, with segment/battery identification.
- Predicted vs reference SOC on held-out data with identity line.
- SOC residual (predicted minus reference or clearly defined sign) vs predicted SOC and/or time, with zero line.
- SOC model comparison table/plot using MAE, RMSE, and R².
- Selected SOC features list/table.
- A reference-quality diagnostic showing current/time integration and, where available, integrated discharged Ah versus measured capacity. Show suspicious range/drift/reset behavior rather than hiding it.

If SOC construction is not defensible, still save `results/soc_assessment.md` describing inspected current/time/capacity evidence, the problem found, and why training/dashboard inclusion was skipped. Do not produce fabricated SOC metrics or prediction plots.

### Required signal/context plots

For representative actual records, create voltage, current, and temperature vs time plots for charge/discharge where those signals exist. Label operation, battery, cycle, units, and distinguish measured from derived values. These are data-understanding plots, not model-accuracy evidence.

## Saved models and reproducibility

Save each fitted best estimator as a complete sklearn pipeline (preprocessing + selected columns + regressor), using `joblib`, under `models/soh/` and, only when viable, `models/soc/`. Also save JSON metadata: model name, target, feature order, selected features, training battery/cycle groups, test groups, metrics, units, random seed, library versions, input filenames, and SOC reference assumptions. Dashboard inference must load these artifacts; it must not retrain on startup. Warn in documentation that loading joblib artifacts requires trusted files from this project.

Provide clear commands in README, for example:

```text
python scripts/inspect_data.py
python scripts/prepare_data.py
python scripts/train_soh.py
python scripts/train_soc.py
python scripts/make_report.py
streamlit run dashboard.py
```

Commands should gracefully report missing input files and skip SOC training if no defensible reference/data exists. Avoid test suites or optional infrastructure beyond what is needed to run the workflow.

## Streamlit BMS dashboard

Create a simple dashboard that integrates results only at presentation time. It must have separate SOH and SOC sections and use their respective saved pipelines. Include:

- Dataset demonstration mode: select battery and cycle/segment from processed held-out examples. Clearly show predicted values and, for validation/demo records, the measured/reference values separately. Mark whether the selection is training or held-out data.
- Manual prediction mode only when required inputs can be supplied reliably. Show fields generated from the saved model's selected feature schema, units, and reasonable observed training ranges. Reject missing/non-finite inputs and flag values outside observed training ranges. Never imply arbitrary inputs are trustworthy.
- SOH prediction display in percent only if SOH is supported by the documented capacity reference. Include selected model, an uncertainty caveat, and available cycle context.
- SOC prediction display only if SOC assessment has approved it based on actual evidence and a defensible reference. Otherwise show an honest status panel explaining that SOC is not displayed as a validated estimate.
- Context charts for capacity/SOH trends and available voltage/current/temperature signals. No raw training or test labels should be disguised as model predictions.
- Dashboard errors for missing model artifacts should be actionable (which training command is needed); no silent fallback to fake/demo numbers.

The dashboard is a presentation and inference layer, not another training pipeline. SOH and SOC remain independent from data extraction through artifacts; only their final results appear together on the dashboard.

## Completion checklist

- [ ] Actual supplied files inspected; raw `.mat` files preserved unchanged.
- [ ] README and presentation reviewed if available; findings and input inventory recorded.
- [ ] MATLAB nested records parsed from observed schema; missing fields and parsing issues reported.
- [ ] Cleaned processed tables and data dictionary saved.
- [ ] SOH target formula and rated-capacity provenance documented; capacity excluded from predictors.
- [ ] SOH has separate training-only RFECV, three-model comparison, grouped/chronological held-out evaluation, required metrics and figures, and saved pipeline.
- [ ] SOC reference sign, integration, initial condition, capacity, and reset assumptions inspected and documented.
- [ ] SOC is trained/evaluated with independent leakage-safe processing only if reference is defensible; otherwise a clear assessment is delivered.
- [ ] Required data plots, validation plots, and CSV metric/feature tables exist for every viable path.
- [ ] Streamlit dashboard loads persisted artifacts, labels reference vs prediction clearly, and includes SOC only when validation supports it.
- [ ] README gives setup/run instructions and explains limitations without fabricated findings.

## One-day priority order

1. Find/inspect inputs and produce the dataset inventory.
2. Parse and normalize MATLAB cycles; save processed records.
3. Build and validate SOH features/target, then train/evaluate/save SOH models and figures.
4. Independently inspect SOC current/time convention and generate a candidate coulomb-counting reference; proceed with SOC models only if defensible.
5. Generate required summary plots/tables and a short results/limitations report.
6. Build the Streamlit dashboard around saved artifacts; confirm the documented run sequence works.

Do not sacrifice split integrity, leakage prevention, or honest reporting to add optional features. If time or data availability prevents a section, leave a clear status and limitation rather than generating placeholder results.
