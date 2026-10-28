# Delivery Reliability & Early-Warning System for Supply Chains

> **Can I trust this delivery, and what should I do if I can't?**

A machine-learning project on the DataCo supply-chain dataset (about 180,000 order lines, 2015 to 2018). It measures
how reliable each shipping option really is, predicts which orders will arrive late and when, raises early warnings
with calibrated risk tiers, and turns the scores into recommended actions. A Streamlit app, **Delivery Reliability
Studio**, lets you explore all of it.

## Headline findings

| # | Finding | Evidence |
|---|---|---|
| 1 | **Lateness is set by the promise, not by slow delivery.** Second Class delivers as slowly as Standard (about 4 days) but promises 2 days instead of 4, so it is late twice as often. | Second Class late 79.8%, Standard 39.8% |
| 2 | **First Class is the only genuinely faster option**, but its 1-day promise is never kept. | Always 2 days, 100% late |
| 3 | **Same Day has a noon cut-off.** Ordered before 12:00 it arrives the same day; from 12:00 it arrives the next day and counts as late. | 47.9% late |
| 4 | **A 5-row lookup table is as good as LightGBM.** The remaining lateness is random, so about 0.78 ROC-AUC is the ceiling on this data. | ROC-AUC 0.777 (table) vs 0.779 (LightGBM) |
| 5 | **A random split inflates the score.** 89% of rows belong to multi-item orders, so related rows land on both sides of the split. | 0.801 random vs 0.779 time-based |
| 6 | **The high-risk alert is precise.** It flags 37% of orders, nine in ten of which are late, and catches 58% of all late orders. | Precision 0.898, recall 0.584, 0.11 false alarms per late order caught |

The honest conclusion is part of the result: the model confirms how lateness is generated in this dataset, and it
should not be oversold as having found hidden structure.

## What the project does

1. **Reliability analysis:** a scorecard per shipping option (promised vs real days, late rate with confidence
   intervals), arrival-within-N-days tables, and region/market reliability corrected for shipping-mode mix.
2. **Late-delivery prediction:** logistic regression, random forest and LightGBM against simple baselines, scored with
   tie-aware metrics (ROC-AUC, PR-AUC, precision at 80% recall, lift at top 20%).
3. **ETA with prediction intervals:** an 80% delivery-time range from three methods (empirical per-mode quantiles,
   LightGBM quantile regression, split conformal); coverage is checked on the test period.
4. **Early warning:** scores are calibrated (isotonic, fit on validation only) and mapped to High / Medium / Low tiers.
5. **Decision support:** rules turn scores into actions (RE-PROMISE, ALERT, WARN CUSTOMER, NO ACTION) and are
   evaluated for precision, recall and false alarms.
6. **Explainability:** TreeSHAP for any single order and for the model overall.
7. **Leakage audits:** two further logistics datasets are examined and found to have no honest predictive signal
   (see below), which is reported as a negative result.

## Datasets

| File (in `data/raw/`) | Role |
|---|---|
| `DataCoSupplyChainDataset.csv` | **Main dataset.** All predictive modelling, the scorecard and the decision rules. |
| `dynamic_supply_chain_logistics_dataset.csv` | **Audit case study only.** Features are unrelated to the targets, and `risk_classification` is a binned copy of `disruption_likelihood_score`. |
| `smart_logistics_dataset.csv` | **Audit case study only.** `Logistics_Delay` is an exact rule on two columns (Delayed shipment status or Heavy traffic). |
| `DescriptionDataCoSupplyChain.csv` | Reference only (data dictionary). |

The datasets share no keys and are never joined. The raw data is not stored in this repository.

**Data notes**
- DataCo is latin-1 encoded. Re-saving it in Excel swaps month and day in many dates (39% in one case);
  `src/data/load.py` repairs both formats and warns about anything unparseable.
- Cancelled and suspected-fraud orders (7,754) are labelled "on time" by the data, so late-delivery evaluation uses
  shipped orders only.
- Columns known only after delivery (`Days for shipping (real)`, `Delivery Status`, `Order Status`,
  `shipping date (DateOrders)`) are never used as features; a unit test enforces this.
- All evaluation uses a time-based split: the oldest 70% trains, the next 15% validates, the newest 15% tests.

## Repository structure

```
delivery_reliability/
├── config.yaml                 paths, seed, split sizes, decision-rule thresholds
├── requirements.txt
├── data/
│   ├── raw/                    input CSVs (not tracked)
│   ├── interim/  processed/    unused: notebooks build features in memory
├── notebooks/
│   ├── 01_eda_dataco.ipynb                  exploration of DataCo
│   ├── 02_eda_logistics.ipynb               audit of the logistics files
│   ├── 03_baselines.ipynb                   baseline classifiers and ETA baselines
│   ├── 04_eta_models.ipynb                  LightGBM models, ETA intervals, saved models
│   ├── 05_early_warning.ipynb               risk tiers, lookup table vs model, split comparison
│   └── 06_explainability_reliability.ipynb  scorecard, regions, SHAP, rule evaluation
├── src/
│   ├── data/        load.py, clean.py
│   ├── features/    build_features.py        features, leakage list, time split, preprocessing
│   ├── models/      evaluate.py, train_eta.py, train_risk.py, anomaly.py, export_artifacts.py
│   ├── decision/    rules.py                 risk tiers and recommended actions
│   ├── utils/       plotting.py
│   ├── app_data.py  app_charts.py  app_support.py     everything behind the Streamlit app
├── app/streamlit_app.py        Delivery Reliability Studio
├── models/                     trained models (generated, not tracked)
├── reports/                    figures, result tables, experiment log, final report
└── test/                       unit tests
```

## Setup (Windows)

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```
Place the CSVs in `data/raw/`. In VS Code select the `.venv` interpreter and kernel. If PowerShell blocks
activation, call `.venv\Scripts\python.exe` directly, as in the commands below.

## How to run

**1. Notebooks.** Run `01` to `06` in order with the `.venv` kernel. Notebook 04 saves
`models/late_delivery_lgbm.joblib` and `models/eta_lgbm.joblib`; 03 to 06 write tables and figures to `reports/`.

**2. Build the app data** (once, after notebooks 04 to 06):
```
.venv\Scripts\python.exe -m src.models.export_artifacts
```

**3. Start the app:**
```
.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

**4. Tests:**
```
.venv\Scripts\python.exe -m pytest test
```

## The app: Delivery Reliability Studio

| Page | What you can do |
|---|---|
| Overview | Headline numbers and the main findings |
| Shipping Lab | Compare options, change the promised days and watch the late rate move, find the promise that meets a late-rate target |
| Order Checker | Score one order live, compare every shipping option for it, see the arrival window and the SHAP drivers |
| Early Warning | Tune the alert threshold, see precision, recall and false alarms, estimate money value with your own assumptions, download a priority watchlist |
| Where & When | Regions, markets and departments (corrected for shipping-mode mix), monthly trend, hour-by-weekday heatmap |
| Model Lab | ROC and calibration curves, the 5-row lookup table, the time-split timeline, the leakage guard and the limitations |

## Methods in brief

- **Models:** logistic regression, random forest and LightGBM, compared with a mode-only baseline and a 5-row lookup
  table. LightGBM is tuned with a small random search on validation ROC-AUC.
- **Preprocessing:** scikit-learn pipelines fit on the training split only (imputation, one-hot encoding with rare
  categories grouped).
- **Metrics:** ROC-AUC, PR-AUC, precision at 80% recall and lift at the top 20%. F1 is avoided because flagging
  every order already scores 0.71. Ties in scores are broken randomly so that models with few distinct scores are not
  penalised unfairly.
- **Calibration:** isotonic regression fit on the validation split, so a score of 0.8 means about 80% of such orders
  are late.
- **Intervals:** 80% delivery-time ranges; the conformal method is calibrated per shipping mode.

## Limitations

- DataCo is synthetic: lateness comes from the promise and the Same Day cut-off plus random noise, so the achievable
  accuracy is capped near 0.78 ROC-AUC and about 42% of late orders (Standard Class) cannot be predicted.
- There are no supplier IDs and no shipping-cost column, so supplier ranking and cost optimisation are out of scope.
- The two logistics datasets are used only as leakage audits.
- The money estimate in the app uses assumptions you enter; it is a way to reason about the alert, not a forecast.

## Reproducibility

Random seed 42 (see `config.yaml`); all splits are chronological. Trained models and raw data are not committed:
rerun the notebooks and the export command to regenerate them.
