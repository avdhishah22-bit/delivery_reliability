# Delivery Reliability and Early Warning for Supply Chains

**Question:** *can I trust this delivery, and what should I do if I can't?*

Using the DataCo supply-chain dataset (about 180,000 order lines, 2015 to 2018), the project measures how reliable each
shipping option really is, predicts which orders will arrive late and when, raises calibrated early warnings, and turns
them into recommended actions. A Streamlit app lets you explore all of it.

## What we found

| # | Finding |
|---|---|
| 1 | **Lateness is set by the promise, not by slow delivery.** Second Class takes as long as Standard Class (about 4 days) but promises 2 days instead of 4, so it is late 79.8% of the time against 39.8%. |
| 2 | **First Class** always takes 2 days against a 1-day promise: every order is late. **Same Day** has a noon cut-off: from 12:00 it arrives the next day. |
| 3 | **A 5-row lookup table matches LightGBM** (ROC-AUC 0.777 against 0.779). The rest of the lateness is random. |
| 4 | **A random split inflates the score** (0.799 against 0.779), because 89% of rows are in multi-item orders. All results use a time split. |
| 5 | **The High-risk alert is precise:** it flags 37.9% of orders, 89.8% of them are late, and it catches 59.1% of all late orders. |
| 6 | **Three traps in the data:** Excel-damaged dates (39% failed to parse), leakage columns, and cancelled orders labelled "on time". |

Details and the experiment log are in [`REPORT.md`](REPORT.md).

## The project in one picture

```
data/DataCoSupplyChainDataset.csv
     |
     v   core.py  (load, repair dates, features, time split)
 1_data_audit_and_reliability.ipynb      find the traps, prepare the data, measure each shipping option
     |
     v
 2_modelling_and_baseline.ipynb          baselines -> LightGBM, delivery-time windows   -> models/models.joblib
     |
     v
 3_calibration_and_early_warning.ipynb   calibrate, risk tiers, SHAP, decision rules      -> models/app_data.joblib
     |
     v
 4_robustness_and_validation.ipynb       bootstrap CIs, rolling validation, decision curve, drift -> models/evidence.joblib
     |
     v
 5_customer_impact_of_delays.ipynb       does a late delivery change repeat orders? (natural experiment) -> models/customer_impact.joblib
     |
     v
 app.py                                  Streamlit app that reads the .joblib files above
```

## Repository layout

```
delivery_reliability/
├── README.md                       this file
├── REPORT.md                       final report, experiment log, who did what
├── core.py                         ALL the logic (11 labelled sections); notebooks and app import it
├── 1_data_audit_and_reliability.ipynb      notebook 1
├── 2_modelling_and_baseline.ipynb          notebook 2
├── 3_calibration_and_early_warning.ipynb   notebook 3
├── 4_robustness_and_validation.ipynb       notebook 4
├── 5_customer_impact_of_delays.ipynb       notebook 5
├── app.py                          Streamlit app (6 pages)
├── test_core.py                    unit tests for core.py
├── requirements.txt
├── .streamlit/config.toml          app theme
├── data/                           put the DataCo CSV here
└── models/                         created by notebooks 2 to 5 (.joblib files)
```

`core.py` is organised as: 1 Settings, 2 Loading, 3 Splitting, 4 Metrics, 5 Models, 6 Early warning, 7 Analysis,
8 Order scoring, 9 App data, 10 Deeper checks, 11 Customer impact. Read it top to bottom; it follows the order the project happens in.

## Setup

Python 3.12 is recommended (the saved models were trained with the versions pinned in `requirements.txt`).

```
pip install -r requirements.txt
```

**Data:** `data/DataCoSupplyChainDataset.csv` is included in the repository. It is the *DataCo Smart Supply Chain for Big Data
Analysis* dataset (Constante, Silva and Pereira, Mendeley Data), a synthetic dataset. If you need to download it again, put the CSV in `data/`.

## Run

1. Open and run the notebooks **in order**, 1 to 5. Notebook 2 takes a few minutes (it tunes LightGBM) and notebook 4 a few more (it refits the model five times). Notebook 5 is independent of 2 to 4 and only needs the data.
2. `streamlit run app.py`
3. Tests: `python -m pytest -q`

## The app

| Page | What it answers |
|---|---|
| Overview | Headline numbers; promise against reality for each option |
| Shipping options | When each option really arrives; promise designer; profit by option |
| Check an order | Score one order: chance of being late, delivery window, recommended action, SHAP drivers, every option compared |
| Early warnings | Alert threshold, precision and recall, cost of mistakes, tiers, calibration, watchlist |
| Evidence | Is the model really no better than the 5-row table? Bootstrap, rolling validation, decision curve, groups, drift (needs notebook 4) |
| Where and when | Region / market / department fairness after controlling for shipping mode; weekday-hour heat map; trends |

## Deploy (Streamlit Community Cloud)

1. Commit `models/models.joblib` and `models/app_data.joblib` (they are not git-ignored).
2. Pin the library versions you trained with, so the saved models load: run
   `pip freeze | findstr /i "pandas numpy scikit-learn scipy lightgbm joblib streamlit plotly"` and put those exact
   versions in `requirements.txt` (you may delete matplotlib and pytest there).
3. On share.streamlit.io choose this repository, branch `main`, main file path **`app.py`**, and a Python version
   close to your local one (3.12 if that is what you used).

## Method in short

- **Time-based split:** oldest 70% train, next 15% validation (tuning and calibration), newest 15% test.
- **Shipped orders only** for lateness: cancelled orders never ship, yet are labelled "on time".
- **Metrics:** ROC-AUC, PR-AUC, precision at 80% recall and lift in the top 20%, with ties broken at random so that
  rule-based baselines are not penalised. F1 is not used (flagging every order already scores 0.71).
- **Leakage guard:** `core.py` stops with an error if a post-delivery column is added to the features.
- **Calibration:** isotonic regression on the validation split, so a score of 0.8 means about 80% are late.
- **Delivery window:** 80% intervals three ways (per-mode quantiles, quantile regression, split conformal).

## Limitations

DataCo is synthetic: lateness comes from the promise and the Same Day cut-off plus random noise, so about 0.78 ROC-AUC
is the ceiling and about 41% of late orders cannot be predicted. There are no supplier IDs or shipping costs.
