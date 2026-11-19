# Delivery Reliability & Early-Warning System for Supply Chains

## 1. Summary

This project asks one practical question: **can I trust this delivery, and what should I do if I can't?** Using the
DataCo supply-chain dataset (about 180,000 order lines, 2015 to 2018), it measures how reliable each shipping option
really is, predicts which orders will arrive late and when, raises calibrated early warnings, and turns the scores
into recommended actions. A Streamlit app, Delivery Reliability Studio, lets a user explore every result.

The main findings:

1. **Lateness is set by the promise, not by slow delivery.** Second Class delivers as slowly as Standard Class
   (about 4 days) but promises 2 days instead of 4, so it is late 79.8% of the time against 39.8%.
2. **First Class is the only genuinely faster option**, but it always takes 2 days against a 1-day promise, so every
   First Class order is late.
3. **Same Day has a noon cut-off.** Orders placed before 12:00 arrive the same day; from 12:00 they arrive the next
   day and count as late.
4. **A 5-row lookup table matches the machine-learning model** (ROC-AUC 0.777 against 0.779). The remaining
   lateness is random, so about 0.78 is the ceiling on this data.
5. **Evaluation must be time-based.** A random split inflates ROC-AUC from 0.779 to 0.801 because items of the same
   order land on both sides of the split.
6. **The high-risk alert is precise.** It flags 37.4% of orders; 89.8% of them are late, and it catches 58.4% of all
   late orders with 0.11 false alarms per late order caught.

These results are reported as they are. The model is valuable for showing how lateness arises and for producing
calibrated risk scores, but it has not found hidden structure beyond the promise and the cut-off, and the report does
not claim otherwise.

## 2. Problem and goals

A retailer promises a delivery time when an order is placed. Three questions follow:

- **Reliability:** how often is each shipping option late, and why?
- **Prediction:** given an order at the moment it is placed, how likely is it to be late, and in what range will it
  arrive?
- **Action:** which orders should trigger an alert, and what should be done?

Success means answers that hold up on future orders (a time-based test), that are honest about uncertainty, and that
can be used by a non-technical reader.

## 3. Data

| File | Role |
|---|---|
| `DataCoSupplyChainDataset.csv` | Main dataset: all predictive modelling, the scorecard and the decision rules |
| `dynamic_supply_chain_logistics_dataset.csv` | Audit case study only |
| `smart_logistics_dataset.csv` | Audit case study only |
| `DescriptionDataCoSupplyChain.csv` | Reference (data dictionary) |

The datasets share no keys and are never joined.

**Problems found and how they were handled**

| Problem | What was found | Handling |
|---|---|---|
| Corrupted dates | 71,103 of 180,519 order dates (39%) failed to parse. The file had been re-saved in Excel, which rewrote some month/day dates as day/month | A parser that reads both formats; all rows now parse |
| Leakage columns | `Days for shipping (real)`, `Delivery Status`, `Order Status` and `shipping date` are only known after delivery, and `Late_delivery_risk` equals `Delivery Status == "Late delivery"` | Never used as features; a unit test fails if one reaches the model |
| Cancelled orders | 7,754 cancelled or suspected-fraud orders (all payment type TRANSFER) are labelled "on time" although they never shipped | Late-delivery evaluation uses the 172,765 shipped orders. TRANSFER's late rate rises from 0.485 to 0.574 once they are removed |
| Random split leakage | 89% of rows belong to multi-item orders | All splits are chronological |

**Time split.** The oldest 70% of orders trains the models, the next 15% tunes and calibrates them, and the newest
15% (about June 2017 to January 2018) is the test set.

## 4. Methods

**Features.** Only information available when the order is placed: shipping mode, promised days, payment type,
market, region, customer segment, department, category, order value, quantity, discount, and the hour, weekday and
month of the order. Preprocessing is a scikit-learn pipeline fit on the training split only (imputation, one-hot
encoding with rare categories grouped).

**Models.** Logistic regression, random forest and LightGBM (small random search on validation ROC-AUC), compared with
a shipping-mode-only baseline and a 5-row lookup table (shipping mode, with Same Day split at the noon order cut-off).

**Metrics.** ROC-AUC, PR-AUC, precision at 80% recall and lift at the top 20% of scores. F1 is not used, because
flagging every order already scores 0.71. PR-AUC, precision-at-recall and lift are tie-aware: a baseline with only a
few distinct scores is not penalised, because ties are broken randomly before ranking.

**Delivery-time range.** A LightGBM point estimate plus an 80% interval from three methods: empirical per-mode
quantiles, LightGBM quantile regression and split conformal prediction per shipping mode.

**Calibration and tiers.** Late-delivery scores are calibrated with isotonic regression fit on the validation split,
so a score of 0.8 means about 80% of such orders are late. Calibrated scores map to High (at least 0.70), Medium and
Low (below 0.20) risk tiers.

**Decision rules.** RE-PROMISE (High tier and a high-value order), ALERT (High tier), WARN CUSTOMER (wide delivery
window) and NO ACTION. "Expedite" was rejected as an action because nothing is faster than First Class: the useful
actions are to correct the promise and to tell the customer.

**Explainability.** TreeSHAP contributions for individual orders and for the model overall.

## 5. Results

### 5.1 Shipping-option reliability

| Shipping mode | Orders | Late rate | Behaviour |
|---|---|---|---|
| Standard Class | 103,153 | 39.8% | Promises 4 days; real delivery is uniform over 2 to 6 days |
| Second Class | 33,806 | 79.8% | Promises 2 days; real delivery takes the same 3.99 days on average as Standard |
| First Class | 26,513 | 100% | Promises 1 day; always takes exactly 2 days |
| Same Day | 9,293 | 47.9% | Before noon it arrives the same day; from noon, the next day |

Profit per order is similar across modes (about 21 to 23 with overlapping confidence intervals), so late delivery is a
service problem here and not a margin problem. A "promise designer" in the app shows the promise each option would
need to reach a chosen late rate: Second Class would need to promise about as long as Standard.

### 5.2 What the model learns

| Model | Test ROC-AUC |
|---|---|
| 5-row lookup table | 0.777 |
| LightGBM | 0.779 |

The two are almost identical, so the model has learned the promise and the noon cut-off and nothing more. Removing
the order-time features lowers LightGBM to 0.743, which confirms that order hour matters, through the Same Day
cut-off. For Second and Standard Class, delivery time is random, so no model can do better than the lookup table.

### 5.3 Why random splits mislead

| Split | ROC-AUC |
|---|---|
| Time-based (honest) | 0.779 |
| Random | 0.801 |

Items from one order share a shipping mode, order time and delivery time. A random split lets the model recognise
siblings of a test row in the training set. Since 89% of rows are in multi-item orders, the random split overstates
performance by about 0.022.

### 5.4 Early warning and decision rules

| Rule | Orders flagged | Precision | Late orders caught | False alarms per late order caught |
|---|---|---|---|---|
| High-tier alert / re-promise | 37.4% | 0.898 | 58.4% | 0.113 |
| Warn customer (wide interval) | 79.4% | 0.497 | 68.4% | 1.01 |
| Any action | 97.5% | 0.591 | 100% | 0.69 |

The high-tier alert is the useful rule. The other 42% of late orders are Standard Class orders that are late about
40% of the time at random; flagging them would cost many false alarms. The "warn customer" rule is about date
uncertainty and not lateness: it fires on Second and Standard Class, whose delivery time is random, so its precision
is meaningless as a lateness measure. The app shows it as a delivery window on every order. Calibrated scores form
a few clusters (about 0.00, 0.38 to 0.40, 0.80 and 1.00, one per shipping option), which is why the threshold curves
step instead of sloping.

### 5.5 Delivery-time ranges

The 80% interval reaches about 80% coverage on the test period (0.797). Because delivery times are whole days and
random within Second and Standard Class, the intervals are wide for those modes (several days) and have no width for
First Class. The width is a property of the data and not of the method.

### 5.6 Regions, markets and departments

Raw late rates by region mislead, because a region that uses more Second Class looks unreliable only through the
short promise. The analysis therefore compares each region's late rate with the rate expected from its shipping-mode
mix and reports the gap with a z-score (see `reports/tables/region_reliability.csv` and the Where & When page of the
app).

### 5.7 Leakage audits of the other two datasets

| Dataset | Finding |
|---|---|
| Dynamic supply-chain logistics | Features are unrelated to the targets (maximum absolute correlation 0.014, no autocorrelation); `risk_classification` is a binned copy of `disruption_likelihood_score` |
| Smart logistics | `Logistics_Delay` is an exact rule (Delayed shipment status or Heavy traffic); without those columns ROC-AUC is 0.488, no better than chance |

Both are reported as negative results and used only to illustrate how leakage and synthetic labels hide in tabular
data.

## 6. The app

Delivery Reliability Studio has six pages: Overview, Shipping Lab (scorecard, promise simulator, promise designer,
profit), Order Checker (live scoring, comparison of every shipping option for one order, SHAP drivers), Early Warning
(threshold tuning, money estimate with user-set assumptions, priority watchlist), Where & When (regions, markets,
calendar, departments) and Model Lab (ROC, calibration, the lookup table, leakage guard, limitations).

## 7. Lessons

- **Check for a simple baseline first.** A 5-row table was as good as the model; without it, the model would have
  looked like a success.
- **Tied scores bias ranking metrics.** PR-AUC, precision at recall and lift were initially computed without
  tie-breaking, which understated the mode-only baseline (precision at 80% recall 0.576, against 0.653 once ties were
  broken) and inflated the apparent gain from LightGBM.
- **Choose thresholds after looking at the scores.** The first tier thresholds (0.75 and 0.45) did not match the
  clustered score distribution and were changed to 0.70 and 0.20.
- **Look at the labels.** Cancelled orders labelled "on time", a target that is a copy of a post-delivery column, and
  dates silently corrupted by Excel would each have produced wrong conclusions.

## 8. Limitations and future work

- DataCo is synthetic: lateness comes from the promise and the Same Day cut-off plus random noise. Accuracy is capped
  near 0.78 ROC-AUC, and about 42% of late orders cannot be predicted.
- There are no supplier IDs and no shipping-cost column, so supplier ranking and cost optimisation are out of scope.
- The two logistics datasets contribute only as audits.
- The money estimate in the app depends on assumptions the user supplies; it is a reasoning aid, not a forecast.
- Future work: test the approach on real carrier data with in-transit events (the early-warning idea is strongest
  once there is something to observe after the order ships), split by order ID to remove the multi-item effect from
  random-split comparisons, and add anomaly detection (`src/models/anomaly.py` is a stub).

## 9. Reproducibility

Seed 42 (`config.yaml`); all splits are chronological. Run notebooks 01 to 06, then
`python -m src.models.export_artifacts`, then `streamlit run app/streamlit_app.py`. Unit tests are in `test/`.
Result tables are in `reports/tables/` and can be exported to markdown with `python -m src.utils.tables_to_markdown`.
