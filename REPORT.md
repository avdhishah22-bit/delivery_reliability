# Delivery Reliability and Early Warning for Supply Chains

## 1. Summary

This project asks one practical question: **can I trust this delivery, and what should I do if I can't?** Using the
DataCo supply-chain dataset (about 180,000 order lines, 2015 to 2018), it measures how reliable each shipping option
really is, predicts which orders will arrive late and when, raises calibrated early warnings, and turns the scores into
recommended actions. A Streamlit app lets a user explore every result.

The main findings:

1. **Lateness is set by the promise, not by slow delivery.** Second Class delivers as slowly as Standard Class (about
   4 days) but promises 2 days instead of 4, so it is late 79.8% of the time against 39.8%.
2. **First Class is the only genuinely faster option**, but it always takes 2 days against a 1-day promise, so every
   First Class order is late.
3. **Same Day has a noon cut-off.** Orders placed before 12:00 arrive the same day; from 12:00 they arrive the next day
   and count as late.
4. **A 5-row lookup table matches the machine-learning model** (ROC-AUC 0.777 against 0.779). The remaining lateness is
   random, so about 0.78 is the ceiling on this data.
5. **Evaluation must be time-based.** A random split inflates ROC-AUC from 0.779 to 0.799, because items of the same
   order land on both sides of the split.
6. **The conclusion survives stress tests.** The model-minus-table AUC gap is +0.002 with a 95% interval of -0.003 to +0.007
   (includes zero); across five rolling time windows the model wins 2 and loses 3; and inside any one shipping option the
   model ranks orders no better than chance (AUC 0.505). Details in section 5.7.
7. **The high-risk alert is precise.** It flags 37.9% of orders; 89.8% of them are late, and it catches 59.1% of all late
   orders with 0.11 false alarms per late order caught.

These results are reported as they are. The model is valuable for showing how lateness arises and for producing
calibrated risk scores, but it has not found structure beyond the promise and the cut-off, and the report does not claim
otherwise.

## 2. Problem and goals

A retailer promises a delivery time when an order is placed. Three questions follow:

- **Reliability:** how often is each shipping option late, and why?
- **Prediction:** given an order at the moment it is placed, how likely is it to be late, and in what range will it arrive?
- **Action:** which orders should trigger an alert, and what should be done?

Success means answers that hold up on future orders (a time-based test), that are honest about uncertainty, and that can
be used by a non-technical reader.

## 3. Data

The project uses one dataset: `DataCoSupplyChainDataset.csv` (all modelling, the scorecard and the decision rules).

**Problems found and how they were handled**

| Problem | What was found | Handling |
|---|---|---|
| Corrupted dates | 71,103 of 180,519 order dates (39%) failed to parse. The file had been re-saved in Excel, which rewrote some month/day dates as day/month | `parse_dates` reads both formats; all rows now parse |
| Leakage columns | `Days for shipping (real)`, `Delivery Status`, `Order Status` and `shipping date` are known only after delivery, and `Late_delivery_risk` equals `Delivery Status == "Late delivery"` | Never used as features; `core.py` stops with an error if one is added, and a test checks it |
| Cancelled orders | 7,754 cancelled or suspected-fraud orders (all payment type TRANSFER) are labelled "on time" although they never shipped | Lateness modelling uses the 172,765 shipped orders. TRANSFER's late rate rises from 0.485 to 0.575 once they are removed |
| Random split leakage | 89% of rows belong to multi-item orders | All splits are chronological |

**Time split.** The oldest 70% of orders trains the models, the next 15% tunes and calibrates them, and the newest 15%
(about June 2017 to January 2018) is the test set.

## 4. Methods

**Features.** Only information available when the order is placed: shipping mode, promised days, payment type, market,
region, customer segment, department, category, order value, quantity, discount, and the hour, weekday and month of the
order. Preprocessing is a scikit-learn pipeline fit on the training split only (median imputation, one-hot encoding with
rare categories grouped).

**Models.** Logistic regression, random forest and LightGBM (small random search on validation ROC-AUC), compared with a
shipping-mode-only baseline and a 5-row lookup table (shipping mode, with Same Day split at the noon order cut-off).

**Metrics.** ROC-AUC, PR-AUC, precision at 80% recall and lift at the top 20% of scores. F1 is not used, because flagging
every order already scores 0.71. PR-AUC, precision-at-recall and lift are tie-aware: a baseline with only a few distinct
scores is not penalised, because ties are broken randomly before ranking.

**Delivery-time window.** A LightGBM point estimate plus an 80% interval from three methods: empirical per-mode quantiles,
LightGBM quantile regression and split conformal prediction per shipping mode. The app uses the conformal windows.

**Calibration and tiers.** Late-delivery scores are calibrated with isotonic regression fit on the validation split, so a
score of 0.8 means about 80% of such orders are late. Calibrated scores map to High (at least 0.70), Medium and Low
(below 0.20) risk tiers.

**Decision rules.** RE-PROMISE (High tier and order value of at least 200), ALERT (High tier), WARN CUSTOMER (80% delivery
window of 3 days or more) and NO ACTION. "Expedite" was rejected as an action because nothing is faster than First Class:
the useful actions are to correct the promise and to tell the customer.

**Explainability.** TreeSHAP contributions from LightGBM, for individual orders and for the model overall.

## 5. Results

### 5.1 Shipping-option reliability

| Shipping mode | Orders | Late rate | Behaviour |
|---|---|---|---|
| Standard Class | 103,153 | 39.8% | Promises 4 days; real delivery is uniform over 2 to 6 days |
| Second Class | 33,806 | 79.8% | Promises 2 days; real delivery takes the same 3.99 days on average as Standard |
| First Class | 26,513 | 100% | Promises 1 day; always takes exactly 2 days |
| Same Day | 9,293 | 47.9% | Before noon it arrives the same day; from noon, the next day |

Profit per order is similar across modes (about 21 to 23 with overlapping confidence intervals), so late delivery is a
service problem here and not a margin problem. The app's promise designer shows the promise each option would need to
reach a chosen late rate: Second Class would need to promise about as long as Standard.

### 5.2 What the model learns

| Model | ROC-AUC | PR-AUC | Precision at 80% recall | Lift, top 20% |
|---|---|---|---|---|
| LightGBM (tuned) | 0.779 | 0.860 | 0.679 | 1.71 |
| 5-row lookup table | 0.777 | 0.860 | 0.677 | 1.70 |
| Random forest, all features | 0.777 | 0.857 | 0.679 | 1.68 |
| Logistic regression, all features | 0.752 | 0.842 | 0.653 | 1.65 |
| Shipping mode only | 0.746 | 0.840 | 0.653 | 1.65 |
| Logistic regression, without mode and promise | 0.522 | 0.591 | 0.582 | 1.05 |
| Random forest, without mode and promise | 0.521 | 0.588 | 0.582 | 1.04 |
| Constant (training late rate) | 0.500 | 0.576 | 0.577 | 0.99 |

The two are almost identical, so the model has learned the promise and the noon cut-off and nothing more. Removing the
order-time features lowers LightGBM to 0.745, which confirms that order hour matters, through the Same Day cut-off. For
Second and Standard Class, delivery time is random, so no model can do better than the lookup table. Without shipping
mode and promised days, every model drops to about 0.52, so the other columns (region, product, payment type) carry almost
no signal. Removing cancelled orders shrinks LightGBM's edge over mode-only from 0.043 to 0.033 (all orders: 0.731 against
0.774; shipped only: 0.746 against 0.779), so part of the old edge was predicting cancellation, not lateness.

### 5.3 Why random splits mislead

| Split | ROC-AUC |
|---|---|
| Time-based (honest) | 0.779 |
| Random | 0.799 |

Items from one order share a shipping mode, order time and delivery time. A random split lets the model recognise siblings
of a test row in the training set. Since 89% of rows are in multi-item orders, the random split overstates performance by
about 0.020.

### 5.4 Early warning and decision rules

| Rule | Orders flagged | Precision | Late orders caught | False alarms per late order caught |
|---|---|---|---|---|
| High-tier alert / re-promise | 37.9% | 0.898 | 59.1% | 0.114 |
| Warn customer (wide window) | 79.4% | 0.497 | 68.4% | 1.01 |
| Any action | 97.5% | 0.591 | 100% | 0.69 |

The high-tier alert is the useful rule. The other 41% of late orders are Standard Class orders that are late about 40% of
the time at random; flagging them would cost many false alarms. The "warn customer" rule is about date uncertainty and not
lateness: it fires on Second and Standard Class, whose delivery time is random, so its precision is not a lateness
measure. The app shows it as a delivery window on every order. Calibrated scores form a few clusters (about 0.00, 0.40, 0.75 to
0.80 and close to 1.00, roughly one per shipping option), which is why the threshold curves step instead of sloping. The
calibration holds on the test split: a predicted 0.39 is observed at 0.39, 0.78 at 0.80 and 0.97 at 0.97.

### 5.5 Delivery-time windows

Delivery time is predicted with LightGBM (MAE 0.973 days, R-squared 0.387), barely better than the mean real days of each
shipping mode (MAE 0.982, R-squared 0.385) and far better than assuming the promise is kept (MAE 1.294, R-squared 0.019).

| 80% window method | Coverage on test | Mean width (days) |
|---|---|---|
| Per-mode quantiles of the training data | 1.000 | 3.23 |
| LightGBM quantile regression | 0.899 | 3.23 |
| Split conformal, per shipping mode | 0.790 | 3.19 |

Only the conformal windows reach the intended 80%; the other two over-cover at the same width. By mode, conformal coverage is
0.800 for Second Class, 0.796 for Standard Class, 0.839 for Same Day and 0.731 for First Class. First Class always takes
exactly 2 days, so its window is almost zero wide and small prediction drift puts a few orders just outside it; in practice
the right First Class message is simply "2 days". Because delivery times are whole days and about uniform from 2 to 6 within
Second and Standard Class, the windows there are 4 days wide. The width is a property of the data and not of the method.

### 5.6 Regions, markets and departments

Raw late rates by region mislead, because a region that uses more Second Class looks unreliable only through the short
promise. The analysis therefore compares each region's late rate with the rate expected from its shipping-mode mix and
reports the gap with a z-score (notebook 1, section 5, and the "Where and when" page of the app). Result: 3 of 21 regions
are unusual (|z| above 1.96) against about 1 expected by chance, and the largest gap is 2.4 percentage points (Central
Africa), so regions differ very little once the shipping-mode mix is taken into account.

### 5.7 How sure are we? Deeper checks

Notebook 4 asks whether "the table matches the model" is solid or just one lucky test period (test set: 25,915 orders).

| Check | Method | Result |
|---|---|---|
| Is the gap noise? | Bootstrap, resampling whole orders (500 draws), paired on the same resample | LightGBM 0.7786 [0.7696, 0.7877], table 0.7766 [0.7695, 0.7844]; difference +0.0020 [-0.0028, +0.0066], includes 0 |
| Does it depend on the cut? | Rolling-origin validation, 5 expanding windows, settings fixed | Mean difference +0.0003, standard deviation 0.0050; model ahead in 2 of 5 folds (range -0.0040 to +0.0077) |
| Does acting on the model pay more? | Decision curve (net benefit, Vickers and Elkin 2006) | Differ by at most 0.0032 per order at any of 19 thresholds (0.05 to 0.95), and by under 0.0005 at the six thresholds printed in the notebook (0.3024 against 0.3024 at t = 0.5; 0.2498 against 0.2503 at t = 0.7) |
| Anything beyond the promise? | AUC inside one shipping option | Second Class 0.505, Standard Class 0.505 (table 0.5); Same Day 1.0 for both through the noon cut-off |
| Equal for all groups? | Calibration error and AUC by market and region (groups of 300+ orders) | Calibration error 0.005 to 0.021 everywhere except South America (592 orders): 0.057, with model AUC 0.685 below the table's 0.713 |
| Stable in operation? | Population stability index (PSI), train against test | Market 3.19 and Department 1.14 (large shifts), Product Price 0.10 and Quantity 0.15 (moderate); model score 0.057 (stable); quarterly late rate 0.56 to 0.59 |

Reading: the model and the table are statistically and practically indistinguishable, so the 5-row table is the better
choice for use (simpler, auditable, no retraining). The South America result is the same overfit to a small group that the
app guards against by falling back to the table. The PSI result shows that input drift is not performance drift: the
newest orders come almost entirely from two markets (Europe and Pacific Asia), a very different mix from the training
period, yet the score distribution and the late rate did not move, because lateness depends only on the shipping option.
The monitoring rule therefore watches the score PSI, the shipping-option mix and the observed late rate, not every input.
This also means the test period represents mostly two markets, a limit on how far the held-out numbers generalise.

## 6. The app

Five pages: Overview; Shipping options (real arrival days, promise designer, profit); Check an order (live scoring,
comparison of every shipping option for one order, SHAP drivers); Early warnings (threshold tuning, user-set cost of
mistakes, tiers, calibration, priority watchlist); Where and when (region, market and department fairness, weekday-hour
heat map, monthly trends).

## 7. Lessons

- **Check for a simple baseline first.** A 5-row table was as good as the model; without it, the model would have looked
  like a success.
- **Tied scores bias ranking metrics.** PR-AUC, precision at recall and lift were initially computed without tie-breaking,
  which understated the mode-only baseline (precision at 80% recall 0.576, against 0.653 once ties were broken) and
  inflated the apparent gain from LightGBM.
- **Choose thresholds after looking at the scores.** The first tier thresholds (0.75 and 0.45) did not match the clustered
  score distribution and were changed to 0.70 and 0.20.
- **Look at the labels.** Cancelled orders labelled "on time", a target that is a copy of a post-delivery column, and dates
  silently corrupted by Excel would each have produced wrong conclusions.

## 8. Limitations and future work

- DataCo is synthetic: lateness comes from the promise and the Same Day cut-off plus random noise. Accuracy is capped near
  0.78 ROC-AUC, and about 41% of late orders cannot be predicted.
- The newest 15% of orders (the test period) come almost entirely from Europe and Pacific Asia, so the held-out numbers
  say little about other markets. The rolling-origin check, which spans the whole timeline, gives the same answer.
- There are no supplier IDs and no shipping-cost column, so supplier ranking and cost optimisation are out of scope.
- The cost estimate in the app depends on assumptions the user supplies; it is a reasoning aid, not a forecast.
- Future work: test the approach on real carrier data with in-transit events (an early warning is strongest once there is
  something to observe after the order ships), split by order ID to remove the multi-item effect from random-split
  comparisons, and add anomaly detection.

## 9. Reproducibility

Seed 42 (`core.py`); all splits are chronological. Run notebooks 1, 2, 3 and 4 in order, then `streamlit run app.py`. Unit
tests: `python -m pytest -q`.

---

## Appendix A. Experiment log

Every run that changed a decision or a number, in order. Test-set numbers are on the newest 15% of shipped orders.

| # | Notebook | What was done | Result | Decision |
|---|---|---|---|---|
| 1 | 1 | Loaded DataCo (latin-1) and parsed dates | 71,103 of 180,519 order dates failed to parse (39%) | The CSV had been re-saved in Excel. `parse_dates` reads both formats; all rows now parse |
| 2 | 1 | Checked the target against `Delivery Status` | `Late_delivery_risk` equals `Delivery Status == "Late delivery"` | The four post-delivery columns are never features; `core.py` enforces it |
| 3 | 1 | Looked at cancelled / suspected-fraud orders | 7,754 orders, all TRANSFER, labelled "on time" | Shipped orders only (172,765). TRANSFER late rate 0.485 to 0.575 |
| 4 | 2 | Six classifiers and baselines on a time split | LightGBM 0.779, 5-row lookup 0.777, random forest 0.777, logistic regression 0.752, mode only 0.746; without mode and promise about 0.52 | F1 not used (flagging everything gives 0.71); reported ROC-AUC, PR-AUC, precision at 80% recall, lift at top 20% |
| 5 | 2 | Found ranking metrics biased against tied scores | Mode-only precision at 80% recall: 0.576 before tie-breaking, 0.653 after | Added random tie-breaking (`break_ties`). The earlier "+0.10 gain from LightGBM" was inflated; discarded |
| 6 | 2 | LightGBM on all orders vs shipped only | All orders: mode only 0.731, LightGBM 0.774. Shipped only: 0.746 and 0.779 | Cancelled orders are noise for "late given it ships"; shipped only |
| 7 | 2 | Removed the order-time features | ROC-AUC 0.745 without, 0.779 with | Order hour carries real signal: the Same Day noon cut-off |
| 8 | 2 | ETA point model plus three 80% interval methods | Coverage: conformal 0.790, quantile regression 0.899, per-mode quantiles 1.000 | Width is a property of the data (delivery time is random within a mode); the app uses conformal windows |
| 9 | 2, 3 | 5-row lookup table vs LightGBM | ROC-AUC 0.777 vs 0.779 | The model found nothing beyond the promise and the noon cut-off; about 0.78 is the ceiling |
| 10 | 2 | Random split vs time split | 0.799 vs 0.779 | 89% of rows are in multi-item orders; report the time split |
| 11 | 3 | Calibrated scores (isotonic, validation only) and looked at tiers | Scores cluster near 0.00, 0.40, 0.75 to 0.80 and 1.00 | Tier thresholds first 0.75 / 0.45, changed to 0.70 / 0.20 after seeing the clusters |
| 12 | 1 | Shipping-mode scorecard with Wilson intervals | Standard 39.8%, Second 79.8%, First Class 100%, Same Day 47.9% late | Second Class takes the same 3.99 days as Standard but promises 2; profit per order similar across modes |
| 13 | 3 | Renamed the rule action EXPEDITE to RE-PROMISE | | Nothing is faster than First Class; the useful actions are to correct the promise and to tell the customer |
| 14 | 3 | Evaluated the rules on the test split | High-tier alert: 37.9% flagged, precision 0.898, recall 0.591, 0.114 false alarms per late order caught | The High-tier alert is the useful rule; the wide-window warning is a delivery-window message |
| 15 | 4 | Bootstrap (500 draws, whole orders) on LightGBM against the 5-row table | Difference +0.0020, 95% interval -0.0028 to +0.0066 | Gap is not distinguishable from zero; the table is preferred for use |
| 16 | 4 | Rolling-origin validation, 5 expanding windows | Mean difference +0.0003, standard deviation 0.0050; model ahead in 2 of 5 folds | Conclusion does not depend on the split point |
| 17 | 4 | Decision curve, thresholds 0.05 to 0.95 | Net benefit differs by at most 0.0032 per order at any threshold | No decision is changed by using the model |
| 18 | 4 | AUC inside each shipping option; calibration and AUC by market and region | Second and Standard Class 0.505; South America (592 orders) calibration error 0.057, model AUC 0.685 against table 0.713 | Nothing beyond the promise; keep the fall-back to the table for small groups |
| 19 | 4 | PSI, train against test | Market 3.19, Department 1.14, score 0.057, quarterly late rate 0.56 to 0.59 | Input drift is not performance drift; monitor score PSI and late rate, and note the test period is mostly two markets |

Mistakes caught along the way: tie handling in the metrics (row 5), found by asking why a baseline looked worse than a
lookup table; a test seed collision in an early synthetic check; and thresholds and rule names chosen before looking at
the data (rows 11 and 13), revised once the data was seen.

## Appendix B. Who did what

Replace `[Name 1]` and `[Name 2]` with your names and edit any row so it matches who actually did the work. Both of us can
explain every part of the project; the split shows who led each piece.

| Workstream | Led by | Where |
|---|---|---|
| Problem framing, dataset choice, README and REPORT, GitHub repository | [Name 1] | `README.md`, `REPORT.md` |
| Loading, date repair, cleaning, features, time split | [Name 1] | `core.py` sections 2 and 3; notebook 1, sections 1 to 3 |
| Scorecard and region fairness | [Name 1] | `core.py` section 7; notebook 1, sections 4 and 5 |
| Metrics (tie-aware) and baselines | [Name 2] | `core.py` section 4; notebook 2, section 1 |
| LightGBM late-delivery model and delivery-time windows | [Name 1] | `core.py` section 5; notebook 2, sections 2 to 4 |
| Calibration, risk tiers, lookup-table comparison | [Name 2] | `core.py` section 6; notebook 3, sections 1 to 3 |
| SHAP explanations | [Name 1] | notebook 3, section 4; `core.py` section 8 |
| Decision rules and their evaluation | [Name 2] | `core.py` section 6; notebook 3, section 5 |
| Streamlit app and deployment | [Name 2] | `app.py`, `core.py` sections 8 and 9 |
| Tests | [Name 1] and [Name 2] | `test_core.py` |

Shared: choosing the datasets and the final direction, reviewing each other's work and checking the numbers in the report,
and preparing for the chalk talk.
