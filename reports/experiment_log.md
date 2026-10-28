# Experiment log

Every run that changed a decision or a number, in order. Test-set numbers are on the newest 15% of orders (shipped
orders only, time-based split) unless stated. Full result tables are in `reports/tables/`
(`python -m src.utils.tables_to_markdown` turns them into `reports/results_tables.md`).

## Data and setup

| # | Notebook | What I did | Result | Decision |
|---|---|---|---|---|
| 1 | 01 | Loaded DataCo (latin-1) and parsed dates | 71,103 of 180,519 order dates failed to parse (39%) | The CSV had been re-saved in Excel, which rewrote some month/day dates as day/month. Wrote `parse_dataco_dates` to read both formats; all 180,519 rows now parse. |
| 2 | 01 | Checked the target against `Delivery Status` | `Late_delivery_risk` equals `Delivery Status == "Late delivery"` | `Delivery Status`, `Order Status`, `Days for shipping (real)` and `shipping date` are leakage and are never features (unit test enforces it). |
| 3 | 01, 03 | Looked at cancelled / suspected-fraud orders | 7,754 orders, all payment type TRANSFER, labelled "on time" | Evaluate on shipped orders only (172,765). TRANSFER late rate rises from 0.485 to 0.574 once cancelled orders are removed, which explains its apparent reliability. |
| 4 | 02 | Audited the two logistics files | Dynamic file: features unrelated to targets (max abs correlation 0.014), `risk_classification` is a binned copy of `disruption_likelihood_score`. Smart file: `Logistics_Delay` is exactly "Delayed OR Heavy traffic"; ROC-AUC 0.488 without those columns | Decision A: DataCo is the only predictive dataset. The logistics files are kept as leakage / negative-result case studies. |

## Modelling

| # | Notebook | Change | Result | Takeaway |
|---|---|---|---|---|
| 5 | 03 | Six classifiers and ETA baselines on a time split | Tables in `reports/tables/` (baselines) | Flagging every order gives F1 0.71, so F1 is not used. Reported metrics: ROC-AUC, PR-AUC, precision at 80% recall, lift at top 20%. |
| 6 | 03 | Found that PR-AUC, precision@recall and lift are biased against models with tied scores (the mode-only baseline) | Mode-only precision at 80% recall: 0.576 before tie-breaking, 0.653 after | Added random tie-breaking in `src/models/evaluate.py` (`break_ties`). The earlier "+0.10 gain from LightGBM" was inflated; discard it. ROC-AUC is unaffected. |
| 7 | 04 | LightGBM (random search, 8 draws, validation ROC-AUC) on all orders vs shipped orders | See the comparison table in notebook 04 | Cancelled orders are labelled "on time" without ever shipping, so they are noise for "late given it ships"; evaluation uses shipped orders only. |
| 8 | 04 | Removed the order-time features | Test ROC-AUC 0.743 without them, 0.779 with | Order hour carries real signal: the Same Day noon cut-off. |
| 9 | 04 | ETA point model plus three 80% interval methods (empirical per-mode quantiles, LightGBM quantile regression, split conformal per mode) | Coverage about 0.80 (0.797) for the 80% interval | If the three methods give similar widths, the width is a property of the data (delivery time is random within a mode), not of the model. First Class should have width 0 because it always takes 2 days; check this in the notebook output. |
| 10 | 05 | 5-row lookup table (shipping mode, Same Day split at noon) vs LightGBM | ROC-AUC 0.777 vs 0.779 | The model has found nothing beyond the promise and the noon cut-off. About 0.78 is the ceiling on this data. |
| 11 | 05 | Random split vs time split, same model | ROC-AUC 0.801 vs 0.779 (+0.022) | 89% of rows belong to multi-item orders, so a random split puts related rows on both sides. Report the time split. |
| 12 | 05 | Calibrated scores with isotonic regression (fit on validation only) and looked at the tiers | Scores cluster near 0.00, 0.38 to 0.40, 0.80 and 1.00 | Tier thresholds first set at 0.75 / 0.45 before seeing the clusters; changed to 0.70 / 0.20 (`config.yaml`). |

## Decision rules and findings

| # | Notebook | Change | Result | Takeaway |
|---|---|---|---|---|
| 13 | 06 | Shipping-mode scorecard with Wilson intervals | Standard 103,153 orders, 39.8% late; Second 33,806, 79.8%; First Class 26,513, 100%; Same Day 9,293, 47.9% | Second Class takes the same 3.99 days as Standard but promises 2, so it is late twice as often. First Class is always 2 days against a 1-day promise. Profit per order is similar across modes (about 21 to 23, overlapping intervals). |
| 14 | 06 | Renamed the rule action EXPEDITE to RE-PROMISE | | Nothing is faster than First Class, so the only useful actions are to correct the promise and to tell the customer. |
| 15 | 06 | Evaluated the rules at the new thresholds | High-tier alert: flags 37.4% of orders, precision 0.898, recall 0.584, 0.113 false alarms per late order caught. Warn customer: 79.4% of orders, precision 0.497. Any action: 97.5% of orders | The high-tier alert is the useful rule. The "wide interval" warning fires on almost every Second and Standard order, so the app shows it as a delivery window rather than an alarm. About 42% of late orders (Standard Class, late 40% of the time at random) cannot be predicted. |

## Mistakes caught along the way

- Tie handling in the metrics (row 6): a metric bug, found by checking why the baseline looked worse than a lookup table.
- Test seed collision: labels and noise used the same seed in a synthetic check, giving a false average precision of
  0.346. Fixed by using an independent seed; the corrected check returns the base rate, as it should.
- Thresholds and rule names chosen before looking at the data (rows 12 and 14) were revised once the data was seen.

## Still to record

- The exact baseline and ETA tables from notebooks 03 and 04 (copy from `reports/tables/`).
- Any new run: add a row with what changed, the metric before and after, and the takeaway.
