"""Order-level early warning on DataCo (decision A: DataCo is the only predictive dataset).

  - take the late-delivery score from the best classifier (notebook 04)
  - calibrate it on the VALIDATION split (isotonic) so a score means a real late rate
  - map calibrated P(late) to tiers High / Medium / Low with thresholds from config.yaml

Also holds the 5-row lookup table that notebook 05 compares the ML model against.
Always use time_split(); never a random split.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_calibrator(val_scores, val_y) -> IsotonicRegression:
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
        np.asarray(val_scores, float), np.asarray(val_y, float))


def assign_tiers(p, high: float, low: float):
    p = np.asarray(p, float)
    return np.where(p >= high, "High", np.where(p < low, "Low", "Medium"))


def rule_key(df):
    """Key of the lookup table: shipping mode, with Same Day split at the noon order cutoff."""
    mode, hour = df["Shipping Mode"].to_numpy(), df["order_hour"].to_numpy()
    return np.where((mode == "Same Day") & (hour >= 12), "Same Day (from noon)",
                    np.where(mode == "Same Day", "Same Day (before noon)", mode))
