import numpy as np
import pandas as pd

from src.app_data import promise_for_target, promise_late_rate, threshold_metrics, threshold_sweep


def test_promise_math():
    row = pd.Series([0, 0, 0.2, 0.2, 0.2, 0.2, 0.2], index=range(7))   # Second/Standard-like: 2..6 days uniform
    assert abs(promise_late_rate(row, 2) - 0.8) < 1e-9
    assert abs(promise_late_rate(row, 4) - 0.4) < 1e-9
    assert promise_for_target(row, 0.4) == 4
    assert promise_for_target(row, 0.0) == 6


def test_threshold_metrics_and_sweep():
    df = pd.DataFrame({"p_late": [0.9, 0.9, 0.5, 0.1], "y": [1, 0, 1, 0]})
    m = threshold_metrics(df, 0.8, "y")
    assert m["flagged"] == 2 and m["precision"] == 0.5 and m["recall"] == 0.5 and m["false_per_catch"] == 1.0
    sweep = threshold_sweep(df, "y")
    assert sweep["recall"].is_monotonic_decreasing and len(sweep) == 91
    empty = threshold_metrics(df, 0.99, "y")
    assert empty["flagged"] == 0 and np.isnan(empty["precision"])
