"""Feature definitions, builders and sklearn preprocessing.
Leakage columns are listed here in one place."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------- DataCo
DATACO_DATE_COL = "order date (DateOrders)"
DATACO_TARGET_CLF = "Late_delivery_risk"          # late-delivery classification
DATACO_TARGET_REG = "Days for shipping (real)"    # ETA regression (a target, never a feature)

# Known only AFTER the order ships/arrives -> never use as features.
LEAKY_DATACO = [
    "Days for shipping (real)",
    "Delivery Status",
    "Order Status",
    "shipping date (DateOrders)",
]

# Cancelled / suspected-fraud orders never ship. DataCo labels them Late_delivery_risk = 0,
# which mixes "cancelled" into "on time". Payment type TRANSFER is the only type that has them.
CANCELLED_STATUS = "Shipping canceled"


def shipped_only(df: pd.DataFrame) -> pd.DataFrame:
    """Drop cancelled orders. Use for evaluation of 'late given it ships' and for ETA."""
    return df[df["Delivery Status"] != CANCELLED_STATUS].reset_index(drop=True)


DATACO_CAT = [
    "Type", "Shipping Mode", "Market", "Order Region", "Customer Segment",
    "Department Name", "Category Name",
]
DATACO_NUM = [
    "Days for shipment (scheduled)", "Order Item Discount Rate",
    "Order Item Quantity", "Product Price", "Sales",
    "order_hour", "order_weekday", "order_month",
    "same_country", "discount_ratio",
]


def build_dataco_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["order_hour"] = out[DATACO_DATE_COL].dt.hour
    out["order_weekday"] = out[DATACO_DATE_COL].dt.weekday
    out["order_month"] = out[DATACO_DATE_COL].dt.month
    out["same_country"] = (out["Customer Country"] == out["Order Country"]).astype(int)
    price = out["Order Item Product Price"].replace(0, np.nan)
    out["discount_ratio"] = out["Order Item Discount"] / price
    return out


# ------------------------------------------------------------- Logistics
LOGISTICS_TARGET_REG = "delivery_time_deviation"
LOGISTICS_TARGET_CLF = "risk_classification"

# Probably derived from the other columns -> run with and without (leakage experiment).
LOGISTICS_DERIVED = ["delay_probability", "disruption_likelihood_score"]
LOGISTICS_TARGETS = ["delivery_time_deviation", "risk_classification", "delay_probability"]


# --------------------------------------------------------------- Shared
def time_split(df: pd.DataFrame, train: float = 0.70, val: float = 0.15):
    """Chronological split. Input must already be sorted by time."""
    n = len(df)
    i, j = int(n * train), int(n * (train + val))
    return df.iloc[:i], df.iloc[i:j], df.iloc[j:]


def make_preprocessor(cat, num, scale: bool = False) -> ColumnTransformer:
    """Imputer + (optional) scaler for numeric, one-hot for categorical.
    Put this inside a Pipeline so it is fit on the training split only."""
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    transformers = [("num", Pipeline(num_steps), num)]
    if cat:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), cat))
    return ColumnTransformer(transformers)
