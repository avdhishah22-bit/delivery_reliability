"""Cleaning steps. Cleaning removes junk columns and duplicates only; it never
uses the target, so it is safe to run before splitting."""
import pandas as pd

# Personal data / free text / ID columns with no predictive value.
DATACO_DROP = [
    "Customer Email", "Customer Password", "Customer Fname", "Customer Lname",
    "Customer Street", "Customer Zipcode", "Order Zipcode",
    "Product Image", "Product Description", "Product Status",
]


def clean_dataco(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop(columns=[c for c in DATACO_DROP if c in df.columns])
    out = out.drop_duplicates()
    out.columns = [c.strip() for c in out.columns]
    return out.reset_index(drop=True)


def clean_logistics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop_duplicates().copy()
    num = out.select_dtypes("number").columns
    # Time series: carry the last known value forward, then fill any leading gaps.
    out[num] = out[num].ffill().bfill()
    return out.reset_index(drop=True)
