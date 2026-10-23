"""Dataset loaders."""
import pandas as pd

DATACO_DATE_COL = "order date (DateOrders)"
DATACO_SHIP_DATE_COL = "shipping date (DateOrders)"
DATACO_TARGET = "Late_delivery_risk"


def parse_dataco_dates(s: pd.Series) -> pd.Series:
    """Parse DataCo dates, which come in two formats if the CSV was ever saved by Excel.

    - '1/31/2018 22:56'  : original month/day/year (Excel could not misread these).
    - '01-12-2018 23:51' : Excel re-read the original 1/12/2018 (12 Jan) as day/month
                           and rewrote it day-first. The FIRST number is really the month.
    """
    s = s.astype(str).str.strip()
    original = pd.to_datetime(s, format="%m/%d/%Y %H:%M", errors="coerce")
    swapped = pd.to_datetime(s, format="%m-%d-%Y %H:%M", errors="coerce")
    return original.fillna(swapped)


def load_dataco(path: str) -> pd.DataFrame:
    """DataCo is latin-1 encoded. Rows are sorted by order date."""
    df = pd.read_csv(path, encoding="latin-1")
    df[DATACO_DATE_COL] = parse_dataco_dates(df[DATACO_DATE_COL])
    if DATACO_SHIP_DATE_COL in df.columns:
        df[DATACO_SHIP_DATE_COL] = parse_dataco_dates(df[DATACO_SHIP_DATE_COL])

    n_bad = int(df[DATACO_DATE_COL].isna().sum())
    if n_bad:
        print(f"WARNING: {n_bad} order dates could not be parsed and will be dropped.")
    df = df.dropna(subset=[DATACO_DATE_COL, DATACO_TARGET])
    return df.sort_values(DATACO_DATE_COL).reset_index(drop=True)


def load_logistics(path: str) -> pd.DataFrame:
    """Time-series logistics file, sorted by timestamp."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)
