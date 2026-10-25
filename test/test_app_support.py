from datetime import datetime

from src.app_support import lookup_key, make_order_frame
from src.features.build_features import DATACO_CAT, DATACO_NUM, LEAKY_DATACO

FEATS = DATACO_CAT + DATACO_NUM
ART = {"features": FEATS, "scheduled_days": {"Standard Class": 4, "Same Day": 0}, "same_country_default": 1}


def test_order_frame_matches_model_features_and_has_no_leaky_column():
    f = make_order_frame(ART, mode="Standard Class", order_dt=datetime(2017, 6, 20, 14, 5), type_="DEBIT", market="LATAM",
                         region="Western Europe", department="Golf", category="Cleats", segment="Consumer",
                         sales=120.0, price=60.0, quantity=2, discount_rate=0.1)
    assert list(f.columns) == FEATS and not set(f.columns) & set(LEAKY_DATACO)
    assert f["order_hour"].iloc[0] == 14 and f["Days for shipment (scheduled)"].iloc[0] == 4


def test_same_day_noon_cutoff():
    assert lookup_key("Same Day", 11) == "Same Day (before noon)"
    assert lookup_key("Same Day", 12) == "Same Day (from noon)"
    assert lookup_key("First Class", 3) == "First Class"
