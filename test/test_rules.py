from src.decision.rules import recommend, risk_tier
from src.models.train_risk import assign_tiers

RULES = {"tier_high_prob": 0.75, "tier_low_prob": 0.45, "risk_alert_level": "High",
         "high_value_order": 200.0, "eta_interval_wide_days": 3.0}


def test_tiers():
    assert risk_tier(0.9, RULES) == "High" and risk_tier(0.6, RULES) == "Medium" and risk_tier(0.1, RULES) == "Low"
    assert list(assign_tiers([0.9, 0.6, 0.1], 0.75, 0.45)) == ["High", "Medium", "Low"]


def test_recommend():
    assert recommend(0.9, 500, 1.0, RULES)[0].startswith("RE-PROMISE")
    assert recommend(0.9, 50, 1.0, RULES)[0].startswith("ALERT")
    assert recommend(0.1, 500, 4.0, RULES) == ["WARN CUSTOMER: delivery date is uncertain"]
    assert recommend(0.1, 50, 1.0, RULES) == ["NO ACTION"]
