"""Task 8: rule-based decision support on top of model outputs.
Thresholds come from config.yaml (section 'rules').

In DataCo, lateness is set mostly by the PROMISED shipping time (a First Class order is always late),
so a late-prone order cannot be fixed by "expediting". The useful actions are to correct the promise
and to tell the customer."""


def risk_tier(late_prob: float, rules: dict) -> str:
    if late_prob >= rules["tier_high_prob"]:
        return "High"
    return "Low" if late_prob < rules["tier_low_prob"] else "Medium"


def recommend(late_prob: float, order_value: float, eta_interval_days: float, rules: dict) -> list:
    """late_prob must be CALIBRATED (see src/models/train_risk.py)."""
    actions = []
    tier = risk_tier(late_prob, rules)
    if tier == rules["risk_alert_level"] and order_value >= rules["high_value_order"]:
        actions.append("RE-PROMISE: very likely late; update the promised date or offer a shipping option with a realistic promise")
    elif tier == rules["risk_alert_level"]:
        actions.append("ALERT: high chance of late delivery")
    if eta_interval_days >= rules["eta_interval_wide_days"]:
        actions.append("WARN CUSTOMER: delivery date is uncertain")
    return actions or ["NO ACTION"]
