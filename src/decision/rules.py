"""Task 8: rule-based decision support on top of model outputs.
Thresholds come from config.yaml (section 'rules')."""


def recommend(late_prob: float, order_value: float, risk_level: str,
              eta_interval_days: float, rules: dict) -> list:
    actions = []
    if late_prob >= rules["late_prob_expedite"] and order_value >= rules["high_value_order"]:
        actions.append("EXPEDITE: switch to a more reliable shipping mode")
    if risk_level == rules["risk_alert_level"]:
        actions.append("ALERT: high-risk shipment, consider rerouting")
    if eta_interval_days >= rules["eta_interval_wide_days"]:
        actions.append("WARN CUSTOMER: delivery date is uncertain")
    return actions or ["NO ACTION"]
