"""Build models/app_artifacts.joblib for the Streamlit app.
Run once after notebooks 04-06 (needs models/late_delivery_lgbm.joblib and models/eta_lgbm.joblib):

    python -m src.models.export_artifacts
"""
import sys
from pathlib import Path

import joblib
import yaml

from src.app_data import build_aggregates
from src.data.clean import clean_dataco
from src.data.load import load_dataco
from src.features.build_features import (
    DATACO_CAT, DATACO_DATE_COL, DATACO_NUM, DATACO_TARGET_CLF, DATACO_TARGET_REG,
    build_dataco_features, shipped_only, time_split)
from src.models.train_risk import fit_calibrator, rule_key

PROMISE, MODE, DEPT, CAT = "Days for shipment (scheduled)", "Shipping Mode", "Department Name", "Category Name"


def main(config: str = "config.yaml"):
    cfg = yaml.safe_load(Path(config).read_text())
    mdir = Path(cfg["paths"]["models"]); Y, YR = DATACO_TARGET_CLF, DATACO_TARGET_REG
    data = shipped_only(build_dataco_features(clean_dataco(load_dataco(cfg["paths"]["dataco"]))))
    train, val, test = time_split(data, cfg["split"]["train"], cfg["split"]["val"])

    bundle = joblib.load(mdir / "late_delivery_lgbm.joblib")
    model, feats = bundle["model"], bundle["features"]
    eta = joblib.load(mdir / "eta_lgbm.joblib"); hw = eta["conformal_halfwidths"]

    cal = fit_calibrator(model.predict_proba(val[feats])[:, 1], val[Y])               # validation only
    lookup = train.assign(key=rule_key(train)).groupby("key")[Y].mean().to_dict()      # the 5-row table
    mode_rate_train = train.groupby(MODE)[Y].mean()

    scored = test.copy()
    scored["p_late"] = cal.predict(model.predict_proba(scored[feats])[:, 1])
    scored["option"] = rule_key(scored)
    scored["p_lookup"] = scored["option"].map(lookup)
    scored["p_mode"] = scored[MODE].map(mode_rate_train)
    scored["interval_days"] = [2 * hw.get(m, hw["__all__"]) for m in scored[MODE]]
    keep = [DATACO_DATE_COL, MODE, "option", "Type", "Market", "Order Region", DEPT, "Sales", "order_hour",
            "order_weekday", YR, Y, "p_late", "p_lookup", "p_mode", "interval_days"]
    scored = scored[keep].reset_index(drop=True)

    split_info = {n: {"orders": len(d), "start": d[DATACO_DATE_COL].min(), "end": d[DATACO_DATE_COL].max()}
                  for n, d in (("Train", train), ("Validation", val), ("Test", test))}
    agg = build_aggregates(data, scored, y=Y, yr=YR, promise=PROMISE, date_col=DATACO_DATE_COL,
                           option_fn=rule_key, split_info=split_info)

    art = {
        "features": feats, "calibrator": cal, "lookup": lookup, "scored": scored, **agg,
        "options": {c: sorted(data[c].dropna().unique().tolist()) for c in DATACO_CAT},
        "category_by_dept": {d: sorted(g[CAT].dropna().unique().tolist()) for d, g in data.groupby(DEPT)},
        "num_defaults": data[DATACO_NUM].median().to_dict(),
        "scheduled_days": {m: int(s.mode()[0]) for m, s in data.groupby(MODE)[PROMISE]},
        "same_country_default": int(data["same_country"].mode()[0]),
        "conformal_halfwidths": hw,
    }
    joblib.dump(art, mdir / "app_artifacts.joblib")
    print("saved", mdir / "app_artifacts.joblib", "| test orders:", len(scored))


if __name__ == "__main__":
    main(*sys.argv[1:])
