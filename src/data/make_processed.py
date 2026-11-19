"""Optional: save the cleaned, shipped-only DataCo table to data/processed/ (compressed CSV).

    python -m src.data.make_processed
"""
from pathlib import Path

import yaml

from src.data.clean import clean_dataco
from src.data.load import load_dataco
from src.features.build_features import build_dataco_features, shipped_only


def main(config: str = "config.yaml"):
    cfg = yaml.safe_load(Path(config).read_text())
    df = shipped_only(build_dataco_features(clean_dataco(load_dataco(cfg["paths"]["dataco"]))))
    out = Path(cfg["paths"]["processed"]) / "dataco_shipped.csv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, compression="gzip")
    print(f"saved {len(df):,} shipped orders to {out}")


if __name__ == "__main__":
    main()
