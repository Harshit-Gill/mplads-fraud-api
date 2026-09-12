"""
Trains the final MPLADS fraud detection model on the FULL dataset and exports
everything the API needs to score a brand-new, single entry:

  - the fitted IsolationForest
  - frequency-encoding lookup tables (state/district/category/agency/vendor/party)
  - vendor-level aggregate lookup tables (for vendor_cat_deviation, vendor_mp_work_count,
    vendor_distinct_mp_count, vendor_total_expenditure)
  - fallback values for anything never seen before (a brand-new vendor, etc.)
  - fixed thresholds/constants used by the rule flags
  - iso_score min/max used to rescale into the 0-100 composite score

NOTE: this trains on ALL 8,658 rows for the best possible production model.
The recall/precision numbers you report on your slide should come from
mplads_fraud_detection_v2.py's held-out evaluation, NOT from this script --
this script's job is just to produce the artifacts the live API uses.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from mplads_fraud_detection_v2 import (
    load_and_engineer_features,
    apply_rule_flags,
    combine_and_score,
    ANNUAL_ENTITLEMENT_INR,
    SPLIT_LIMIT_BAND,
    FISCAL_YEAR_END_MONTHS,
    GHOST_COMPLETION_PERCENTILE,
)

# Chosen contamination: NOT fit to match the exact ground-truth anomaly fraction
# (that would be circular). 0.07 is a stated assumption -- audit sampling
# literature on public works schemes typically finds single-digit-percentage
# irregularity rates. State this explicitly on your limitations slide.
CONTAMINATION = 0.07

FREQ_COLS = ["state", "district", "work_category", "implementing_agency", "vendor_name", "mp_party"]
IF_FEATURE_COLS = [
    "vendor_cat_deviation", "vendor_distinct_mp_count", "vendor_total_expenditure",
    "utilization_ratio", "duration_days_filled",
    "state_freq", "district_freq", "work_category_freq",
    "implementing_agency_freq", "vendor_name_freq", "mp_party_freq",
]


def main():
    df = load_and_engineer_features("mplads_dataset_v2.csv")
    df = apply_rule_flags(df)

    # --- Fit IsolationForest on the full dataset for production ---
    X = df[IF_FEATURE_COLS].copy()
    iso = IsolationForest(n_estimators=200, contamination=CONTAMINATION, random_state=42, n_jobs=-1)
    iso.fit(X)
    df["iso_pred"] = iso.predict(X)
    df["iso_score"] = iso.decision_function(X)
    df = combine_and_score(df)

    # --- Frequency lookup tables (with fallback = rarest observed frequency) ---
    freq_maps = {}
    freq_fallback = {}
    for col in FREQ_COLS:
        vc = df[col].value_counts(normalize=True)
        freq_maps[col] = vc.to_dict()
        freq_fallback[col] = float(vc.min())  # unseen category treated as rare

    # --- Vendor-level aggregate lookups ---
    vendor_cat_mean = (
        df.groupby(["vendor_name", "work_category"])["expenditure_inr"].mean()
    )
    vendor_cat_mean_map = {f"{v}||{c}": float(m) for (v, c), m in vendor_cat_mean.items()}

    category_mean_fallback = df.groupby("work_category")["expenditure_inr"].mean().to_dict()
    category_mean_fallback = {k: float(v) for k, v in category_mean_fallback.items()}
    overall_mean_expenditure = float(df["expenditure_inr"].mean())

    vendor_mp_work_count = df.groupby(["vendor_name", "mp_name"])["work_id"].count()
    vendor_mp_work_count_map = {f"{v}||{m}": int(c) for (v, m), c in vendor_mp_work_count.items()}

    vendor_distinct_mp_count_map = df.groupby("vendor_name")["mp_name"].nunique().to_dict()
    vendor_distinct_mp_count_map = {k: int(v) for k, v in vendor_distinct_mp_count_map.items()}

    vendor_total_expenditure_map = df.groupby("vendor_name")["expenditure_inr"].sum().to_dict()
    vendor_total_expenditure_map = {k: float(v) for k, v in vendor_total_expenditure_map.items()}

    # --- Fixed constants needed at inference time ---
    median_duration = float(df.loc[df["is_completed"] == 1, "duration_days"].median())
    completed = df.loc[df["is_completed"] == 1, "duration_days"]
    ghost_fast_cutoff = float(completed.quantile(GHOST_COMPLETION_PERCENTILE))
    iso_score_min = float(df["iso_score"].min())
    iso_score_max = float(df["iso_score"].max())

    artifacts = {
        "freq_maps": freq_maps,
        "freq_fallback": freq_fallback,
        "vendor_cat_mean_map": vendor_cat_mean_map,
        "category_mean_fallback": category_mean_fallback,
        "overall_mean_expenditure": overall_mean_expenditure,
        "vendor_mp_work_count_map": vendor_mp_work_count_map,
        "vendor_distinct_mp_count_map": vendor_distinct_mp_count_map,
        "vendor_total_expenditure_map": vendor_total_expenditure_map,
        "median_duration": median_duration,
        "ghost_fast_cutoff": ghost_fast_cutoff,
        "iso_score_min": iso_score_min,
        "iso_score_max": iso_score_max,
        "annual_entitlement_inr": ANNUAL_ENTITLEMENT_INR,
        "split_limit_band": SPLIT_LIMIT_BAND,
        "fiscal_year_end_months": FISCAL_YEAR_END_MONTHS,
        "if_feature_cols": IF_FEATURE_COLS,
        "freq_cols": FREQ_COLS,
        "contamination_used": CONTAMINATION,
    }

    joblib.dump(iso, "iso_model.joblib")
    with open("artifacts.json", "w") as f:
        json.dump(artifacts, f)

    print("Saved iso_model.joblib and artifacts.json")
    print(f"Trained on {len(df)} rows, contamination={CONTAMINATION}")
    print(f"Unique vendors: {df['vendor_name'].nunique()}, unique MPs: {df['mp_name'].nunique()}")


if __name__ == "__main__":
    main()
