"""
MPLADS Fraud Detection API (SIH26102)

Scores a single new work entry using the same rule flags + Isolation Forest
model as the offline v2 pipeline. Because several features are vendor/MP
history-dependent (vendor_cat_deviation, vendor_mp_work_count,
vendor_distinct_mp_count, vendor_total_expenditure) and frequency-encoded
(state/district/category/agency/vendor/party), this API looks those up from
tables computed once at training time (train_and_export.py) rather than
recomputing them from scratch on every request.

COLD START: if a vendor, MP, state, etc. was never seen in training data,
we fall back to a category-level or global average / the rarest observed
frequency. This is an honest limitation -- flag it in your presentation.
"""

from datetime import date, datetime
from typing import Optional

import joblib
import json
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="MPLADS Fraud Detection API", version="2.0")

# CORS: allow the dashboard (any origin) to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load artifacts once at startup ---
iso_model = joblib.load("iso_model.joblib")
with open("artifacts.json") as f:
    A = json.load(f)

RULE_SEVERITY = {
    "flag_cost_overrun": 90,
    "flag_fiscal_dumping": 85,
    "flag_fund_splitting": 85,
    "flag_ghost_fast": 80,
    "flag_round_invoice": 60,
}


class WorkEntry(BaseModel):
    state: str
    district: str
    mp_name: str
    mp_party: str
    work_category: str
    implementing_agency: str
    vendor_name: str
    sanction_date: date
    sanctioned_amount_inr: float = Field(gt=0)
    expenditure_inr: float = Field(ge=0)
    status: str  # "Completed" | "Ongoing" | "Delayed" | "Sanctioned - Not Started"
    completion_date: Optional[date] = None


def freq_lookup(col: str, value: str) -> float:
    return A["freq_maps"][col].get(value, A["freq_fallback"][col])


def build_features(entry: WorkEntry) -> dict:
    utilization_ratio = entry.expenditure_inr / entry.sanctioned_amount_inr
    is_completed = 1 if entry.completion_date is not None else 0

    if entry.completion_date is not None:
        duration_days = (entry.completion_date - entry.sanction_date).days
        duration_days_filled = duration_days
    else:
        duration_days = None
        duration_days_filled = A["median_duration"]

    sanction_month = entry.sanction_date.month
    is_round_expenditure = 1 if (entry.expenditure_inr % 10000 == 0) else 0

    # --- Vendor-level lookups (cold-start fallback if unseen) ---
    vc_key = f"{entry.vendor_name}||{entry.work_category}"
    if vc_key in A["vendor_cat_mean_map"]:
        vendor_cat_mean = A["vendor_cat_mean_map"][vc_key]
    elif entry.work_category in A["category_mean_fallback"]:
        vendor_cat_mean = A["category_mean_fallback"][entry.work_category]
    else:
        vendor_cat_mean = A["overall_mean_expenditure"]
    vendor_cat_deviation = (entry.expenditure_inr - vendor_cat_mean) / vendor_cat_mean if vendor_cat_mean else 0.0

    vm_key = f"{entry.vendor_name}||{entry.mp_name}"
    # +1 to represent this new work itself being added to that vendor-MP pairing
    vendor_mp_work_count = A["vendor_mp_work_count_map"].get(vm_key, 0) + 1

    vendor_distinct_mp_count = A["vendor_distinct_mp_count_map"].get(entry.vendor_name, 0)
    # if this MP hasn't worked with this vendor before, this entry adds a new distinct MP
    if vm_key not in A["vendor_mp_work_count_map"]:
        vendor_distinct_mp_count += 1

    vendor_total_expenditure = A["vendor_total_expenditure_map"].get(entry.vendor_name, 0.0) + entry.expenditure_inr

    features = {
        "utilization_ratio": utilization_ratio,
        "is_completed": is_completed,
        "duration_days": duration_days,
        "duration_days_filled": duration_days_filled,
        "sanction_month": sanction_month,
        "is_round_expenditure": is_round_expenditure,
        "vendor_cat_deviation": vendor_cat_deviation,
        "vendor_mp_work_count": vendor_mp_work_count,
        "vendor_distinct_mp_count": vendor_distinct_mp_count,
        "vendor_total_expenditure": vendor_total_expenditure,
        "state_freq": freq_lookup("state", entry.state),
        "district_freq": freq_lookup("district", entry.district),
        "work_category_freq": freq_lookup("work_category", entry.work_category),
        "implementing_agency_freq": freq_lookup("implementing_agency", entry.implementing_agency),
        "vendor_name_freq": freq_lookup("vendor_name", entry.vendor_name),
        "mp_party_freq": freq_lookup("mp_party", entry.mp_party),
    }
    return features


def score_entry(entry: WorkEntry) -> dict:
    f = build_features(entry)

    lower = A["split_limit_band"][0] * A["annual_entitlement_inr"]
    upper = A["split_limit_band"][1] * A["annual_entitlement_inr"]

    flag_fund_splitting = int(
        f["vendor_mp_work_count"] >= 3
        and lower <= entry.sanctioned_amount_inr <= upper
    )
    flag_fiscal_dumping = int(
        f["utilization_ratio"] < 0.05 and f["sanction_month"] in A["fiscal_year_end_months"]
    )
    flag_cost_overrun = int(f["utilization_ratio"] > 1.0)
    flag_ghost_fast = int(
        f["is_completed"] == 1 and f["duration_days"] is not None and f["duration_days"] <= A["ghost_fast_cutoff"]
    )
    flag_round_invoice = f["is_round_expenditure"]

    flags = {
        "flag_fund_splitting": flag_fund_splitting,
        "flag_fiscal_dumping": flag_fiscal_dumping,
        "flag_cost_overrun": flag_cost_overrun,
        "flag_ghost_fast": flag_ghost_fast,
        "flag_round_invoice": flag_round_invoice,
    }
    any_rule_fired = int(any(flags.values()))

    X = np.array([[f[col] for col in A["if_feature_cols"]]])
    iso_pred = int(iso_model.predict(X)[0])       # -1 anomaly, 1 normal
    iso_score = float(iso_model.decision_function(X)[0])

    final_pred = int(any_rule_fired == 1 or iso_pred == -1)

    rule_based_score = 0
    for flag_name, severity in RULE_SEVERITY.items():
        if flags[flag_name] == 1:
            rule_based_score = max(rule_based_score, severity)

    iso_min, iso_max = A["iso_score_min"], A["iso_score_max"]
    # clip in case the new score falls outside the training range
    iso_score_clipped = min(max(iso_score, iso_min), iso_max)
    iso_based_score = 100 * (iso_max - iso_score_clipped) / (iso_max - iso_min)

    composite_risk_score = round(max(rule_based_score, iso_based_score), 1)

    return {
        "flags": flags,
        "any_rule_fired": bool(any_rule_fired),
        "iso_pred": "anomaly" if iso_pred == -1 else "normal",
        "iso_score": round(iso_score, 4),
        "final_pred": bool(final_pred),
        "composite_risk_score": composite_risk_score,
        "utilization_ratio": round(f["utilization_ratio"], 4),
        "cold_start_vendor": entry.vendor_name not in A["vendor_total_expenditure_map"],
    }


@app.get("/")
def root():
    return {"status": "ok", "message": "MPLADS Fraud Detection API is running"}


@app.post("/predict")
def predict(entry: WorkEntry):
    return score_entry(entry)


@app.get("/known-values")
def known_values():
    """Lets the dashboard populate dropdowns with values the model has actually seen."""
    return {
        "states": sorted(A["freq_maps"]["state"].keys()),
        "work_categories": sorted(A["freq_maps"]["work_category"].keys()),
        "implementing_agencies": sorted(A["freq_maps"]["implementing_agency"].keys()),
        "mp_parties": sorted(A["freq_maps"]["mp_party"].keys()),
    }
