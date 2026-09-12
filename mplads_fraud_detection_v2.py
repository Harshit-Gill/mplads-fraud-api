"""
SIH26102 - MPLADS Fraud & Anomaly Detection (v2 - defensible thresholds)
Final pipeline: Feature Engineering + Rule-Based Flags + Isolation Forest

WHAT CHANGED FROM v1 AND WHY:
    v1's rule thresholds (e.g. sanctioned_amount between 2.0M-2.6M, duration<=20 days,
    contamination=0.116) were reverse-engineered by looking at labeled anomaly rows.
    That's circular on synthetic data and impossible to replicate on real MoSPI data,
    where ground truth doesn't exist. A judge who checks this will flag it immediately.

    v2 fixes this in three ways:
    1. Rule thresholds are now derived from domain logic + unsupervised statistics
       (percentiles, regulatory limits, round-number math) computed WITHOUT looking
       at the is_anomaly column at all.
    2. Isolation Forest contamination is set to a fixed, defensible prior
       ("auto", or a stated assumption like "we assume ~5% base irregularity rate
       per CAG audit literature") instead of the exact ground-truth fraction.
    3. All validation is done on a held-out test split that had zero influence on
       threshold selection or model fitting. The labels are used ONLY at the very
       end, to report how well an honestly-built detector performs -- not to build
       the detector itself.

    This is slightly less "impressive" on paper (recall will likely drop from the
    inflated ~80% to something more modest) but it is real, defensible, and will
    survive a judge asking "how did you pick that number?"

Architecture (unchanged):
    - Rule-based flags for archetypes with clean, explainable signatures
      (fund_splitting, fiscal_year_end_dumping, cost_overrun, ghost_fast_completion,
      round_number_invoice)
    - Isolation Forest for the subtler, multivariate archetypes
      (vendor_collusion, price_inflation)
    - Final flag = ANY rule fired OR Isolation Forest flags it as anomalous
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# ---------------------------------------------------------------------------
# MPLADS scheme facts used to justify thresholds (cite these on your slide)
# ---------------------------------------------------------------------------
# - MPLADS annual entitlement per MP is Rs 5 crore (Rs 2.5 crore per installment
#   in recent years). Splitting a single work into pieces just under a sanction/
#   approval limit is a documented gaming pattern in audit reports -- so the
#   THRESHOLD should be "near a known approval limit", not a number tuned to
#   catch labeled rows.
# - Financial year ends March 31. Fiscal-year-end fund dumping is a well
#   documented public-finance phenomenon (rushing to show utilization before
#   year close), independent of this dataset.
ANNUAL_ENTITLEMENT_INR = 25_000_000       # Rs 2.5 crore typical installment
SPLIT_LIMIT_BAND = (0.75, 1.00)            # fraction of installment: "just under" band
FISCAL_YEAR_END_MONTHS = [2, 3]            # Jan-Mar -- pre year-close months
GHOST_COMPLETION_PERCENTILE = 0.01         # bottom 1% of ALL completion durations
                                            # (i.e. "implausibly fast" relative to the
                                            # whole population, not tuned to labels)


def load_and_engineer_features(csv_path: str) -> pd.DataFrame:
    """Load raw MPLADS dataset and engineer all derived features."""
    df = pd.read_csv(csv_path)

    df["sanction_date"] = pd.to_datetime(df["sanction_date"], errors="coerce")
    df["completion_date"] = pd.to_datetime(df["completion_date"], errors="coerce")

    df["utilization_ratio"] = df["expenditure_inr"] / df["sanctioned_amount_inr"]

    df["is_completed"] = df["completion_date"].notna().astype(int)
    df["duration_days"] = (df["completion_date"] - df["sanction_date"]).dt.days
    median_duration = df.loc[df["is_completed"] == 1, "duration_days"].median()
    df["duration_days_filled"] = df["duration_days"].fillna(median_duration)

    df["sanction_month"] = df["sanction_date"].dt.month
    df["is_round_expenditure"] = (df["expenditure_inr"] % 10000 == 0).astype(int)

    vendor_cat_mean = df.groupby(["vendor_name", "work_category"])["expenditure_inr"].transform("mean")
    df["vendor_cat_deviation"] = ((df["expenditure_inr"] - vendor_cat_mean) / vendor_cat_mean).fillna(0)
    df["vendor_mp_work_count"] = df.groupby(["vendor_name", "mp_name"])["work_id"].transform("count")
    df["vendor_distinct_mp_count"] = df.groupby("vendor_name")["mp_name"].transform("nunique")
    df["vendor_total_expenditure"] = df.groupby("vendor_name")["expenditure_inr"].transform("sum")

    for col in ["state", "district", "work_category", "implementing_agency", "vendor_name", "mp_party"]:
        freq = df[col].value_counts(normalize=True)
        df[col + "_freq"] = df[col].map(freq)

    return df


def apply_rule_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply rule-based flags using thresholds derived from domain facts and
    unsupervised statistics of the data -- NEVER from the is_anomaly column.
    """
    # Fund splitting: repeated vendor-MP pairing AND sanctioned amount sitting
    # just under the known annual installment ceiling (a real gaming pattern,
    # not a label-fitted band).
    lower = SPLIT_LIMIT_BAND[0] * ANNUAL_ENTITLEMENT_INR
    upper = SPLIT_LIMIT_BAND[1] * ANNUAL_ENTITLEMENT_INR
    df["flag_fund_splitting"] = (
        (df["vendor_mp_work_count"] >= 3)
        & (df["sanctioned_amount_inr"].between(lower, upper))
    ).astype(int)

    # Fiscal year-end dumping: sanctioned in Jan-Mar with near-zero utilization
    # so far. Threshold (5%) is a round, explainable cutoff for "essentially
    # unspent", not tuned on labels.
    df["flag_fiscal_dumping"] = (
        (df["utilization_ratio"] < 0.05) & (df["sanction_month"].isin(FISCAL_YEAR_END_MONTHS))
    ).astype(int)

    # Cost overrun: spending exceeded what was sanctioned. This threshold (1.0)
    # is definitional, not tuned.
    df["flag_cost_overrun"] = (df["utilization_ratio"] > 1.0).astype(int)

    # Ghost/implausibly-fast completion: derived from the data's OWN duration
    # distribution (bottom 1st percentile of completed works), not a fixed
    # "20 days" picked to match labels.
    completed = df.loc[df["is_completed"] == 1, "duration_days"]
    fast_cutoff = completed.quantile(GHOST_COMPLETION_PERCENTILE)
    df["flag_ghost_fast"] = (
        (df["is_completed"] == 1) & (df["duration_days"] <= fast_cutoff)
    ).astype(int)

    # Round-number invoicing: definitional (exact multiple of 10,000), not tuned.
    df["flag_round_invoice"] = df["is_round_expenditure"]

    rule_cols = [
        "flag_fund_splitting", "flag_fiscal_dumping", "flag_cost_overrun",
        "flag_ghost_fast", "flag_round_invoice",
    ]
    df["any_rule_fired"] = (df[rule_cols].sum(axis=1) > 0).astype(int)
    return df


def fit_isolation_forest(train_df: pd.DataFrame, full_df: pd.DataFrame,
                          contamination="auto") -> pd.DataFrame:
    """
    Fit Isolation Forest on the TRAIN split only, then score the full dataset.
    contamination defaults to 'auto' (sklearn's own estimate) rather than the
    exact ground-truth fraction -- state your assumption explicitly on the
    slide if you override this (e.g. "we assume ~5-8% irregularity rate based
    on CAG audit sampling reports").
    """
    if_feature_cols = [
        "vendor_cat_deviation", "vendor_distinct_mp_count", "vendor_total_expenditure",
        "utilization_ratio", "duration_days_filled",
        "state_freq", "district_freq", "work_category_freq",
        "implementing_agency_freq", "vendor_name_freq", "mp_party_freq",
    ]

    X_train = train_df[if_feature_cols].copy()
    X_full = full_df[if_feature_cols].copy()
    assert X_train.isna().sum().sum() == 0, "NaNs found in Isolation Forest training features"

    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1)
    iso.fit(X_train)

    full_df = full_df.copy()
    full_df["iso_pred"] = iso.predict(X_full)              # -1 = anomaly, 1 = normal
    full_df["iso_score"] = iso.decision_function(X_full)    # lower = more anomalous
    return full_df


def combine_and_score(df: pd.DataFrame) -> pd.DataFrame:
    """Combine rules + Isolation Forest into final binary prediction
    and a 0-100 composite risk score."""
    df["final_pred"] = ((df["any_rule_fired"] == 1) | (df["iso_pred"] == -1)).astype(int)

    rule_severity = {
        "flag_cost_overrun": 90,
        "flag_fiscal_dumping": 85,
        "flag_fund_splitting": 85,
        "flag_ghost_fast": 80,
        "flag_round_invoice": 60,
    }
    df["rule_based_score"] = 0
    for flag_col, severity in rule_severity.items():
        df["rule_based_score"] = np.where(df[flag_col] == 1,
                                           np.maximum(df["rule_based_score"], severity),
                                           df["rule_based_score"])

    iso_min, iso_max = df["iso_score"].min(), df["iso_score"].max()
    df["iso_based_score"] = 100 * (iso_max - df["iso_score"]) / (iso_max - iso_min)

    df["composite_risk_score"] = np.maximum(df["rule_based_score"], df["iso_based_score"]).round(1)
    return df


def validate_on_holdout(df: pd.DataFrame, test_idx) -> None:
    """
    Print validation report using ONLY the held-out test rows. Rules and the
    Isolation Forest were never fit using these rows' labels (rules use no
    labels at all; IF was fit on the train split only) -- so this number is
    an honest, non-circular estimate of real-world performance.
    """
    test_df = df.loc[test_idx]
    y_true = test_df["is_anomaly"]

    print("=== HOLD-OUT TEST SET performance (labels never touched during fitting) ===")
    print(classification_report(y_true, test_df["final_pred"], target_names=["normal", "anomaly"]))

    print("=== Recall by archetype (hold-out only) ===")
    for atype in test_df["anomaly_type"].unique():
        if atype == "none":
            continue
        subset = test_df[test_df["anomaly_type"] == atype]
        if len(subset) == 0:
            continue
        caught = (subset["final_pred"] == 1).sum()
        print(f"{atype}: {caught}/{len(subset)} ({caught/len(subset):.0%})")


if __name__ == "__main__":
    df = load_and_engineer_features("mplads_dataset_v2.csv")

    # Split BEFORE any fitting. Rules don't need fitting (they're fixed logic),
    # but Isolation Forest is fit on train only, and reported numbers come
    # exclusively from test.
    train_df, test_df = train_test_split(
        df, test_size=0.25, random_state=42, stratify=df["is_anomaly"]
    )

    df = apply_rule_flags(df)                       # label-free, safe on full df
    train_df = apply_rule_flags(train_df)            # kept in sync for IF feature use

    df = fit_isolation_forest(train_df=train_df, full_df=df, contamination="auto")
    df = combine_and_score(df)

    validate_on_holdout(df, test_idx=test_df.index)

    output_cols = [
        "work_id", "state", "district", "mp_name", "vendor_name",
        "sanctioned_amount_inr", "expenditure_inr", "utilization_ratio",
        "flag_fund_splitting", "flag_fiscal_dumping", "flag_cost_overrun",
        "flag_ghost_fast", "flag_round_invoice", "iso_pred",
        "composite_risk_score", "final_pred",
    ]
    df[output_cols].to_csv("mplads_scored_output.csv", index=False)
    print("\nSaved scored output to mplads_scored_output.csv")
