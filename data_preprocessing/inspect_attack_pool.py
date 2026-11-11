
"""Inspect the balanced attack pool without modifying it."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

POOL_PATH = ROOT / "data" / "attack_pool" / "balanced_attack_pool.csv"
SCALER_PATH = ROOT / "benign" / "train_scaler.npz"
REPORT_PATH = ROOT / "outputs" / "attack_pool_inspection.csv"

EXPECTED_TYPES = 33
EXPECTED_PER_TYPE = 800
EXPECTED_FEATURES = 39


def main():
    if not POOL_PATH.exists():
        raise FileNotFoundError(f"Pool not found: {POOL_PATH}")

    df = pd.read_csv(POOL_PATH)

    if "attack_type" not in df.columns:
        raise ValueError("Missing attack_type column.")

    feature_cols = [
        col for col in df.columns
        if col != "attack_type"
    ]

    if len(feature_cols) != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected 39 features, found {len(feature_cols)}."
        )

    counts = df["attack_type"].value_counts().sort_index()

    missing_values = int(df[feature_cols].isna().sum().sum())

    nonfinite_values = int(
        (~np.isfinite(df[feature_cols].to_numpy())).sum()
    )

    within_type_duplicates = int(
        df.duplicated(
            subset=["attack_type", *feature_cols]
        ).sum()
    )

    feature_duplicate_extra_rows = int(
        df.duplicated(subset=feature_cols).sum()
    )

    # Check benign scaler feature names and order.
    scaler_match = False

    if SCALER_PATH.exists():
        with np.load(SCALER_PATH, allow_pickle=False) as scaler:
            benign_features = [
                str(x) for x in scaler["feature_names"].tolist()
            ]
            scaler_match = feature_cols == benign_features

    report_rows = []

    for attack_type, count in counts.items():
        group = df[df["attack_type"] == attack_type]

        duplicates = int(
            group.duplicated(subset=feature_cols).sum()
        )

        report_rows.append({
            "attack_type": attack_type,
            "pool_rows": int(count),
            "within_type_duplicates": duplicates,
            "status": (
                "OK"
                if count == EXPECTED_PER_TYPE and duplicates == 0
                else "CHECK"
            ),
        })

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(report_rows).to_csv(
        REPORT_PATH,
        index=False,
    )

    print("\n=== BALANCED ATTACK POOL INSPECTION ===")
    print(f"Total rows: {len(df):,}")
    print(f"Attack types: {len(counts)}")
    print(f"Feature count: {len(feature_cols)}")
    print(f"Missing values: {missing_values}")
    print(f"Non-finite values: {nonfinite_values}")
    print(f"Within-type duplicates: {within_type_duplicates}")
    print(
        "Feature-identical extra rows across pool: "
        f"{feature_duplicate_extra_rows}"
    )
    print(f"Benign scaler feature order match: {scaler_match}")

    valid = (
        len(df) == EXPECTED_TYPES * EXPECTED_PER_TYPE
        and len(counts) == EXPECTED_TYPES
        and counts.eq(EXPECTED_PER_TYPE).all()
        and missing_values == 0
        and nonfinite_values == 0
        and within_type_duplicates == 0
        and scaler_match
    )

    print(f"Overall validation: {'PASS' if valid else 'CHECK'}")
    print(f"Report: {REPORT_PATH}")

    if not valid:
        raise SystemExit(
            "Pool validation needs attention. "
            "Review the results before sampling."
        )


if __name__ == "__main__":
    main()
