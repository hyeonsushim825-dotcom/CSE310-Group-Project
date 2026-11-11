
"""
Scale sampled attack datasets using the benign training scaler.

Inputs:
    benign/train_scaler.npz
    data/attack_pool/attack_5pct_raw.csv
    data/attack_pool/attack_10pct_raw.csv

Outputs:
    data/attack_pool/attack_5pct_scaled.npy
    data/attack_pool/attack_10pct_scaled.npy
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SCALER_PATH = ROOT / "benign" / "train_scaler.npz"
ATTACK_DIR = ROOT / "data" / "attack_pool"

DATASETS = {
    "5%": (
        ATTACK_DIR / "attack_5pct_raw.csv",
        ATTACK_DIR / "attack_5pct_scaled.npy",
        10_526,
    ),
    "10%": (
        ATTACK_DIR / "attack_10pct_raw.csv",
        ATTACK_DIR / "attack_10pct_scaled.npy",
        22_222,
    ),
}


def main():
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found: {SCALER_PATH}"
        )

    with np.load(SCALER_PATH) as scaler:
        mean = scaler["mean"]
        scale = scaler["scale"]
        feature_names = scaler["feature_names"].tolist()

    if len(feature_names) != 39:
        raise ValueError("Expected 39 scaler features.")

    if not np.all(np.isfinite(mean)):
        raise ValueError("Scaler mean contains invalid values.")

    if not np.all(np.isfinite(scale)) or np.any(scale == 0):
        raise ValueError("Scaler scale contains invalid values.")

    print("=== ATTACK SCALING ===")
    print(f"Scaler: {SCALER_PATH}")
    print(f"Feature count: {len(feature_names)}")

    for label, (input_path, output_path, expected_rows) in DATASETS.items():
        if not input_path.exists():
            raise FileNotFoundError(
                f"Attack dataset not found: {input_path}"
            )

        if output_path.exists():
            raise FileExistsError(
                f"Output already exists: {output_path}"
            )

        df = pd.read_csv(input_path)

        if "attack_type" not in df.columns:
            raise ValueError(
                f"{label}: Missing attack_type column."
            )

        actual_features = [
            col for col in df.columns
            if col != "attack_type"
        ]

        if actual_features != feature_names:
            raise ValueError(
                f"{label}: Feature names or order do not "
                "match the benign scaler."
            )

        if len(df) != expected_rows:
            raise ValueError(
                f"{label}: Expected {expected_rows} rows, "
                f"found {len(df)}."
            )

        # Preserve the exact benign scaler feature order.
        X_raw = df[feature_names].to_numpy(
            dtype=np.float64
        )

        if not np.all(np.isfinite(X_raw)):
            raise ValueError(
                f"{label}: Raw data contains NaN or Infinity."
            )

        # Apply the scaler fitted on benign training data.
        X_scaled = (X_raw - mean) / scale
        X_scaled = X_scaled.astype(np.float32)

        if not np.all(np.isfinite(X_scaled)):
            raise ValueError(
                f"{label}: Scaled data contains invalid values."
            )

        np.save(output_path, X_scaled)

        print(f"\n{label} contamination:")
        print(f"  Input shape: {X_raw.shape}")
        print(f"  Output shape: {X_scaled.shape}")
        print(f"  Output dtype: {X_scaled.dtype}")
        print(f"  Saved: {output_path}")

    print("\nATTACK SCALING COMPLETED")
    print("NOTE: Benign and Attack have not been combined yet.")


if __name__ == "__main__":
    main()
