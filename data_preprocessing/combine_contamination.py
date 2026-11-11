
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BENIGN_PATH = (
    PROJECT_ROOT / "benign" / "benign_train_200k_scaled.npy"
)

ATTACK_DIR = PROJECT_ROOT / "data" / "attack_pool"

ATTACK_FILES = {
    "5pct": ATTACK_DIR / "attack_5pct_scaled.npy",
    "10pct": ATTACK_DIR / "attack_10pct_scaled.npy",
}

SEED = 42


def combine_and_save(benign, attack_path, scenario):
    attack = np.load(attack_path)

    if attack.ndim != 2 or attack.shape[1] != 39:
        raise ValueError(f"Unexpected attack shape: {attack.shape}")

    if not np.isfinite(attack).all():
        raise ValueError(f"{scenario}: Attack contains NaN or Inf")

    # Combine feature arrays (no labels included)
    combined = np.concatenate([benign, attack], axis=0)

    # Labels are for analysis only, NOT model training.
    # 0 = Benign, 1 = Attack
    labels = np.concatenate([
        np.zeros(len(benign), dtype=np.uint8),
        np.ones(len(attack), dtype=np.uint8),
    ])

    # Shuffle features and labels using the same permutation.
    rng = np.random.default_rng(SEED)
    permutation = rng.permutation(len(combined))

    combined = combined[permutation].astype(np.float32)
    labels = labels[permutation]

    # Validate results
    expected_total = len(benign) + len(attack)

    if combined.shape != (expected_total, 39):
        raise ValueError(f"Unexpected combined shape: {combined.shape}")

    if not np.isfinite(combined).all():
        raise ValueError(f"{scenario}: Combined data contains NaN or Inf")

    if int(labels.sum()) != len(attack):
        raise ValueError(f"{scenario}: Label count mismatch")

    # Save feature data and labels separately.
    feature_path = ATTACK_DIR / f"train_{scenario}_contaminated.npy"
    label_path = ATTACK_DIR / f"train_{scenario}_labels.npy"

    np.save(feature_path, combined)
    np.save(label_path, labels)

    attack_ratio = len(attack) / expected_total * 100

    print(f"\n=== {scenario} ===")
    print(f"Benign samples: {len(benign):,}")
    print(f"Attack samples: {len(attack):,}")
    print(f"Combined shape: {combined.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Attack ratio: {attack_ratio:.6f}%")
    print(f"Features saved: {feature_path}")
    print(f"Labels saved: {label_path}")


def main():
    benign = np.load(BENIGN_PATH)

    if benign.shape != (200_000, 39):
        raise ValueError(f"Unexpected benign shape: {benign.shape}")

    if not np.isfinite(benign).all():
        raise ValueError("Benign contains NaN or Inf")

    for scenario, attack_path in ATTACK_FILES.items():
        combine_and_save(benign, attack_path, scenario)

    print("\nAll contamination datasets created successfully.")


if __name__ == "__main__":
    main()
