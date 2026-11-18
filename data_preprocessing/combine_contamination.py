"""Combine fixed-size benign subsets with sampled attack data.

Creates two 200,000-row training datasets:
    5% contamination: 190,000 benign + 10,000 attack
    10% contamination: 180,000 benign + 20,000 attack

The benign subset used for 10% is contained in the benign subset used for 5%.
Labels are saved separately for analysis and are not included in model features.
"""

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

TOTAL_COUNT = 200_000
FEATURE_COUNT = 39
SEED = 42


def combine_and_save(benign_subset, attack_path, scenario, expected_attack, shuffle_seed):
    attack = np.load(attack_path)

    if attack.shape != (expected_attack, FEATURE_COUNT):
        raise ValueError(
            f"{scenario}: expected attack shape "
            f"({expected_attack}, {FEATURE_COUNT}), found {attack.shape}"
        )

    if not np.isfinite(attack).all():
        raise ValueError(f"{scenario}: Attack contains NaN or Inf")

    if benign_subset.shape != (TOTAL_COUNT - expected_attack, FEATURE_COUNT):
        raise ValueError(
            f"{scenario}: unexpected benign subset shape {benign_subset.shape}"
        )

    # Combine feature arrays. Labels are kept separately for analysis only.
    combined = np.concatenate([benign_subset, attack], axis=0)
    labels = np.concatenate([
        np.zeros(len(benign_subset), dtype=np.uint8),
        np.ones(len(attack), dtype=np.uint8),
    ])

    # Shuffle features and labels with the same reproducible permutation.
    rng = np.random.default_rng(shuffle_seed)
    permutation = rng.permutation(len(combined))
    combined = combined[permutation].astype(np.float32, copy=False)
    labels = labels[permutation]

    if combined.shape != (TOTAL_COUNT, FEATURE_COUNT):
        raise ValueError(f"{scenario}: unexpected combined shape {combined.shape}")

    if not np.isfinite(combined).all():
        raise ValueError(f"{scenario}: Combined data contains NaN or Inf")

    if int(labels.sum()) != expected_attack:
        raise ValueError(f"{scenario}: label count mismatch")

    feature_path = ATTACK_DIR / f"train_{scenario}_contaminated.npy"
    label_path = ATTACK_DIR / f"train_{scenario}_labels.npy"

    np.save(feature_path, combined)
    np.save(label_path, labels)

    attack_ratio = len(attack) / TOTAL_COUNT * 100

    print(f"\n=== {scenario} ===")
    print(f"Benign samples: {len(benign_subset):,}")
    print(f"Attack samples: {len(attack):,}")
    print(f"Combined shape: {combined.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Attack ratio: {attack_ratio:.2f}%")
    print(f"Features saved: {feature_path}")
    print(f"Labels saved: {label_path}")


def main():
    benign = np.load(BENIGN_PATH)

    if benign.shape != (TOTAL_COUNT, FEATURE_COUNT):
        raise ValueError(
            f"Expected benign shape ({TOTAL_COUNT}, {FEATURE_COUNT}), "
            f"found {benign.shape}"
        )

    if not np.isfinite(benign).all():
        raise ValueError("Benign contains NaN or Inf")

    for path in ATTACK_FILES.values():
        if not path.exists():
            raise FileNotFoundError(f"Attack file not found: {path}")

    # One fixed random ordering makes the 180k benign subset a subset
    # of the 190k benign subset, while keeping both selections reproducible.
    rng = np.random.default_rng(SEED)
    benign_indices = rng.permutation(len(benign))

    benign_5 = benign[benign_indices[:190_000]]
    benign_10 = benign[benign_indices[:180_000]]

    combine_and_save(
        benign_5,
        ATTACK_FILES["5pct"],
        "5pct",
        expected_attack=10_000,
        shuffle_seed=42,
    )

    combine_and_save(
        benign_10,
        ATTACK_FILES["10pct"],
        "10pct",
        expected_attack=20_000,
        shuffle_seed=42,
    )

    print("\nBoth contaminated datasets contain exactly 200,000 samples.")
    print("The 10% benign subset is contained in the 5% benign subset.")


if __name__ == "__main__":
    main()
