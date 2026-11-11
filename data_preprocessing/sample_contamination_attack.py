
"""
Sample attack data for 5% and 10% contamination.

Input:
    data/attack_pool/balanced_attack_pool.csv

Outputs:
    data/attack_pool/attack_5pct_raw.csv
    data/attack_pool/attack_10pct_raw.csv
    outputs/attack_contamination_sampling_report.csv

This script only samples attack rows.
It does not scale features or combine benign data.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

POOL_PATH = ROOT / "data" / "attack_pool" / "balanced_attack_pool.csv"

OUTPUT_5 = ROOT / "data" / "attack_pool" / "attack_5pct_raw.csv"
OUTPUT_10 = ROOT / "data" / "attack_pool" / "attack_10pct_raw.csv"

REPORT_PATH = (
    ROOT / "outputs" / "attack_contamination_sampling_report.csv"
)

SEED = 42

BENIGN_COUNT = 200_000
ATTACK_COUNT_5 = 10_526
ATTACK_COUNT_10 = 22_222


def make_quotas(total, attack_types):
    """Distribute the requested total evenly across attack types."""

    base, remainder = divmod(total, len(attack_types))

    return {
        attack_type: base + (i < remainder)
        for i, attack_type in enumerate(attack_types)
    }


def main():
    if not POOL_PATH.exists():
        raise FileNotFoundError(
            f"Balanced attack pool not found: {POOL_PATH}"
        )

    for path in [OUTPUT_5, OUTPUT_10, REPORT_PATH]:
        if path.exists():
            raise FileExistsError(
                f"Output already exists: {path}\n"
                "Back up or remove the old output before rerunning."
            )

    pool = pd.read_csv(POOL_PATH)

    if "attack_type" not in pool.columns:
        raise ValueError("Missing attack_type column.")

    attack_types = sorted(pool["attack_type"].unique())

    if len(attack_types) != 33:
        raise ValueError(
            f"Expected 33 attack types, found {len(attack_types)}."
        )

    quota_5 = make_quotas(ATTACK_COUNT_5, attack_types)
    quota_10 = make_quotas(ATTACK_COUNT_10, attack_types)

    rng = np.random.default_rng(SEED)

    sampled_5 = []
    sampled_10 = []
    report_rows = []

    for attack_type in attack_types:
        group = pool[pool["attack_type"] == attack_type]

        n5 = quota_5[attack_type]
        n10 = quota_10[attack_type]

        if len(group) < n10:
            raise ValueError(
                f"{attack_type}: need {n10} rows, "
                f"but only {len(group)} are available."
            )

        # Shuffle the 800 pool rows for this attack type.
        shuffled_indices = rng.permutation(group.index.to_numpy())

        # The 5% selection is contained in the 10% selection.
        indices_5 = shuffled_indices[:n5]
        indices_10 = shuffled_indices[:n10]

        sampled_5.extend(indices_5.tolist())
        sampled_10.extend(indices_10.tolist())

        report_rows.append({
            "attack_type": attack_type,
            "pool_count": len(group),
            "sampled_5pct": n5,
            "sampled_10pct": n10,
        })

        print(
            f"{attack_type}: "
            f"5%={n5}, 10%={n10}"
        )

    # Confirm exact sample counts.
    if len(sampled_5) != ATTACK_COUNT_5:
        raise ValueError("Incorrect 5% attack count.")

    if len(sampled_10) != ATTACK_COUNT_10:
        raise ValueError("Incorrect 10% attack count.")

    # Confirm the 5% selection is a subset of the 10% selection.
    if not set(sampled_5).issubset(set(sampled_10)):
        raise ValueError(
            "5% attack samples are not a subset of 10% samples."
        )

    # Randomize output row order.
    sampled_5 = rng.permutation(sampled_5)
    sampled_10 = rng.permutation(sampled_10)

    attack_5 = pool.loc[sampled_5].reset_index(drop=True)
    attack_10 = pool.loc[sampled_10].reset_index(drop=True)

    OUTPUT_5.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    attack_5.to_csv(OUTPUT_5, index=False)
    attack_10.to_csv(OUTPUT_10, index=False)

    pd.DataFrame(report_rows).to_csv(
        REPORT_PATH,
        index=False,
    )

    actual_ratio_5 = (
        len(attack_5) / (BENIGN_COUNT + len(attack_5))
    )
    actual_ratio_10 = (
        len(attack_10) / (BENIGN_COUNT + len(attack_10))
    )

    print("\n=== ATTACK SAMPLING COMPLETED ===")
    print(f"Attack types: {len(attack_types)}")
    print(f"5% attack samples: {len(attack_5):,}")
    print(f"10% attack samples: {len(attack_10):,}")
    print(f"5% actual contamination ratio: {actual_ratio_5:.6%}")
    print(f"10% actual contamination ratio: {actual_ratio_10:.6%}")
    print("5% attack set is a subset of 10% attack set: YES")
    print(f"5% output: {OUTPUT_5}")
    print(f"10% output: {OUTPUT_10}")
    print(f"Report: {REPORT_PATH}")
    print("NOTE: Attack features are not scaled yet.")


if __name__ == "__main__":
    main()
