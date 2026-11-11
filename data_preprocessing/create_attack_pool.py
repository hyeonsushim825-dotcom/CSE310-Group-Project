
"""Create a balanced attack pool using only the attack TRAIN split."""

from pathlib import Path
import csv
import heapq

import numpy as np
import pandas as pd


# Project paths
ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "split_attack" / "train"
OUTPUT = ROOT / "data" / "attack_pool" / "balanced_attack_pool.csv"
REPORT = ROOT / "outputs" / "attack_pool_report.csv"

TARGET = 800
CHUNK_SIZE = 100_000
HASH_KEY = "0123456789123456"


def main():
    if not INPUT.is_dir():
        raise SystemExit(f"Missing attack training directory: {INPUT}")

    if OUTPUT.exists() or REPORT.exists():
        raise SystemExit(
            "Output already exists. Back up the previous pool "
            "and report before rerunning."
        )

    groups = sorted(p for p in INPUT.iterdir() if p.is_dir())

    if len(groups) != 33:
        raise SystemExit(
            f"Expected 33 attack-type folders, found {len(groups)}. "
            "Check the input directory structure."
        )

    all_samples = []
    report_rows = []
    expected_features = None

    for group_index, group in enumerate(groups, 1):
        files = sorted(group.rglob("*.csv"))

        if not files:
            raise SystemExit(f"No CSV files for {group.name}")

        # Keep the 800 rows with the smallest stable hashes.
        # This avoids loading the entire training set into memory.
        heap = []
        retained_hashes = set()
        seen_rows = 0

        for path in files:
            for chunk in pd.read_csv(
                path,
                chunksize=CHUNK_SIZE,
                dtype=np.float64,
            ):
                if expected_features is None:
                    expected_features = list(chunk.columns)

                    if len(expected_features) != 39:
                        raise SystemExit(
                            f"Expected 39 features, "
                            f"found {len(expected_features)} in {path}"
                        )

                if list(chunk.columns) != expected_features:
                    raise SystemExit(
                        f"Feature names/order mismatch: {path}"
                    )

                if not np.isfinite(chunk.to_numpy()).all():
                    raise SystemExit(
                        f"Non-finite values found: {path}"
                    )

                seen_rows += len(chunk)

                # Generate a reproducible fingerprint for each row.
                hashes = pd.util.hash_pandas_object(
                    chunk,
                    index=False,
                    hash_key=HASH_KEY,
                ).to_numpy(dtype=np.uint64)

                # Remove duplicate hashes within this chunk.
                unique_hashes, first_indices = np.unique(
                    hashes,
                    return_index=True,
                )

                n = min(TARGET, len(unique_hashes))

                if n == 0:
                    continue

                selected = np.argpartition(
                    unique_hashes,
                    n - 1,
                )[:n]

                selected = selected[
                    np.argsort(unique_hashes[selected])
                ]

                for ix in selected:
                    fingerprint = int(unique_hashes[ix])

                    if fingerprint in retained_hashes:
                        continue

                    if (
                        len(heap) == TARGET
                        and fingerprint >= -heap[0][0]
                    ):
                        continue

                    values = tuple(
                        float(x)
                        for x in chunk.iloc[
                            int(first_indices[ix])
                        ].to_numpy()
                    )

                    if len(heap) == TARGET:
                        _, old_hash, _ = heapq.heappop(heap)
                        retained_hashes.remove(old_hash)

                    heapq.heappush(
                        heap,
                        (-fingerprint, fingerprint, values),
                    )

                    retained_hashes.add(fingerprint)

        selected_count = len(heap)

        print(
            f"[{group_index}/{len(groups)}] "
            f"{group.name}: "
            f"train_rows={seen_rows:,}, "
            f"selected={selected_count}",
            flush=True,
        )

        report_rows.append({
            "attack_type": group.name,
            "train_rows": seen_rows,
            "selected": selected_count,
            "target": TARGET,
            "status": (
                "OK"
                if selected_count == TARGET
                else "INSUFFICIENT_UNIQUE_ROWS"
            ),
        })

        if selected_count < TARGET:
            raise SystemExit(
                f"{group.name}: only {selected_count} "
                f"distinct row fingerprints found. "
                f"Cannot construct {TARGET} unique samples. "
                "No output files written."
            )

        for _, _, values in sorted(
            heap,
            key=lambda item: item[1],
        ):
            all_samples.append((group.name, *values))

    # Save only after all 33 attack types pass.
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "attack_type",
            *expected_features,
        ])

        writer.writerows(all_samples)

    pd.DataFrame(report_rows).to_csv(
        REPORT,
        index=False,
    )

    print(
        f"\nDone: {len(all_samples):,} rows "
        f"across {len(groups)} attack types"
    )
    print(f"Pool: {OUTPUT}")
    print(f"Report: {REPORT}")
    print(
        "NOTE: Pool is raw/unscaled. "
        "Apply the benign train scaler before training."
    )


if __name__ == "__main__":
    main()
