
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Original CICIoT2023 CSV folder
ROOT = PROJECT_ROOT / "CSV"

# Cleaned Attack CSV output
OUT = PROJECT_ROOT / "data" / "cleaned_attack"
OUT.mkdir(parents=True, exist_ok=True)

# Preprocessing report output
REPORT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 50_000

# ============================================================
# 1. Load Benign feature structure
# ============================================================

benign_file = ROOT / "Benign_Final" / "BenignTraffic.pcap.csv"

if not benign_file.exists():
    raise FileNotFoundError(
        f"Benign reference file not found: {benign_file}"
    )

feature_cols = pd.read_csv(
    benign_file, nrows=0
).columns.tolist()

print("=== ATTACK PREPROCESSING ===")
print("Benign features:", len(feature_cols))

# ============================================================
# 2. Find Attack CSV files
# ============================================================

attack_files = sorted(
    p for p in ROOT.rglob("*.csv")
    if "Benign_Final" not in p.parts
)

if not attack_files:
    raise FileNotFoundError(f"No Attack CSV files found in {ROOT}")

print("Attack CSV files:", len(attack_files))

report_rows = []

# ============================================================
# 3. Preprocess Attack CSVs
# ============================================================

for i, path in enumerate(attack_files, start=1):

    attack_type = path.parent.name

    print(
        f"[{i}/{len(attack_files)}] "
        f"{attack_type}/{path.name}",
        flush=True
    )

    # Check feature names against Benign
    attack_cols = pd.read_csv(
        path, nrows=0
    ).columns.tolist()

    if (
        len(attack_cols) != len(feature_cols)
        or set(attack_cols) != set(feature_cols)
    ):
        raise ValueError(
            f"Feature mismatch: {path}"
        )

    # Keep the original folder structure
    relative_path = path.relative_to(ROOT)
    output_path = OUT / relative_path
    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    original_rows = 0
    removed_rows = 0
    cleaned_rows = 0

    first_chunk = True

    for chunk in pd.read_csv(
        path, chunksize=CHUNK_SIZE
    ):

        original_rows += len(chunk)

        # Match Benign feature order
        chunk = chunk[feature_cols]

        # Convert all feature values to numeric
        numeric = chunk.apply(
            pd.to_numeric, errors="coerce"
        )

        # Check for NaN and +/-Inf
        values = numeric.to_numpy(
            dtype=np.float64
        )

        valid_mask = np.isfinite(values).all(axis=1)

        # Keep only valid rows
        cleaned = numeric.loc[valid_mask]

        removed_rows += int((~valid_mask).sum())
        cleaned_rows += len(cleaned)

        # Save all cleaned rows (no sampling)
        cleaned.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        first_chunk = False

    # Verify row counts
    assert original_rows == removed_rows + cleaned_rows

    report_rows.append({
        "attack_type": attack_type,
        "file": path.name,
        "original_rows": original_rows,
        "removed_rows": removed_rows,
        "cleaned_rows": cleaned_rows,
        "feature_count": len(feature_cols)
    })

    print(
        f"  Original: {original_rows:,} | "
        f"Removed: {removed_rows:,} | "
        f"Cleaned: {cleaned_rows:,}",
        flush=True
    )

# ============================================================
# 4. Save preprocessing report
# ============================================================

report = pd.DataFrame(report_rows)

report_path = (
    REPORT_DIR / "attack_preprocessing_report.csv"
)

report.to_csv(report_path, index=False)

print("\n=== PREPROCESSING SUMMARY ===")
print("Attack CSV files:", len(attack_files))
print("Original rows:", report["original_rows"].sum())
print("Removed rows:", report["removed_rows"].sum())
print("Cleaned rows:", report["cleaned_rows"].sum())
print("Feature count:", len(feature_cols))

print("\nCleaned data saved to:")
print(OUT)

print("\nReport saved to:")
print(report_path)