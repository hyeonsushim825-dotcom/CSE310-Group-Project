
from pathlib import Path
import pandas as pd
import numpy as np
import csv

# Project root: CSE310-Group-Project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Original dataset location inside the project
ROOT = PROJECT_ROOT / "CSV"

# Inspection report output
OUT = PROJECT_ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# Benign 컬럼을 기준으로 사용
benign_path = ROOT / "Benign_Final" / "BenignTraffic.pcap.csv"
benign_cols = pd.read_csv(benign_path, nrows=0).columns.tolist()

results = []

# Benign_Final을 제외한 모든 공격 CSV 검사
attack_files = sorted(
    p for p in ROOT.rglob("*.csv")
    if "Benign_Final" not in p.parts
)

print(f"Attack CSV files: {len(attack_files)}")

for i, path in enumerate(attack_files, 1):
    print(f"[{i}/{len(attack_files)}] {path.name}")

    cols = pd.read_csv(path, nrows=0).columns.tolist()

    same_features = set(cols) == set(benign_cols)
    same_order = cols == benign_cols

    total_rows = 0
    invalid_rows = 0

    # 한 번에 50,000행씩 읽기
    for chunk in pd.read_csv(path, chunksize=50_000):
        total_rows += len(chunk)

        # 숫자형 변환이 불가능한 값도 NaN으로 처리
        numeric = chunk.apply(
            pd.to_numeric, errors="coerce"
        )

        invalid = ~np.isfinite(
            numeric.to_numpy(dtype=float)
        ).all(axis=1)

        invalid_rows += invalid.sum()

    results.append({
        "attack_type": path.parent.name,
        "file": path.name,
        "rows": total_rows,
        "invalid_rows": int(invalid_rows),
        "same_features": same_features,
        "same_order": same_order,
        "feature_count": len(cols),
    })

report = pd.DataFrame(results)
report.to_csv(OUT / "attack_inspection.csv", index=False)

print("\n=== SUMMARY ===")
print("Total attack rows:", report["rows"].sum())
print("Invalid rows:", report["invalid_rows"].sum())
print("Schema mismatches:", (~report["same_features"]).sum())
print("\nSaved: outputs/attack_inspection.csv")