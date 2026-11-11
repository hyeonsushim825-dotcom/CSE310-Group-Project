
from pathlib import Path
import numpy as np
import pandas as pd

files = sorted(Path(".").glob("BenignTraffic*.pcap.csv"))
rng = np.random.default_rng(42)

all_samples = []
reports = []

for file in files:
    print(f"\nProcessing: {file.name}")

    total = 0
    missing = None
    infinite = None
    min_values = None
    max_values = None
    sample_parts = []

    # 전체 파일을 50,000행씩 처리
    for chunk in pd.read_csv(file, chunksize=50000):
        numeric = chunk.select_dtypes(include="number")

        if missing is None:
            missing = pd.Series(0, index=numeric.columns)
            infinite = pd.Series(0, index=numeric.columns)
            min_values = pd.Series(np.inf, index=numeric.columns)
            max_values = pd.Series(-np.inf, index=numeric.columns)

        values = numeric.to_numpy(dtype=float)

        missing += numeric.isna().sum()
        infinite += pd.Series(
            np.isinf(values).sum(axis=0),
            index=numeric.columns
        )

        clean = numeric.replace(
            [np.inf, -np.inf], np.nan
        )

        min_values = pd.concat(
            [min_values, clean.min()], axis=1
        ).min(axis=1)

        max_values = pd.concat(
            [max_values, clean.max()], axis=1
        ).max(axis=1)

        # 무작위 우선순위 기반 샘플링
        # 전체 파일에서 최대 10,000행 유지
        keys = rng.random(len(chunk))
        chunk = chunk.copy()
        chunk["_sample_key"] = keys

        sample_parts.append(chunk)

        combined = pd.concat(
            sample_parts, ignore_index=True
        )
        combined = combined.nsmallest(
            10000, "_sample_key"
        )
        sample_parts = [combined]

        total += len(chunk)

    sampled = sample_parts[0].drop(
        columns="_sample_key"
    )
    sampled["source_file"] = file.name
    all_samples.append(sampled)

    print("Total rows:", total)
    print("NaN count:", int(missing.sum()))
    print("Inf count:", int(infinite.sum()))

    constants = min_values.index[
        min_values.eq(max_values)
    ].tolist()

    print("Constant features:", constants)

    reports.append({
        "file": file.name,
        "rows": total,
        "nan_count": int(missing.sum()),
        "inf_count": int(infinite.sum()),
        "constant_features": ", ".join(constants)
    })

# 결과 저장
pd.DataFrame(reports).to_csv(
    "quality_report.csv", index=False
)

pd.concat(all_samples, ignore_index=True).to_csv(
    "benign_sample_40k.csv", index=False
)

print("\nAnalysis completed!")
