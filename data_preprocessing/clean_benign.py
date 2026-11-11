
from pathlib import Path
import pandas as pd
import numpy as np

files = [
    "BenignTraffic.pcap.csv",
    "BenignTraffic1.pcap.csv",
    "BenignTraffic2.pcap.csv",
    "BenignTraffic3.pcap.csv"
]

output_file = "benign_clean.csv"

total_original = 0
total_removed = 0
total_clean = 0

expected_columns = None

# 기존 출력 파일이 있다면 제거
output_path = Path(output_file)
if output_path.exists():
    output_path.unlink()

first_chunk = True

for file in files:
    print(f"\nProcessing: {file}")

    file_original = 0
    file_removed = 0

    for chunk in pd.read_csv(file, chunksize=50000):

        # Feature 구조 확인
        if expected_columns is None:
            expected_columns = chunk.columns.tolist()
        elif chunk.columns.tolist() != expected_columns:
            raise ValueError(
                f"Column mismatch in {file}"
            )

        original_count = len(chunk)

        # Inf를 NaN으로 변환
        chunk = chunk.replace(
            [np.inf, -np.inf], np.nan
        )

        # NaN 또는 Inf가 있는 행 제거
        chunk = chunk.dropna()

        clean_count = len(chunk)
        removed_count = original_count - clean_count

        # Benign label 추가
        chunk["label"] = 0

        # CSV에 이어서 저장
        chunk.to_csv(
            output_file,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        first_chunk = False

        file_original += original_count
        file_removed += removed_count

    file_clean = file_original - file_removed

    total_original += file_original
    total_removed += file_removed
    total_clean += file_clean

    print("Original:", file_original)
    print("Removed:", file_removed)
    print("Clean:", file_clean)

print("\n" + "=" * 45)
print("FINAL CLEANING SUMMARY")
print("=" * 45)

print("Original rows:", total_original)
print("Removed rows:", total_removed)
print("Clean rows:", total_clean)
print("Features:", len(expected_columns))
print("Output:", output_file)
