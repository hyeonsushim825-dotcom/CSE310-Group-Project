"""Split cleaned attack CSVs into reproducible train/val/test partitions.

Run from anywhere: python scripts/split_attack.py
Input:  PROJECT_ROOT/data/cleaned_attack/**/*.csv
Output: PROJECT_ROOT/data/split_attack/{train,val,test}/...
Report: PROJECT_ROOT/outputs/attack_split_report.csv

This is a deterministic hash-based, approximately 70/15/15 split. Identical
39-feature rows always receive the same partition, including across files.
No scaler fitting, balancing, or contamination occurs here.
"""
from pathlib import Path
import csv
import re
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / 'data' / 'cleaned_attack'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'split_attack'
REPORT_PATH = PROJECT_ROOT / 'outputs' / 'attack_split_report.csv'
CHUNK_SIZE = 100_000
HASH_KEY = '0123456789123456'  # fixed for reproducible partitions


def attack_type_for(path: Path) -> str:
    """Prefer top-level attack folder; otherwise derive type from CSV name."""
    relative = path.relative_to(INPUT_DIR)
    if len(relative.parts) > 1:
        return relative.parts[0]
    stem = path.stem
    # Common CICIoT2023 naming: Type.pcap.csv / Type.pcap1.csv
    stem = re.sub(r'\.pcap\d*$', '', stem, flags=re.IGNORECASE)
    stem = re.sub(r'(?:[_-](?:part|chunk)\d+)$', '', stem, flags=re.IGNORECASE)
    return stem


def main() -> None:
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f'Input folder not found: {INPUT_DIR}')
    files = sorted(INPUT_DIR.rglob('*.csv'))
    if not files:
        raise FileNotFoundError(f'No CSV files found in {INPUT_DIR}')
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.rglob('*.csv')):
        raise RuntimeError(
            f'{OUTPUT_DIR} already contains CSV files. '
            'Move/remove old split outputs before running to avoid mixed results.'
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    totals = {}
    expected_columns = None
    for file_index, source in enumerate(files, 1):
        attack_type = attack_type_for(source)
        counts = totals.setdefault(attack_type, {'total': 0, 'train': 0, 'val': 0, 'test': 0})
        relative = source.relative_to(INPUT_DIR)
        per_file = {'train': 0, 'val': 0, 'test': 0}

        for chunk in pd.read_csv(source, chunksize=CHUNK_SIZE, low_memory=False):
            if expected_columns is None:
                expected_columns = list(chunk.columns)
                if len(expected_columns) != 39:
                    raise ValueError(f'Expected 39 features, found {len(expected_columns)}: {source}')
            if list(chunk.columns) != expected_columns:
                raise ValueError(f'Feature names/order differ in: {source}')
            if chunk.isna().to_numpy().any():
                raise ValueError(f'NaN found in cleaned file: {source}')

            # Same feature vector -> same hash -> same partition, even across CSVs.
            hashes = pd.util.hash_pandas_object(
                chunk, index=False, hash_key=HASH_KEY
            ).to_numpy(dtype='uint64')
            buckets = hashes % 10_000
            partitions = {
                'train': buckets < 7000,
                'val': (buckets >= 7000) & (buckets < 8500),
                'test': buckets >= 8500,
            }
            counts['total'] += len(chunk)
            for split, mask in partitions.items():
                selected = chunk.loc[mask]
                if selected.empty:
                    continue
                destination = OUTPUT_DIR / split / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                selected.to_csv(
                    destination, mode='a', index=False,
                    header=not destination.exists(), float_format='%.17g'
                )
                n = len(selected)
                counts[split] += n
                per_file[split] += n

        print(f'[{file_index}/{len(files)}] {relative} -> '
              f'train={per_file["train"]:,}, val={per_file["val"]:,}, '
              f'test={per_file["test"]:,}', flush=True)

    with REPORT_PATH.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['attack_type', 'total', 'train', 'val', 'test',
                         'train_pct', 'val_pct', 'test_pct'])
        grand = {'total': 0, 'train': 0, 'val': 0, 'test': 0}
        for attack_type, counts in sorted(totals.items()):
            total = counts['total']
            writer.writerow([attack_type, total, counts['train'], counts['val'],
                             counts['test'], *[round(100 * counts[s] / total, 4)
                                               for s in ('train', 'val', 'test')]])
            for key in grand:
                grand[key] += counts[key]
        total = grand['total']
        writer.writerow(['TOTAL', total, grand['train'], grand['val'], grand['test'],
                         *[round(100 * grand[s] / total, 4)
                           for s in ('train', 'val', 'test')]])

    print(f'\nDone. CSV files: {len(files)}, attack types inferred: {len(totals)}')
    print(f'Rows: {grand["total"]:,} | Train: {grand["train"]:,} | '
          f'Val: {grand["val"]:,} | Test: {grand["test"]:,}')
    print(f'Split files: {OUTPUT_DIR}')
    print(f'Report: {REPORT_PATH}')
    if len(totals) != 33:
        print('WARNING: Expected 33 attack types. Check attack_type_for() and report.')


if __name__ == '__main__':
    main()