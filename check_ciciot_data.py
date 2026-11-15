"""Check prepared CICIoT2023 arrays without changing them or fitting a scaler.

Optional label checks are dataset QA only. train_ciciot.py never reads labels.
This cannot verify feature names, scaler provenance, or train/test leakage.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from train_ciciot import load_features, sha256


EXPECTED = {
    'benign_train_200k_scaled.npy': (200_000, 0, None),
    'train_5pct_contaminated.npy': (210_526, 10_526, 'train_5pct_labels.npy'),
    'train_10pct_contaminated.npy': (222_222, 22_222, 'train_10pct_labels.npy'),
}


def row_keys(x):
    """Exact float32 row representation; allows order-independent multiset checks."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    return x.view(np.dtype((np.void, x.dtype.itemsize * x.shape[1]))).reshape(-1)


def check_label_alignment(x, labels, clean, expected_attacks):
    if labels.shape != (len(x),) or labels.dtype.kind not in 'biuf':
        raise ValueError(f'Invalid label shape/type: {labels.shape}, {labels.dtype}')
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('Labels must be 0=benign and 1=attack')
    attacks = int((labels == 1).sum())
    if attacks != expected_attacks:
        raise ValueError(f'Expected {expected_attacks} attacks, got {attacks}')
    benign = x[labels == 0]
    if not np.array_equal(np.sort(row_keys(clean)), np.sort(row_keys(benign))):
        raise ValueError('Labelled benign rows do not match the fixed clean training array')
    return dict(benign_rows=len(benign), attack_rows=attacks,
                contamination_percent=100 * attacks / len(x),
                same_fixed_benign_rows=True)


def check_directory(data_dir, check_labels=False):
    data_dir = Path(data_dir)
    report = dict(status='PASS', checks=[], labels_checked=check_labels,
                  limitation='Does not verify scaler fitting, feature order, or held-out leakage')
    clean = load_features(data_dir / 'benign_train_200k_scaled.npy')
    for name, (rows, attacks, label_name) in EXPECTED.items():
        path = data_dir / name
        x = clean if label_name is None else load_features(path)
        if x.shape != (rows, 39):
            raise ValueError(f'{name}: expected {(rows,39)}, got {x.shape}')
        result = dict(file=name, shape=list(x.shape), training_dtype=str(x.dtype),
                      sha256=sha256(path), finite=True,
                      min_value=float(x.min()), max_value=float(x.max()))
        if check_labels and label_name is not None:
            label_path = data_dir / label_name
            labels = np.load(label_path, allow_pickle=False)
            result['label_qa'] = check_label_alignment(x, labels, clean, attacks)
            result['label_sha256'] = sha256(label_path)
        report['checks'].append(result)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', default='datasets')
    p.add_argument('--check-labels', action='store_true',
                   help='Also require the two label arrays and verify counts/alignment')
    p.add_argument('--report', help='Optional output JSON path')
    args = p.parse_args()
    report = check_directory(args.data_dir, args.check_labels)
    payload = json.dumps(report, indent=2)
    if args.report:
        path = Path(args.report)
        inputs = [Path(args.data_dir) / name for name in EXPECTED]
        inputs += [Path(args.data_dir) / value[2] for value in EXPECTED.values() if value[2]]
        if path.resolve() in [f.resolve() for f in inputs]:
            raise ValueError('Report must not overwrite an input array')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding='utf-8')
    print(payload)


if __name__ == '__main__':
    main()
