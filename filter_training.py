"""Label-free filtering for the team's CLAN repository. Uses the shared train_ciciot.py interface. See README_KO.md."""
import argparse
import json
from pathlib import Path
import time
import numpy as np


def select_indices(n_samples, remove_ratio, seed=42, distances=None):
    if n_samples < 1 or not np.isfinite(remove_ratio) or not 0 <= remove_ratio < 1:
        raise ValueError('Need nonempty data and 0 <= remove_ratio < 1.')
    n_remove = int(n_samples * remove_ratio)
    if distances is None:
        removed = np.random.default_rng(seed).choice(n_samples, n_remove, replace=False)
    else:
        distances = np.asarray(distances)
        if distances.shape != (n_samples,) or not np.isfinite(distances).all():
            raise ValueError('Need one finite distance per training row.')
        # Stable sorting breaks ties by original row index.
        removed = np.argsort(-distances, kind='stable')[:n_remove]
    keep = np.ones(n_samples, dtype=bool)
    keep[removed] = False
    return np.flatnonzero(keep), np.sort(removed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True, help='Scaled training .npy; no labels')
    p.add_argument('--output', type=Path, required=True, help='New output directory')
    p.add_argument('--remove-ratio', type=float, required=True)
    p.add_argument('--method', choices=['both', 'distance', 'random'], default='both')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--prefix', required=True, help='Example: train_5pct')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--device', default='cpu', help='cpu or cuda')
    args = p.parse_args()
    if args.method != 'random' and args.checkpoint is None:
        p.error('--checkpoint is required for distance/both')
    if args.batch_size < 1:
        p.error('--batch-size must be positive')
    if not args.prefix or Path(args.prefix).name != args.prefix or '/' in args.prefix or '\\' in args.prefix:
        p.error('--prefix must be a filename prefix, not a path')
    start_time = time.perf_counter()
    from train_ciciot import load_features, load_trained_model, anomaly_scores, sha256
    data = load_features(args.data)
    data_hash = sha256(args.data)
    if not np.isfinite(args.remove_ratio) or not 0 <= args.remove_ratio < 1:
        raise ValueError('remove-ratio must satisfy 0 <= ratio < 1.')
    # Avoid replacing previous results or any original input files.
    if args.output.exists():
        raise FileExistsError('Output already exists. Choose a new output directory.')

    distances = center = None
    checkpoint = None
    if args.method != 'random':
        model, checkpoint = load_trained_model(args.checkpoint, device=args.device)
        metadata = checkpoint.get('metadata', {})
        if metadata.get('data_sha256') != data_hash:
            raise ValueError('Checkpoint training data SHA256 does not match --data.')
        if metadata.get('smoke_test') or metadata.get('used_shape') != list(data.shape):
            raise ValueError('Use a full-data checkpoint, not a subset/smoke-test model.')
        if 'centroid' not in checkpoint:
            raise ValueError('Checkpoint has no centroid. Use completed final.pt, not last.pt.')
        center = checkpoint['centroid'].detach().cpu().numpy()
        if center.ndim != 1 or not np.isfinite(center).all() or np.linalg.norm(center) <= 1e-8:
            raise ValueError('Invalid or near-zero saved centroid.')
        distances = anomaly_scores(model, data, checkpoint['centroid'],
                                   args.device, batch_size=args.batch_size)

    methods = ['distance', 'random'] if args.method == 'both' else [args.method]
    selections = {}
    for method in methods:
        selections[method] = select_indices(
            len(data), args.remove_ratio, args.seed,
            distances if method == 'distance' else None,
        )
    args.output.mkdir(parents=True, exist_ok=False)
    for method, (kept, removed) in selections.items():
        np.save(args.output / f'{args.prefix}_{method}_s{args.seed}.npy', data[kept])
        np.save(args.output / f'{args.prefix}_{method}_s{args.seed}_kept_indices.npy', kept)
        np.save(args.output / f'{args.prefix}_{method}_s{args.seed}_removed_indices.npy', removed)
    if distances is not None:
        np.save(args.output / 'distance_scores.npy', distances)
        np.save(args.output / 'center.npy', center)

    summary = {
        'source_data': str(args.data.resolve()),
        'checkpoint': str(args.checkpoint.resolve()) if args.checkpoint else None,
        'input_shape': list(data.shape), 'input_dtype': str(data.dtype),
        'methods': methods, 'remove_ratio_requested': args.remove_ratio,
        'removed_count': int(len(data) * args.remove_ratio),
        'kept_count': len(data) - int(len(data) * args.remove_ratio),
        'actual_removed_ratio': int(len(data) * args.remove_ratio) / len(data),
        'metric': 'cosine' if distances is not None else None,
        'random_seed': args.seed, 'data_sha256': data_hash,
        'checkpoint_sha256': sha256(args.checkpoint) if args.checkpoint else None,
        'training_seed': checkpoint['metadata'].get('args', {}).get('seed') if checkpoint else None,
        'model_config': checkpoint['model_config'] if checkpoint else None,
        'embedding_source': 'model(x): projection output' if distances is not None else None,
        'center_definition': 'mean of all unlabelled training embeddings',
        'elapsed_seconds': time.perf_counter() - start_time,
        'labels_used': False,
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'Results saved to {args.output}')


if __name__ == '__main__':
    main()
