"""CLAN training on pre-scaled CICIoT2023 NPY features, without labels.

Run from the repository root. Reuses the original encoder, loss, augmentation,
and LR schedule. See TRAINING_CICIOT.md for the experiment contract.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

# Required by deterministic CUDA matrix multiplication; set before CUDA starts.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model.model import ContrastiveMLP
from loss.clan_loss import CLANLoss
from loss.augmentations import make_augmentation
from util.schedules import WarmupCosineSchedule, LRSchedule

FORMAT_VERSION = 2
LOSS_COLUMNS = ['epoch', 'loss', 'frac_pos', 'seconds', 'lr', 'steps']


def load_features(path, expected_features=39):
    """Do not split, scale, remove columns, or use ground-truth labels here."""
    x = np.load(path, allow_pickle=False)
    if x.ndim != 2 or x.shape[1] != expected_features or len(x) < 2:
        raise ValueError(f"Expected (N>=2, {expected_features}), got {x.shape}")
    if x.dtype.kind not in 'fiu' or not np.isfinite(x).all():
        raise ValueError('Features must be finite real numeric values')
    with np.errstate(over='ignore'):
        x = np.ascontiguousarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Features overflow float32')
    return x


def sha256(path):
    result = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def load_trained_model(path, device='cpu'):
    """Shared entry point for evaluation/filtering; no labels needed."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != FORMAT_VERSION:
        raise ValueError('Unsupported checkpoint format; use a checkpoint from this training file')
    model = ContrastiveMLP(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device).eval(), checkpoint


@torch.no_grad()
def anomaly_scores(model, x, centroid, device, batch_size=1024):
    """Scores in ORIGINAL input row order; larger cosine distance = suspicious.

    x must already have the training feature order and training scaler applied.
    The caller owns held-out labels and evaluation; this function takes no labels.
    """
    if batch_size < 1 or x.ndim != 2 or len(x) == 0:
        raise ValueError('Nonempty 2-D input and positive batch-size required')
    model.eval()
    center = torch.as_tensor(centroid, dtype=torch.float32, device=device)
    scores = np.empty(len(x), dtype=np.float32)
    for start in range(0, len(x), batch_size):
        batch = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
        z = model(batch)
        distance = 1 - torch.nn.functional.cosine_similarity(z, center[None], dim=1)
        if not torch.isfinite(distance).all():
            raise FloatingPointError('Non-finite anomaly scores')
        scores[start:start + len(batch)] = distance.cpu().numpy()
    return scores


@torch.no_grad()
def training_centroid(model, x, device, batch_size=1024):
    """Mean of raw projected model(x) embeddings of ALL supplied train rows."""
    model.eval()
    total = None
    for start in range(0, len(x), batch_size):
        z = model(torch.from_numpy(x[start:start + batch_size]).to(device))
        if not torch.isfinite(z).all():
            raise FloatingPointError('Non-finite embeddings')
        part = z.double().sum(0).cpu()
        total = part if total is None else total + part
    return (total / len(x)).float()


def source_hashes():
    root = Path(__file__).resolve().parent
    names = ['train_ciciot.py', 'model/model.py', 'loss/clan_loss.py',
             'loss/augmentations.py', 'util/schedules.py', 'util/distance.py', 'util/factory.py']
    return {name: sha256(root / name) for name in names}


def rng_state(generator, device):
    np_state = np.random.get_state()
    return dict(torch=torch.get_rng_state(), loader=generator.get_state(),
                python=random.getstate(), numpy=[np_state[0], np_state[1].tolist(), *np_state[2:]],
                cuda=torch.cuda.get_rng_state(device) if device.startswith('cuda') else None)


def restore_rng(state, generator, device):
    torch.set_rng_state(state['torch'])
    generator.set_state(state['loader'])
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
    if device.startswith('cuda'):
        torch.cuda.set_rng_state(state['cuda'], device)


def save_checkpoint(checkpoint, destination):
    temp = destination.with_suffix('.tmp')
    torch.save(checkpoint, temp)
    temp.replace(destination)


def save_history(history, run_dir):
    # Reconstruct from the checkpoint on resume, avoiding duplicate/partial rows.
    temp = run_dir / 'loss.tmp'
    with temp.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=LOSS_COLUMNS)
        writer.writeheader()
        writer.writerows(history)
    temp.replace(run_dir / 'loss.csv')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-npy', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--neurons', default='1024,1024,1024,1024')
    p.add_argument('--d-out', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--margin', type=float, default=0.5)
    p.add_argument('--p-feature', type=float, default=0.1)
    p.add_argument('--resample-max', type=float, default=1.7)
    p.add_argument('--device', default='cuda')
    p.add_argument('--max-samples', type=int, default=None,
                   help='Random subset for smoke tests ONLY; omit for experiments')
    p.add_argument('--resume', action='store_true',
                   help='Continue run-dir/last.pt with the SAME data and training settings')
    p.add_argument('--stop-after-epoch', type=int, default=None,
                   help='Pause after this epoch, preserving the original total LR schedule')
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        p.error('epochs >= 1 and batch-size >= 2 required')
    if not 0 < args.p_feature <= 1 or args.resample_max <= 0 or args.lr <= 0:
        p.error('Require 0 < p-feature <= 1, resample-max > 0, lr > 0')
    if args.max_samples is not None and args.max_samples < 2:
        p.error('max-samples must be >= 2')
    if not all(math.isfinite(v) for v in [args.lr, args.margin, args.p_feature, args.resample_max]):
        p.error('Hyperparameters must be finite')
    if args.margin <= 0 or args.d_out < 1 or not 0 <= args.seed < 2**32:
        p.error('margin > 0, d-out >= 1 and 0 <= seed < 2**32 required')
    try:
        neurons = [int(n) for n in args.neurons.split(',')]
        if any(n < 1 for n in neurons):
            raise ValueError
    except ValueError:
        p.error('neurons must be comma-separated positive integers')
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= args.epochs:
        p.error('stop-after-epoch must be between 1 and epochs')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; verify server environment or use --device cpu for a smoke test')
    run_dir = Path(args.run_dir)
    previous = None
    if args.resume:
        if (run_dir / 'final.pt').exists():
            raise FileExistsError('This run is already complete (final.pt exists)')
        previous = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=True)
        if previous.get('format_version') != FORMAT_VERSION:
            raise ValueError('Cannot resume an older checkpoint format')
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty run directory: {run_dir}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    x = load_features(args.train_npy)
    source_shape = list(x.shape)
    if args.max_samples is not None:
        ids = np.random.default_rng(args.seed).choice(len(x), min(args.max_samples, len(x)), replace=False)
        x = x[ids]
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(x)),
                        batch_size=args.batch_size, shuffle=True, generator=generator,
                        num_workers=0, drop_last=True)
    if not len(loader):
        raise ValueError('Fewer samples than batch-size; reduce batch-size')
    config = dict(d_in=x.shape[1], neurons=neurons,
                  d_out=args.d_out, n_classes=None, dropout=0.0, residual=True)
    model = ContrastiveMLP(**config).to(args.device)
    augment = make_augmentation(dict(uniform_resample=dict(
        max_val=args.resample_max, mean=0.0, p_feature=args.p_feature, p_sample=1.0)))
    criterion = CLANLoss(m=args.margin, squared=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.0)
    schedule = LRSchedule(optimizer, WarmupCosineSchedule(
        start_val=1e-6, end_val=1e-6, ref_val=args.lr,
        T_max=args.epochs * len(loader), warmup_steps=(args.epochs // 10) * len(loader)))
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent,
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    metadata = dict(args=vars(args), model_config=config, source_shape=source_shape,
                    used_shape=list(x.shape), data_sha256=sha256(args.train_npy),
                    source_sha256=source_hashes(), base_commit=commit,
                    torch_version=str(torch.__version__), numpy_version=str(np.__version__),
                    cuda_version=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None,
                    steps_per_epoch=len(loader), drop_last=True,
                    omitted_per_epoch=len(x) % args.batch_size,
                    training_labels_used=False, smoke_test=args.max_samples is not None)
    first_epoch = 1
    history = []
    elapsed_training = 0.0
    if previous is not None:
        old = previous['metadata']
        for key in ['data_sha256', 'source_sha256', 'model_config', 'used_shape', 'torch_version', 'numpy_version']:
            if old[key] != metadata[key]:
                raise ValueError(f'Resume mismatch: {key}. Use the original data/code/environment')
        for key, value in vars(args).items():
            if key not in ['resume', 'stop_after_epoch', 'train_npy', 'run_dir'] and old['args'][key] != value:
                raise ValueError(f'Resume mismatch: {key}. Repeat the original training options')
        first_epoch = previous['epoch'] + 1
        if args.stop_after_epoch is not None and args.stop_after_epoch < first_epoch:
            raise ValueError('stop-after-epoch must be later than the saved epoch')
        model.load_state_dict(previous['model_state_dict'])
        optimizer.load_state_dict(previous['optimiser_state_dict'])
        schedule.load_state_dict(previous['schedular_state_dict'])
        history = previous['history']
        elapsed_training = previous['elapsed_training_seconds']
        restore_rng(previous['rng_state'], generator, args.device)
        checkpoint = previous
        metadata = old
        print(f'Resuming after epoch {previous["epoch"]}', flush=True)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / 'config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2), flush=True)
    save_history(history, run_dir)
    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        if args.device.startswith('cuda'):
            torch.cuda.synchronize(args.device)
        start = time.perf_counter()
        sum_loss = sum_frac = 0.0
        for (batch,) in loader:
            batch = batch.to(args.device)
            z, z_aug = model(torch.cat([batch, augment(batch)])).chunk(2)
            loss, frac = criterion(x=z, x_aug=z_aug)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at epoch {epoch}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
            optimizer.step()
            schedule.step()
            sum_loss += loss.item()
            sum_frac += frac.item()
        if args.device.startswith('cuda'):
            torch.cuda.synchronize(args.device)
        elapsed = time.perf_counter() - start
        elapsed_training += elapsed
        avg_loss = sum_loss / len(loader)
        history.append(dict(epoch=epoch, loss=avg_loss, frac_pos=sum_frac / len(loader),
                            seconds=elapsed, lr=optimizer.param_groups[0]['lr'], steps=len(loader)))
        # Save only tensors and primitive values; supports weights_only=True.
        checkpoint = dict(format_version=FORMAT_VERSION, model_state_dict=model.state_dict(),
                          model_config=config, optimiser_state_dict=optimizer.state_dict(),
                          schedular_state_dict=schedule.state_dict(), epoch=epoch,
                          metadata=metadata, loss=avg_loss, history=history,
                          elapsed_training_seconds=elapsed_training,
                          rng_state=rng_state(generator, args.device))
        save_checkpoint(checkpoint, run_dir / 'last.pt')
        save_history(history, run_dir)
        print(f'epoch {epoch}/{args.epochs} loss={avg_loss:.6f} seconds={elapsed:.2f}', flush=True)
        if args.stop_after_epoch == epoch and epoch < args.epochs:
            print('Paused. Repeat the same command with --resume (omit --stop-after-epoch).', flush=True)
            return
    start_center = time.perf_counter()
    centroid = training_centroid(model, x, args.device)
    checkpoint['centroid'] = centroid
    checkpoint['centroid_seconds'] = time.perf_counter() - start_center
    np.save(run_dir / 'centroid.npy', centroid.numpy())
    save_checkpoint(checkpoint, run_dir / 'final.pt')
    print(f'Finished: {run_dir / "final.pt"}', flush=True)


if __name__ == '__main__':
    main()
