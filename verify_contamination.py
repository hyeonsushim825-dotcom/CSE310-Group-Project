
import numpy as np
from pathlib import Path

folder = Path("data/attack_pool")

for ratio in ["5pct", "10pct"]:
    X = np.load(folder / f"train_{ratio}_contaminated.npy")
    y = np.load(folder / f"train_{ratio}_labels.npy")

    print(f"\n=== {ratio} Verification ===")
    print("Feature shape:", X.shape)
    print("Label shape:", y.shape)
    print("Data type:", X.dtype)
    print("NaN:", np.isnan(X).sum())
    print("Inf:", np.isinf(X).sum())
    print("Attack count:", np.sum(y == 1))
    print("Benign count:", np.sum(y == 0))
    print("Unique labels:", np.unique(y))

    assert X.shape == (200000, 39)
    assert y.shape == (200000,)
    assert np.isfinite(X).all()
    assert np.sum(y == 1) == (10000 if ratio == "5pct" else 20000)
    assert np.sum(y == 0) == (190000 if ratio == "5pct" else 180000)

print("\nALL CHECKS PASSED!")
