
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load sampled benign data
df = pd.read_csv("benign_sample_40k.csv")

source = df["source_file"]
X = df.drop(columns=["source_file"])

# Replace infinite values
X = X.replace([np.inf, -np.inf], np.nan)

# Fill missing values using median
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratio:")
print(pca.explained_variance_ratio_)
print("Total:", pca.explained_variance_ratio_.sum())

# Visualization
plt.figure(figsize=(10, 7))

for filename in source.unique():
    mask = source == filename

    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        s=3,
        alpha=0.3,
        label=filename
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("CICIoT2023 Benign Traffic PCA")
plt.legend(markerscale=4)
plt.tight_layout()

plt.savefig("benign_pca.png", dpi=200)
plt.show()
