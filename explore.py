"""Step 1: look at the GB1 data and fit a linear baseline (no PyTorch yet)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

DATA = "data/splits"
SPLITS = ["sampled", "three_vs_rest", "two_vs_rest", "one_vs_rest", "low_vs_high"]
AMINO = "ACDEFGHIKLMNPQRSTVWY"  # the 20 amino acids


def load(split):
    return pd.read_csv(f"{DATA}/{split}.csv")


# ---- 1. How big is each split? ------------------------------------------------
print("1. Rows per split")
for s in SPLITS:
    df = load(s)
    n_val = (df["validation"] == True).sum()
    print(f"   {s:14s} total={len(df):5d}  train={(df.set == 'train').sum():5d} "
          f"(of which validation={n_val:4d})  test={(df.set == 'test').sum():5d}")

df = load("sampled")

# ---- 2. What does fitness look like? ------------------------------------------
y = df["target"].to_numpy()
print("\n2. Fitness (target)")
print(f"   min={y.min():.3f}  median={np.median(y):.3f}  mean={y.mean():.3f}  max={y.max():.3f}")
print(f"   share below 0.1: {(y < 0.1).mean():.1%}   share above 1.0 (better than wild type): {(y > 1).mean():.1%}")
plt.figure(figsize=(6, 3.5))
plt.hist(y, bins=80)
plt.xlabel("fitness (wild type = 1)")
plt.ylabel("number of variants")
plt.title("GB1 fitness distribution")
plt.tight_layout()
plt.savefig("fitness_hist.png", dpi=150)

# ---- 3. Which positions actually vary? ----------------------------------------
seqs = np.array([list(s) for s in df["sequence"]])          # shape (n, length)
varying = [i for i in range(seqs.shape[1]) if len(set(seqs[:, i])) > 1]
wild_type = df.loc[df["target"] == 1.0, "sequence"].iloc[0]  # wild type has fitness exactly 1
print("\n3. Sequences")
print(f"   length={seqs.shape[1]}  positions that vary (0-based)={varying}")
print(f"   wild type at those positions: {''.join(wild_type[i] for i in varying)}")


# ---- 4 and 5. Linear baseline --------------------------------------------------
def one_hot(sequences):
    """Each varying position becomes 20 zeros with a single 1 for its amino acid."""
    X = np.zeros((len(sequences), len(varying) * 20))
    for row, s in enumerate(sequences):
        for k, pos in enumerate(varying):
            X[row, k * 20 + AMINO.index(s[pos])] = 1.0
    return X


def n_mutations(sequences):
    return np.array([sum(s[p] != wild_type[p] for p in varying) for s in sequences])


def ridge_fit(X, y, lam=1.0):
    """Ridge regression: solve (X'X + lam*I) w = X'y. The bias is a column of ones."""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    return np.linalg.solve(A, Xb.T @ y)


def ridge_predict(X, w):
    return np.hstack([X, np.ones((len(X), 1))]) @ w


print("\n4-5. Linear baseline (ridge on one-hot), Spearman on the test set")
for s in SPLITS:
    d = load(s)
    tr, te = d[d.set == "train"], d[d.set == "test"]
    w = ridge_fit(one_hot(tr.sequence), tr.target.to_numpy())
    pred = ridge_predict(one_hot(te.sequence), w)
    rho = spearmanr(pred, te.target).statistic
    muts_tr = sorted(int(m) for m in set(n_mutations(tr.sequence)))
    muts_te = sorted(int(m) for m in set(n_mutations(te.sequence)))
    print(f"   {s:14s} spearman={rho:6.3f}   train has {muts_tr} mutations, test has {muts_te}")
