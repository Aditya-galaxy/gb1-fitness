"""Shared pieces: data loading, encoding, models, training with early stopping, evaluation."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import spearmanr

AMINO = "ACDEFGHIKLMNPQRSTVWY"
POSITIONS = [38, 39, 40, 53]          # the four positions that vary (0-based)
WILD_TYPE = "VDGV"                    # wild type at those positions
SPLITS = ["sampled", "three_vs_rest", "two_vs_rest", "one_vs_rest", "low_vs_high"]


# ---------- data ----------------------------------------------------------------
def load_split(split):
    """Return train, validation and test frames. Validation rows are train rows flagged True."""
    df = pd.read_csv(f"data/splits/{split}.csv")
    is_val = df["validation"] == True
    train = df[(df.set == "train") & ~is_val]
    val = df[(df.set == "train") & is_val]
    test = df[df.set == "test"]
    return train, val, test


def encode(frame):
    """One-hot encode the four varying positions: (n, 80) float tensor."""
    idx = torch.tensor([[AMINO.index(s[p]) for p in POSITIONS] for s in frame["sequence"]])
    return nn.functional.one_hot(idx, 20).float().flatten(1)


def targets(frame):
    return torch.tensor(frame["target"].to_numpy(), dtype=torch.float32)


def n_mutations(frame):
    return np.array([sum(s[p] != w for p, w in zip(POSITIONS, WILD_TYPE)) for s in frame["sequence"]])


# ---------- models --------------------------------------------------------------
def ridge_predict(train, test, lam=1.0):
    """Additive baseline: each mutation contributes independently."""
    X, Xt = encode(train).numpy(), encode(test).numpy()
    X, Xt = np.hstack([X, np.ones((len(X), 1))]), np.hstack([Xt, np.ones((len(Xt), 1))])
    w = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ train["target"].to_numpy())
    return Xt @ w


class MLP(nn.Module):
    def __init__(self, hidden=128, relu=True):
        super().__init__()
        act = nn.ReLU() if relu else nn.Identity()   # Identity = no non-linearity
        self.net = nn.Sequential(
            nn.Linear(80, hidden), act,
            nn.Linear(hidden, hidden), act,
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------- training --------------------------------------------------------------
def train_mlp(train, val, seed, relu=True, max_epochs=300, patience=25, lr=1e-3, batch_size=64):
    """Train with early stopping: keep the weights from the epoch with the lowest validation loss."""
    torch.manual_seed(seed)
    model = MLP(relu=relu)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    dl = DataLoader(TensorDataset(encode(train), targets(train)), batch_size=batch_size, shuffle=True)
    xv, yv = encode(val), targets(val)

    best_loss, best_state, best_epoch, waited = float("inf"), None, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, y in dl:
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(xv), yv).item()
        if val_loss < best_loss:
            best_loss, best_epoch, waited = val_loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_epoch


def predict(model, frame):
    model.eval()
    with torch.no_grad():
        return model(encode(frame)).numpy()


# ---------- evaluation -----------------------------------------------------------
def spearman(pred, y):
    return spearmanr(pred, y).statistic


def bootstrap_ci(pred, y, n=1000, seed=0):
    """95% confidence interval for Spearman by resampling the test set with replacement."""
    rng = np.random.default_rng(seed)
    pred, y = np.asarray(pred), np.asarray(y)
    stats = [spearman(pred[i], y[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return np.percentile(stats, [2.5, 97.5])
