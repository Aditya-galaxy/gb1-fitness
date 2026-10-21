"""GB1 part 2: do protein language model features help where one-hot models failed?

Uses features cached by esm_features.py. For each ESM-2 size found in features/ and each split:
  zero-shot marginal / pseudo-likelihood  - no training data at all
  ridge on ESM features                   - representations at the 4 positions, or averaged over the domain
  MLP ensemble on ESM features            - same networks, early stopping and seeds as run_all.py
Compared against the one-hot results in results/results.json. Writes results/esm_results.json and esm_table.md;
plot_esm.py draws the figure.
"""
import glob, json, re, time
import numpy as np
import torch
from scipy.stats import spearmanr
import gb1

SEEDS = [0, 1, 2, 3, 4]
LAMBDAS = [0.1, 1, 10, 100, 1e3, 1e4, 1e5]
SIZE_ORDER = {"8M": 8, "35M": 35, "150M": 150, "650M": 650}


def standardise(train_x, *others):
    mu, sd = train_x.mean(0), train_x.std(0) + 1e-6
    return [(a - mu) / sd for a in (train_x, *others)]


def ridge_fit_predict(x, y, xv, yv, xt):
    """Ridge with an intercept; the penalty is chosen by validation loss."""
    x, xv, xt = standardise(x, xv, xt)
    ym = y.mean()
    best = None
    for lam in LAMBDAS:
        if x.shape[0] < x.shape[1]:            # fewer rows than features: solve the smaller (dual) system
            a = np.linalg.solve(x @ x.T + lam * np.eye(len(x)), y - ym)
            w = x.T @ a
        else:
            w = np.linalg.solve(x.T @ x + lam * np.eye(x.shape[1]), x.T @ (y - ym))
        val = float(np.mean((xv @ w + ym - yv) ** 2))
        if best is None or val < best[0]:
            best = (val, lam, w)
    _, lam, w = best
    return xt @ w + ym, lam


def quintile_uncertainty(pred, spread, y):
    err = np.abs(pred - y)
    bins = np.quantile(pred, [0, .2, .4, .6, .8, 1])
    vals = [spearmanr(spread[m], err[m]).statistic
            for lo, hi in zip(bins[:-1], bins[1:]) if (m := (pred >= lo) & (pred <= hi)).sum() > 20]
    return float(np.mean(vals))


def ci(pred, y):
    return [gb1.spearman(pred, y), *gb1.bootstrap_ci(pred, y).tolist()]


onehot = {r["split"]: r for r in json.load(open("results/results.json"))}
files = sorted(glob.glob("features/esm2_*.npz"), key=lambda f: SIZE_ORDER[re.search(r"esm2_(\w+)\.npz", f).group(1)])
rows = []
for f in files:
    size = re.search(r"esm2_(\w+)\.npz", f).group(1)
    feats = np.load(f)
    index = {s: i for i, s in enumerate(feats["seqs"])}
    for split in gb1.SPLITS:
        t0 = time.time()
        train, val, test = gb1.load_split(split)
        idx = lambda fr: np.array([index[s] for s in fr["sequence"]])
        itr, iva, ite = idx(train), idx(val), idx(test)
        y, yv, yt = (fr["target"].to_numpy() for fr in (train, val, test))
        r = {"size": size, "split": split}
        r["zs_marginal"] = ci(feats["zs_marginal"][ite], yt)
        r["zs_pll"] = ci(feats["zs_pll"][ite], yt)
        for kind in ["pos", "mean"]:
            pred, lam = ridge_fit_predict(feats[kind][itr], y, feats[kind][iva], yv, feats[kind][ite])
            r[f"ridge_{kind}"] = ci(pred, yt); r[f"ridge_{kind}_lambda"] = lam
        x, xv, xt = (torch.tensor(a, dtype=torch.float32) for a in standardise(feats["pos"][itr], feats["pos"][iva], feats["pos"][ite]))
        preds = []
        for s in SEEDS:
            model, _ = gb1.train_mlp_xy(x, torch.tensor(y, dtype=torch.float32), xv, torch.tensor(yv, dtype=torch.float32), s)
            preds.append(gb1.predict_x(model, xt))
        ens, spread = np.mean(preds, 0), np.std(preds, 0)
        r["mlp_mean"] = float(np.mean([gb1.spearman(p, yt) for p in preds]))
        r["mlp_sd"] = float(np.std([gb1.spearman(p, yt) for p in preds]))
        r["ensemble"] = ci(ens, yt)
        r["uncertainty_raw"] = float(spearmanr(spread, np.abs(ens - yt)).statistic)
        r["uncertainty_within_quintiles"] = quintile_uncertainty(ens, spread, yt)
        rows.append(r)
        print(f"{size:5s} {split:14s} {time.time()-t0:5.1f}s  zs_pll={r['zs_pll'][0]:.3f}  ridge_pos={r['ridge_pos'][0]:.3f}  "
              f"ensemble={r['ensemble'][0]:.3f}  (one-hot ensemble {onehot[split]['ensemble']:.3f})", flush=True)

json.dump(rows, open("results/esm_results.json", "w"), indent=2)

fmt = lambda v: f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"
lines = ["| ESM-2 | Split | Zero-shot, wild-type context | Zero-shot, own context | Ridge, 4 positions | Ridge, domain average | Ensemble, 4 positions | One-hot ensemble | Uncertainty vs error* |",
         "|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    o = onehot[r["split"]]
    lines.append(f"| {r['size']} | {r['split']} | {fmt(r['zs_marginal'])} | {fmt(r['zs_pll'])} | {fmt(r['ridge_pos'])} | "
                 f"{fmt(r['ridge_mean'])} | {fmt(r['ensemble'])} | {o['ensemble']:.3f} | {r['uncertainty_within_quintiles']:.3f} |")
lines.append("\n*Spearman between ensemble disagreement and absolute error, averaged within quintiles of predicted fitness.")
open("results/esm_table.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

print("figure: run plot_esm.py")
