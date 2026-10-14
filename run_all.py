"""Steps 3-5: every split, five seeds, baselines, ensemble, uncertainty. Writes results/ ."""
import json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import gb1

SEEDS = [0, 1, 2, 3, 4]
os.makedirs("results", exist_ok=True)
rows, keep = [], {}

for split in gb1.SPLITS:
    t0 = time.time()
    train, val, test = gb1.load_split(split)
    y = test["target"].to_numpy()

    ridge = gb1.ridge_predict(train, test)
    linear_nets = [gb1.predict(gb1.train_mlp(train, val, s, relu=False)[0], test) for s in SEEDS]
    mlps, epochs = [], []
    for s in SEEDS:
        model, ep = gb1.train_mlp(train, val, s)
        mlps.append(gb1.predict(model, test)); epochs.append(ep)
    ensemble = np.mean(mlps, axis=0)
    spread = np.std(mlps, axis=0)            # disagreement between the five models

    single = [gb1.spearman(p, y) for p in mlps]
    lin = [gb1.spearman(p, y) for p in linear_nets]
    rows.append({
        "split": split, "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "ridge": gb1.spearman(ridge, y), "ridge_ci": gb1.bootstrap_ci(ridge, y).tolist(),
        "linear_net_mean": float(np.mean(lin)), "linear_net_sd": float(np.std(lin)),
        "mlp_mean": float(np.mean(single)), "mlp_sd": float(np.std(single)),
        "ensemble": gb1.spearman(ensemble, y), "ensemble_ci": gb1.bootstrap_ci(ensemble, y).tolist(),
        "uncertainty_vs_error": spearmanr(spread, np.abs(ensemble - y)).statistic,
        "stopped_epochs": epochs,
    })
    keep[split] = (ensemble, spread, y, gb1.n_mutations(test))
    print(f"{split:14s} done in {time.time()-t0:5.1f}s  ridge={rows[-1]['ridge']:.3f}  "
          f"mlp={rows[-1]['mlp_mean']:.3f}  ensemble={rows[-1]['ensemble']:.3f}", flush=True)

json.dump(rows, open("results/results.json", "w"), indent=2)

# ---- table ----
def ci(r, k): return f"{r[k]:.3f} [{r[k+'_ci'][0]:.3f}, {r[k+'_ci'][1]:.3f}]"
lines = ["| Split | Train / val / test | Ridge (additive) | Network without ReLU | MLP, 5 seeds | Ensemble of 5 | Uncertainty vs error |",
         "|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['split']} | {r['n_train']} / {r['n_val']} / {r['n_test']} | {ci(r,'ridge')} | "
                 f"{r['linear_net_mean']:.3f} ± {r['linear_net_sd']:.3f} | {r['mlp_mean']:.3f} ± {r['mlp_sd']:.3f} | "
                 f"{ci(r,'ensemble')} | {r['uncertainty_vs_error']:.3f} |")
open("results/table.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# ---- check: is disagreement just tracking the size of the prediction? ----
check = {}
for split in ["sampled", "three_vs_rest", "two_vs_rest", "one_vs_rest", "low_vs_high"]:
    ens, spread, y, _ = keep[split]
    err = np.abs(ens - y)
    bins = np.quantile(ens, [0, .2, .4, .6, .8, 1])
    within = [spearmanr(spread[m], err[m]).statistic
              for lo, hi in zip(bins[:-1], bins[1:]) if (m := (ens >= lo) & (ens <= hi)).sum() > 20]
    check[split] = float(np.mean(within))
    print(f"{split:14s} disagreement vs error, within prediction quintiles: {check[split]:.3f}")
json.dump(check, open("results/uncertainty_check.json", "w"), indent=2)

# ---- figure ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fracs = np.linspace(0.1, 1.0, 10)
for split in ["sampled", "three_vs_rest", "two_vs_rest", "one_vs_rest"]:
    ens, spread, y, _ = keep[split]
    order = np.argsort(spread)               # most confident first
    mae = [np.abs(ens[order[:int(f * len(y))]] - y[order[:int(f * len(y))]]).mean() for f in fracs]
    axes[0].plot(fracs * 100, mae, marker="o", label=split)
axes[0].set_xlabel("% of test set kept (most confident first)")
axes[0].set_ylabel("mean absolute error on kept variants")
axes[0].set_title("Is the ensemble more accurate when it agrees?")
axes[0].legend()

ens, spread, y, muts = keep["sampled"]
ks = [k for k in sorted(set(muts)) if (muts == k).sum() >= 30]   # skip groups too small to mean anything
vals = [spearmanr(ens[muts == k], y[muts == k]).statistic for k in ks]
bars = axes[1].bar([str(k) for k in ks], vals)
for b, k in zip(bars, ks):
    axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"n={int((muts == k).sum())}", ha="center")
axes[1].set_ylim(0, 1.08)
axes[1].set_xlabel("number of mutations (sampled split, test set; groups under 30 omitted)")
axes[1].set_ylabel("Spearman")
axes[1].set_title("Accuracy by number of mutations")
plt.tight_layout()
plt.savefig("results/uncertainty.png", dpi=150)
print("saved results/uncertainty.png")
