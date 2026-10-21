"""Figure for GB1 part 2, from results/*.json (no training)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gb1

onehot = {r["split"]: r for r in json.load(open("results/results.json"))}
esm = {(r["size"], r["split"]): r for r in json.load(open("results/esm_results.json"))}
ctrl = {r["split"]: r for r in json.load(open("results/controls.json"))}
ci = lambda v: (v[0], v[0] - v[1], v[2] - v[0])

fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [3, 2]})

# left: approaches by split, ESM-2 650M
labels = ["one-hot ridge", "one-hot ensemble", "ESM-2 650M zero-shot (no training)", "ESM-2 650M ridge", "ESM-2 650M ensemble"]
w = 0.16
for k, lab in enumerate(labels):
    vals = []
    for sp in gb1.SPLITS:
        o, r = onehot[sp], esm[("650M", sp)]
        v = [[o["ridge"], *o["ridge_ci"]], [o["ensemble"], *o["ensemble_ci"]], r["zs_marginal"], r["ridge_pos"], r["ensemble"]][k]
        vals.append(ci(v))
    m, lo, hi = np.array(vals).T
    axes[0].bar(np.arange(len(gb1.SPLITS)) + (k - 2) * w, m, w, yerr=[lo, hi], capsize=2, label=lab)
axes[0].set_xticks(range(len(gb1.SPLITS)), gb1.SPLITS)
axes[0].set_ylabel("Spearman on test set (95% interval)")
axes[0].set_title("Does a protein language model help?")
axes[0].axhline(0, color="black", lw=0.6)
axes[0].legend(fontsize=8, ncol=2, loc="upper right")

# right: where the two_vs_rest gain comes from
sp = "two_vs_rest"
items = [("one-hot", [onehot[sp]["ensemble"], *onehot[sp]["ensemble_ci"]]),
         ("BLOSUM62", ctrl[sp]["blosum62"]),
         ("ESM-2\n650M\nno context", ctrl[sp]["esm_static"])] + \
        [(f"ESM-2\n{s}", esm[(s, sp)]["ensemble"]) for s in ["8M", "35M", "150M", "650M"]]
m, lo, hi = np.array([ci(v) for _, v in items]).T
colors = ["#888888", "#4c72b0", "#55a868"] + ["#8172b2"] * 4
axes[1].bar(range(len(items)), m, yerr=[lo, hi], capsize=3, color=colors)
axes[1].set_xticks(range(len(items)), [n for n, _ in items], fontsize=8)
axes[1].set_xlabel("purple: ESM-2 reading the whole GB1 domain, by model size")
axes[1].set_ylim(0.4, 0.82)
axes[1].set_ylabel("Spearman, same ensemble")
axes[1].set_title("Train on 1-2 mutations, test on 3-4:\nwhere the gain comes from")
for i, v in enumerate(m):
    axes[1].text(i, v + hi[i] + 0.006, f"{v:.2f}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig("results/esm_comparison.png", dpi=150)
print("saved results/esm_comparison.png")
