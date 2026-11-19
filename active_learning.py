"""In silico directed evolution and active learning on the GB1 fitness landscape.

Simulates iterative experimental rounds where an ensemble surrogate proposes batches
of variants using different acquisition strategies (Random, Greedy, UCB, EI, Uncertainty).

Demonstrates sample efficiency: how rapidly Bayesian optimization discovers the global
optimum (FWAA, fitness = 8.762) and top-1% hits compared to high-throughput random screening.
"""

import json
import os
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import norm
import matplotlib.pyplot as plt

import gb1


# ---------------------------------------------------------------------------
# Acquisition Functions
# ---------------------------------------------------------------------------
def compute_acquisition(mu, sigma, strategy, y_best=0.0):
    """Compute acquisition scores for candidate variants.

    Args:
        mu: array of predicted mean fitness (N,)
        sigma: array of predicted epistemic uncertainty / std (N,)
        strategy: 'random', 'greedy', 'ucb_1', 'ucb_2', 'ei', 'uncertainty'
        y_best: current best fitness value among assayed variants
    """
    if strategy == "random":
        return np.random.uniform(0, 1, size=len(mu))
    elif strategy == "greedy":
        return mu
    elif strategy == "uncertainty":
        return sigma
    elif strategy == "ucb_1":
        return mu + 1.0 * sigma
    elif strategy == "ucb_2":
        return mu + 2.0 * sigma
    elif strategy == "ei":
        # Expected Improvement: E[max(y - y_best, 0)]
        # sigma <= 1e-8 treated as 0 improvement
        std = np.maximum(sigma, 1e-8)
        z = (mu - y_best) / std
        ei = (mu - y_best) * norm.cdf(z) + std * norm.pdf(z)
        ei[sigma < 1e-6] = 0.0
        return ei
    else:
        raise ValueError(f"Unknown acquisition strategy: {strategy}")


# ---------------------------------------------------------------------------
# Fast Ensemble Model
# ---------------------------------------------------------------------------
class FastEnsemble:
    """Ensemble of MLPs for mean prediction and epistemic uncertainty."""

    def __init__(self, n_models=5, hidden=128, in_dim=80, lr=1e-3, epochs=60):
        self.n_models = n_models
        self.hidden = hidden
        self.in_dim = in_dim
        self.lr = lr
        self.epochs = epochs
        self.models = []

    def fit(self, X_train, y_train, seed_base=0):
        """Fit ensemble members with bootstrap resampling and random initializations."""
        self.models = []
        n_samples = len(X_train)

        for m_idx in range(self.n_models):
            torch.manual_seed(seed_base * 100 + m_idx)
            model = gb1.MLP(hidden=self.hidden, in_dim=self.in_dim)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
            loss_fn = nn.MSELoss()

            # Bootstrap sample for each ensemble member
            rng = np.random.default_rng(seed_base * 100 + m_idx)
            boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
            X_b = X_train[boot_idx]
            y_b = y_train[boot_idx]

            dataset = TensorDataset(X_b, y_b)
            batch_size = min(32, max(8, n_samples // 4))
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            model.train()
            for _ in range(self.epochs):
                for xb, yb in loader:
                    loss = loss_fn(model(xb), yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            model.eval()
            self.models.append(model)

    def predict(self, X_candidates):
        """Return (mean, std) across ensemble predictions for candidate tensor."""
        preds = []
        with torch.no_grad():
            for model in self.models:
                preds.append(model(X_candidates).numpy())
        preds = np.array(preds)  # (n_models, N)
        mu = np.mean(preds, axis=0)
        sigma = np.std(preds, axis=0, ddof=1) if self.n_models > 1 else np.zeros_like(mu)
        return mu, sigma


# ---------------------------------------------------------------------------
# Directed Evolution Simulation
# ---------------------------------------------------------------------------
def run_directed_evolution(
    df_all,
    X_all,
    strategy="ucb_2",
    initial_budget=40,
    rounds=8,
    batch_size=25,
    seed=42,
):
    """Run an iterative active learning campaign.

    Args:
        df_all: DataFrame with all 8,733 variants
        X_all: One-hot encoded tensor for all variants
        strategy: acquisition function name
        initial_budget: number of variants in round 0
        rounds: number of active learning batches
        batch_size: number of variants synthesized per round
        seed: random seed for reproducibility

    Returns:
        history: dict of trajectory metrics per round
    """
    rng = np.random.default_rng(seed)
    n_total = len(df_all)
    all_targets = df_all["target"].to_numpy()

    # Identify wild-type index (target == 1.0, seq has VDGV)
    wt_mask = (df_all["sequence"].str[38] == "V") & (df_all["sequence"].str[39] == "D") & \
              (df_all["sequence"].str[40] == "G") & (df_all["sequence"].str[53] == "V")
    wt_idx = np.where(wt_mask)[0][0] if wt_mask.any() else 0

    # Start with wild-type + random initial variants
    initial_pool = set(rng.choice(n_total, size=initial_budget, replace=False))
    initial_pool.add(wt_idx)
    assayed_indices = list(initial_pool)

    history = {
        "round": [0],
        "total_assayed": [len(assayed_indices)],
        "max_fitness": [float(np.max(all_targets[assayed_indices]))],
        "top1_percent_hits": [int(np.sum(all_targets[assayed_indices] >= 5.0))],
        "batch_mean_fitness": [float(np.mean(all_targets[assayed_indices]))],
        "found_fwaa": [bool("FWAA" in [df_all.loc[i, "variant"] for i in assayed_indices])],
    }

    unassayed_mask = np.ones(n_total, dtype=bool)
    unassayed_mask[assayed_indices] = False

    for r in range(1, rounds + 1):
        unassayed_indices = np.where(unassayed_mask)[0]
        if len(unassayed_indices) == 0:
            break

        # Train ensemble on all data gathered so far
        X_train = X_all[assayed_indices]
        y_train = torch.tensor(all_targets[assayed_indices], dtype=torch.float32)

        ensemble = FastEnsemble(n_models=5, hidden=128, in_dim=80, lr=1e-3, epochs=50)
        ensemble.fit(X_train, y_train, seed_base=seed + r * 10)

        # Predict on all unassayed candidates
        X_cand = X_all[unassayed_indices]
        mu, sigma = ensemble.predict(X_cand)

        current_y_best = history["max_fitness"][-1]
        scores = compute_acquisition(mu, sigma, strategy=strategy, y_best=current_y_best)

        # Select top-B candidates by acquisition score
        k = min(batch_size, len(unassayed_indices))
        best_cand_rel = np.argpartition(scores, -k)[-k:]
        best_cand_rel = best_cand_rel[np.argsort(-scores[best_cand_rel])]
        chosen_indices = unassayed_indices[best_cand_rel]

        # Assay batch and update state
        assayed_indices.extend(chosen_indices)
        unassayed_mask[chosen_indices] = False

        batch_targets = all_targets[chosen_indices]
        curr_max = max(history["max_fitness"][-1], float(np.max(batch_targets)))
        top1_hits = int(np.sum(all_targets[assayed_indices] >= 5.0))
        found_fwaa = bool(history["found_fwaa"][-1] or ("FWAA" in [df_all.loc[i, "variant"] for i in chosen_indices]))

        history["round"].append(r)
        history["total_assayed"].append(len(assayed_indices))
        history["max_fitness"].append(curr_max)
        history["top1_percent_hits"].append(top1_hits)
        history["batch_mean_fitness"].append(float(np.mean(batch_targets)))
        history["found_fwaa"].append(found_fwaa)

    return history


# ---------------------------------------------------------------------------
# Benchmark Suite & Plotting
# ---------------------------------------------------------------------------
def run_all_benchmarks(data_path="data/splits/sampled.csv", n_seeds=5):
    """Run active learning benchmark across 5 strategies and multiple seeds."""
    print(f"Loading GB1 dataset from {data_path}...")
    df = pd.read_csv(data_path)

    # Extract 4-amino-acid variant label
    df["variant"] = df["sequence"].apply(
        lambda s: "".join(s[p] for p in gb1.POSITIONS)
    )

    X_all = gb1.encode(df)
    global_max = df["target"].max()
    print(f"Total dataset: {len(df)} variants. Global optimum fitness: {global_max:.4f} (FWAA)")

    strategies = ["random", "greedy", "ucb_1", "ucb_2", "ei"]
    all_results = {}

    for strat in strategies:
        print(f"\nEvaluating strategy: {strat.upper()} ({n_seeds} seeds)...")
        strat_histories = []
        t0 = time.time()
        for s in range(n_seeds):
            hist = run_directed_evolution(
                df, X_all, strategy=strat, initial_budget=40, rounds=8, batch_size=25, seed=s * 7 + 13
            )
            strat_histories.append(hist)
            print(f"  Seed {s+1}/{n_seeds} finished. Final max: {hist['max_fitness'][-1]:.3f}, Top1% hits: {hist['top1_percent_hits'][-1]}")
        print(f"  {strat.upper()} completed in {time.time()-t0:.1f}s")
        all_results[strat] = strat_histories

    os.makedirs("results", exist_ok=True)

    # Summarize & print table
    print("\n" + "=" * 78)
    print(f"{'Strategy':<14} | {'Final Max Fitness':<18} | {'Top-1% Hits':<16} | {'FWAA Found Rate':<15}")
    print("=" * 78)

    summary_json = {}
    for strat, hists in all_results.items():
        final_maxes = [h["max_fitness"][-1] for h in hists]
        final_hits = [h["top1_percent_hits"][-1] for h in hists]
        fwaa_found = [h["found_fwaa"][-1] for h in hists]

        max_mean, max_std = np.mean(final_maxes), np.std(final_maxes)
        hits_mean, hits_std = np.mean(final_hits), np.std(final_hits)
        fwaa_rate = np.mean(fwaa_found) * 100

        print(f"{strat:<14} | {max_mean:.3f} ± {max_std:.3f}       | {hits_mean:.1f} ± {hits_std:.1f}        | {fwaa_rate:.0f}%")
        summary_json[strat] = {
            "max_fitness_mean": float(max_mean),
            "max_fitness_std": float(max_std),
            "top1_hits_mean": float(hits_mean),
            "top1_hits_std": float(hits_std),
            "fwaa_found_rate": float(fwaa_rate),
        }

    print("=" * 78)

    with open("results/active_learning_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    # Generate publication-grade figure
    plot_active_learning(all_results, global_max)
    print("\nTrajectory plot saved to: results/active_learning_trajectory.png")


def plot_active_learning(results, global_max):
    """Plot active learning discovery curves."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    colors = {
        "random": "#7f7f7f",
        "greedy": "#1f77b4",
        "ucb_1": "#ff7f0e",
        "ucb_2": "#2ca02c",
        "ei": "#d62728",
    }
    labels = {
        "random": "Random Screening",
        "greedy": "Greedy Exploitation (β=0)",
        "ucb_1": "UCB (β=1.0)",
        "ucb_2": "UCB (β=2.0)",
        "ei": "Expected Improvement (EI)",
    }

    # Left Panel: Max fitness over total assays
    ax1 = axes[0]
    ax1.axhline(global_max, color="black", linestyle="--", linewidth=1.2, label=f"Global Optimum (FWAA = {global_max:.2f})")
    ax1.axhline(1.0, color="gray", linestyle=":", linewidth=1.0, label="Wild Type (VDGV = 1.0)")

    # Right Panel: Top 1% hits
    ax2 = axes[1]

    for strat, hists in results.items():
        rounds = hists[0]["total_assayed"]
        max_trajs = np.array([h["max_fitness"] for h in hists])
        hits_trajs = np.array([h["top1_percent_hits"] for h in hists])

        max_mean, max_std = np.mean(max_trajs, axis=0), np.std(max_trajs, axis=0)
        hits_mean, hits_std = np.mean(hits_trajs, axis=0), np.std(hits_trajs, axis=0)

        color = colors.get(strat, "blue")
        lbl = labels.get(strat, strat)

        # Plot Max Fitness
        ax1.plot(rounds, max_mean, color=color, linewidth=2, label=lbl)
        ax1.fill_between(rounds, max_mean - max_std, max_mean + max_std, color=color, alpha=0.15)

        # Plot Hits
        ax2.plot(rounds, hits_mean, color=color, linewidth=2, label=lbl)
        ax2.fill_between(rounds, hits_mean - hits_std, hits_mean + hits_std, color=color, alpha=0.15)

    ax1.set_title("A. Discovery of Maximum Fitness", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Cumulative Assayed Variants", fontsize=11)
    ax1.set_ylabel("Maximum Fitness Found", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=9)

    ax2.set_title("B. Top 1% High-Fitness Hits Found (Fitness ≥ 5.0)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Cumulative Assayed Variants", fontsize=11)
    ax2.set_ylabel("Cumulative Hits Discovered", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig("results/active_learning_trajectory.png")
    plt.close()


if __name__ == "__main__":
    run_all_benchmarks(data_path="data/splits/sampled.csv", n_seeds=5)
