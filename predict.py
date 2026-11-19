#!/usr/bin/env python3
"""Interactive inference CLI for GB1 fitness and uncertainty prediction.

Given any GB1 variant (e.g. 'FWAA' or full sequence), predicts binding fitness,
ensemble epistemic uncertainty, percentile rank, and mutation delta from wild type (VDGV).
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import torch

import gb1

POSITIONS_1BASED = [39, 40, 41, 54]
POSITIONS_0BASED = [38, 39, 40, 53]
WILD_TYPE = "VDGV"
CHECKPOINT_PATH = "checkpoints/ensemble_sampled.pt"
DATA_PATH = "data/splits/sampled.csv"


# ---------------------------------------------------------------------------
# Model Loading & Caching
# ---------------------------------------------------------------------------
def get_ensemble():
    """Load or train a 5-model MLP ensemble on the sampled split."""
    if os.path.exists(CHECKPOINT_PATH):
        weights = torch.load(CHECKPOINT_PATH, weights_only=True)
        models = [gb1.MLP() for _ in weights]
        for m, w in zip(models, weights):
            m.load_state_dict(w)
            m.eval()
        return models

    # Train if checkpoint does not exist
    print("Checkpoint not found. Training 5-model ensemble on sampled split (~15s)...", file=sys.stderr)
    train, val, _ = gb1.load_split("sampled")
    models = []
    weights = []
    for s in range(5):
        m, _ = gb1.train_mlp(train, val, seed=s, max_epochs=100, patience=15)
        m.eval()
        models.append(m)
        weights.append(m.state_dict())

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save(weights, CHECKPOINT_PATH)
    return models


# ---------------------------------------------------------------------------
# Sequence Extraction & Validation
# ---------------------------------------------------------------------------
def parse_variant(seq_str):
    """Extract and validate the 4-amino-acid variant code."""
    cleaned = seq_str.strip().upper()
    if len(cleaned) == 4:
        variant = cleaned
    elif len(cleaned) == 56:
        # Full 56-aa wild-type GB1 sequence (positions 39, 40, 41, 54 in 1-based = indices 38, 39, 40, 53)
        variant = "".join(cleaned[p] for p in [38, 39, 40, 53])
    elif len(cleaned) >= 200:
        # FLIP format with leader/tail tags
        variant = "".join(cleaned[p] for p in POSITIONS_0BASED)
    else:
        raise ValueError(
            f"Invalid sequence length ({len(cleaned)}). Expected 4 amino acids (e.g., 'FWAA'), "
            f"56-aa GB1 domain, or full FLIP construct."
        )

    for i, aa in enumerate(variant):
        if aa not in gb1.AMINO:
            raise ValueError(f"Invalid amino acid '{aa}' at position {POSITIONS_1BASED[i]}. Must be one of {gb1.AMINO}.")

    return variant


def get_mutation_delta(variant):
    """List amino acid mutations relative to wild type (VDGV)."""
    muts = []
    for i, (v_aa, wt_aa) in enumerate(zip(variant, WILD_TYPE)):
        pos_num = POSITIONS_1BASED[i]
        if v_aa != wt_aa:
            muts.append(f"{wt_aa}{pos_num}{v_aa}")
    return muts if muts else ["Wild Type (None)"]


def classify_fitness(fitness):
    """Categorize functional binding level."""
    if fitness >= 5.0:
        return "Super-binder (Top 1%)"
    elif fitness >= 1.5:
        return "Enhanced Binding (> WT)"
    elif fitness >= 0.8:
        return "Wild-Type-Like Activity"
    elif fitness >= 0.1:
        return "Impaired Binding"
    else:
        return "Non-functional / Fold Dead (< 0.1)"


# ---------------------------------------------------------------------------
# Prediction Engine
# ---------------------------------------------------------------------------
def predict_variant(variant, models, lookup_df=None):
    """Run ensemble prediction and metadata lookup."""
    # Encode one-hot
    idx = torch.tensor([[gb1.AMINO.index(aa) for aa in variant]])
    x = torch.nn.functional.one_hot(idx, 20).float().flatten(1)

    preds = [float(m(x).item()) for m in models]
    mu = float(np.mean(preds))
    sigma = float(np.std(preds, ddof=1)) if len(preds) > 1 else 0.0
    ci_lower = max(0.0, mu - 1.96 * sigma)
    ci_upper = mu + 1.96 * sigma

    # Check empirical ground truth if lookup dataset is available
    ground_truth = None
    percentile = None
    rank = None
    if lookup_df is not None:
        match = lookup_df[lookup_df["var"] == variant]
        if not match.empty:
            ground_truth = float(match.iloc[0]["target"])
        # Estimate percentile from distribution
        better_count = np.sum(lookup_df["target"].to_numpy() >= mu)
        total_count = len(lookup_df)
        rank = int(better_count) + 1
        percentile = float(100.0 * (1.0 - (better_count / total_count)))

    # Antibody interface contact notes
    interface_notes = []
    if variant[0] != "V":
        interface_notes.append("Position 39 contacts antibody Fc Asn434 (3.28 Å)")
    if variant[1] != "D":
        interface_notes.append("Position 40 contacts antibody Fc His433 (3.37 Å) & Asn434 (3.64 Å)")
    if variant[2] != "G":
        interface_notes.append("Position 41 contacts antibody Fc Asn434 (3.59 Å)")
    if variant[3] != "V":
        interface_notes.append("Position 54 contacts antibody Fc Asn434 (4.62 Å)")
    if not interface_notes:
        interface_notes.append("Wild-type native contacts at Fc Asn434/His433 epitope")

    return {
        "variant": variant,
        "mutations": get_mutation_delta(variant),
        "predicted_fitness_mean": round(mu, 3),
        "predicted_uncertainty_std": round(sigma, 3),
        "confidence_interval_95": [round(ci_lower, 3), round(ci_upper, 3)],
        "empirical_ground_truth": round(ground_truth, 4) if ground_truth is not None else None,
        "classification": classify_fitness(mu),
        "estimated_percentile": round(percentile, 2) if percentile is not None else None,
        "estimated_rank": f"{rank} / {total_count}" if rank is not None else None,
        "interface_contacts": interface_notes,
    }


def print_single_report(res):
    """Pretty print a single variant report."""
    print("=" * 64)
    print(f"  GB1 Variant: {res['variant']}    |  Classification: {res['classification']}")
    print("=" * 64)
    print(f"  Mutations from WT:       {', '.join(res['mutations'])}")
    print(f"  Predicted Fitness (Mean): {res['predicted_fitness_mean']:.3f}")
    print(f"  Uncertainty (Std):        ±{res['predicted_uncertainty_std']:.3f}")
    print(f"  95% Confidence Interval:  [{res['confidence_interval_95'][0]:.3f}, {res['confidence_interval_95'][1]:.3f}]")

    if res['empirical_ground_truth'] is not None:
        diff = res['predicted_fitness_mean'] - res['empirical_ground_truth']
        print(f"  Observed Wet-Lab Fitness: {res['empirical_ground_truth']:.4f} (Error: {diff:+.3f})")

    if res['estimated_percentile'] is not None:
        print(f"  Landscape Ranking:        {res['estimated_percentile']:.1f}th percentile ({res['estimated_rank']})")

    print("\n  Structural Interface Interaction:")
    for note in res["interface_contacts"]:
        print(f"    • {note}")
    print("=" * 64)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Predict binding fitness and uncertainty for GB1 protein variants."
    )
    parser.add_argument(
        "seq",
        nargs="?",
        default=None,
        help="4-letter variant (e.g. 'FWAA', 'VDGV') or full sequence.",
    )
    parser.add_argument(
        "--seq",
        dest="seq_opt",
        help="Alternative flag for 4-letter variant string.",
    )
    parser.add_argument(
        "--seqs",
        help="Comma-separated list of variants (e.g. 'FWAA,FYAA,VDGV,AAAA').",
    )
    parser.add_argument(
        "--file",
        help="Path to text file containing one variant per line.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of human-readable text.",
    )

    args = parser.parse_args()

    # Collect input variants
    input_strings = []
    if args.seq:
        input_strings.append(args.seq)
    elif args.seq_opt:
        input_strings.append(args.seq_opt)
    elif args.seqs:
        input_strings.extend(s.strip() for s in args.seqs.split(",") if s.strip())
    elif args.file:
        if not os.path.exists(args.file):
            print(f"Error: file not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        with open(args.file) as f:
            input_strings.extend(line.strip() for line in f if line.strip())
    else:
        # Default example if no args passed
        input_strings = ["FWAA", "FYAA", "VDGV", "GNGV"]

    # Load lookup table for ground truth if available
    lookup_df = None
    if os.path.exists(DATA_PATH):
        try:
            lookup_df = pd.read_csv(DATA_PATH)
            lookup_df["var"] = lookup_df["sequence"].apply(
                lambda s: "".join(s[p] for p in POSITIONS_0BASED)
            )
        except Exception:
            pass

    models = get_ensemble()

    results = []
    for s in input_strings:
        try:
            var = parse_variant(s)
            res = predict_variant(var, models, lookup_df=lookup_df)
            results.append(res)
        except ValueError as err:
            print(f"Warning: Skipping '{s}': {err}", file=sys.stderr)

    if not results:
        print("No valid variants evaluated.", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(results if len(results) > 1 else results[0], indent=2))
    else:
        for r in results:
            print_single_report(r)
            print()


if __name__ == "__main__":
    main()
