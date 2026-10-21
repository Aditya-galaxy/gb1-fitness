"""Run ESM-2 once over every GB1 variant and cache what the models need.

For each variant this saves:
  pos   - ESM-2's representation at the four mutated positions, concatenated (4 x d numbers)
  mean  - the representation averaged over the whole sequence (d numbers)
  zs_marginal - zero-shot score: log-probability of each mutation in the wild-type context,
                summed over positions (additive by construction)
  zs_pll      - zero-shot score: each mutated position masked in the variant's own context,
                so the score can depend on the other mutations
The model sees only the GB1 domain, the first 56 residues. The rest of each 265-residue sequence is
the fusion construct used in the experiment; it never varies, and all four mutated positions are inside
the domain.
Usage: python esm_features.py 35M     (or 8M, 150M, 650M)
"""
import sys, time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmForMaskedLM
from gb1 import POSITIONS, WILD_TYPE

MODELS = {"8M": "facebook/esm2_t6_8M_UR50D", "35M": "facebook/esm2_t12_35M_UR50D",
          "150M": "facebook/esm2_t30_150M_UR50D", "650M": "facebook/esm2_t33_650M_UR50D"}
size = sys.argv[1] if len(sys.argv) > 1 else "35M"
device = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH = 64

DOMAIN = 56                                   # GB1 domain = first 56 residues of each sequence
full_seqs = sorted(set(pd.read_csv("data/splits/sampled.csv")["sequence"]))
seqs = [s[:DOMAIN] for s in full_seqs]        # what the model reads
template = list(seqs[0])
for p, aa in zip(POSITIONS, WILD_TYPE):
    template[p] = aa
wild = "".join(template)

tok = AutoTokenizer.from_pretrained(MODELS[size])
model = EsmForMaskedLM.from_pretrained(MODELS[size]).to(device).eval()
TOK_POS = [p + 1 for p in POSITIONS]          # +1 because the tokenizer adds a start token


@torch.no_grad()
def run(batch_seqs, mask_at=None, hidden=True):
    enc = tok(batch_seqs, return_tensors="pt").to(device)
    ids = enc["input_ids"]
    if mask_at is not None:                   # mask one token position per sequence
        ids = ids.clone()
        ids[torch.arange(len(batch_seqs), device=device), torch.tensor(mask_at, device=device)] = tok.mask_token_id
    out = model(input_ids=ids, attention_mask=enc["attention_mask"], output_hidden_states=hidden)
    return out.logits.float(), (out.hidden_states[-1].float() if hidden else None)


t0 = time.time()
# 1. representations of every variant, unmasked
pos_feats, mean_feats = [], []
for i in range(0, len(seqs), BATCH):
    _, h = run(seqs[i:i + BATCH])
    pos_feats.append(h[:, TOK_POS, :].flatten(1).cpu())
    mean_feats.append(h[:, 1:-1, :].mean(1).cpu())          # drop start and end tokens
print(f"representations done in {time.time() - t0:.0f}s", flush=True)

aa_ids = {aa: tok.convert_tokens_to_ids(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}

# 2. zero-shot, wild-type marginals: mask each position in the wild type once
wt_lp = {}
for k, tp in enumerate(TOK_POS):
    logits, _ = run([wild], mask_at=[tp], hidden=False)
    wt_lp[k] = torch.log_softmax(logits[0, tp], -1).cpu()
def marginal(s):
    return sum(float(wt_lp[k][aa_ids[s[p]]] - wt_lp[k][aa_ids[WILD_TYPE[k]]]) for k, p in enumerate(POSITIONS))
zs_marginal = np.array([marginal(s) for s in seqs])

# 3. zero-shot, pseudo-likelihood in each variant's own context (4 masked copies per variant)
def pll_at_positions(batch_seqs):
    """Mask each of the four positions in turn; sum log p(true amino acid). One GPU-to-CPU copy per batch."""
    jobs = [(i, k) for i in range(len(batch_seqs)) for k in range(4)]
    logits, _ = run([batch_seqs[i] for i, _ in jobs], mask_at=[TOK_POS[k] for _, k in jobs], hidden=False)
    rows = torch.arange(len(jobs), device=device)
    cols = torch.tensor([TOK_POS[k] for _, k in jobs], device=device)
    true = torch.tensor([aa_ids[batch_seqs[i][POSITIONS[k]]] for i, k in jobs], device=device)
    picked = torch.log_softmax(logits[rows, cols], -1)[rows, true].cpu().numpy()
    return picked.reshape(len(batch_seqs), 4).sum(1)
wt_pll = pll_at_positions([wild])[0]
zs_pll = np.concatenate([pll_at_positions(seqs[i:i + BATCH // 4]) for i in range(0, len(seqs), BATCH // 4)]) - wt_pll
print(f"zero-shot done in {time.time() - t0:.0f}s", flush=True)

np.savez_compressed(f"features/esm2_{size}.npz", seqs=np.array(full_seqs),
                    pos=torch.cat(pos_feats).numpy(), mean=torch.cat(mean_feats).numpy(),
                    zs_marginal=zs_marginal, zs_pll=zs_pll)
print("saved", f"features/esm2_{size}.npz")
