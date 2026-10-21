"""Control: is the ESM-2 gain protein knowledge, or just knowing which amino acids are similar?

Same MLP ensemble, same splits, different encodings of the four mutated positions:
  blosum62    - each amino acid as its row of the BLOSUM62 substitution matrix (4 x 20 numbers)
  esm_static  - ESM-2 650M's input embedding of each amino acid, with no protein context (4 x 1280)
Writes results/controls.json and results/controls_table.md.
"""
import json
import numpy as np
import torch
from Bio.Align import substitution_matrices
from transformers import AutoTokenizer, EsmForMaskedLM
import gb1

SEEDS = [0, 1, 2, 3, 4]
AMINO = gb1.AMINO
blosum = substitution_matrices.load("BLOSUM62")
BLOSUM = np.array([[blosum[a][b] for b in AMINO] for a in AMINO], dtype=np.float32)

name = "facebook/esm2_t33_650M_UR50D"
tok = AutoTokenizer.from_pretrained(name)
emb = EsmForMaskedLM.from_pretrained(name).esm.embeddings.word_embeddings.weight.detach().numpy()
STATIC = np.stack([emb[tok.convert_tokens_to_ids(a)] for a in AMINO]).astype(np.float32)

ENCODINGS = {"blosum62": BLOSUM, "esm_static": STATIC}


def encode(frame, table):
    idx = np.array([[AMINO.index(s[p]) for p in gb1.POSITIONS] for s in frame["sequence"]])
    return table[idx].reshape(len(idx), -1)


def standardise(a, *others):
    mu, sd = a.mean(0), a.std(0) + 1e-6
    return [torch.tensor((x - mu) / sd, dtype=torch.float32) for x in (a, *others)]


rows = []
for split in gb1.SPLITS:
    train, val, test = gb1.load_split(split)
    y, yv, yt = (torch.tensor(f["target"].to_numpy(), dtype=torch.float32) for f in (train, val, test))
    r = {"split": split}
    for enc_name, table in ENCODINGS.items():
        x, xv, xt = standardise(encode(train, table), encode(val, table), encode(test, table))
        preds = [gb1.predict_x(gb1.train_mlp_xy(x, y, xv, yv, s)[0], xt) for s in SEEDS]
        ens = np.mean(preds, 0)
        r[enc_name] = [gb1.spearman(ens, yt.numpy()), *gb1.bootstrap_ci(ens, yt.numpy()).tolist()]
    rows.append(r)
    print(split, {k: round(v[0], 3) for k, v in r.items() if k != "split"}, flush=True)

json.dump(rows, open("results/controls.json", "w"), indent=2)
onehot = {r["split"]: r for r in json.load(open("results/results.json"))}
esm = {(r["size"], r["split"]): r for r in json.load(open("results/esm_results.json"))}
fmt = lambda v: f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"
lines = ["| Split | One-hot | BLOSUM62 | ESM-2 650M, no context | ESM-2 8M, in context | ESM-2 650M, in context |",
         "|---|---|---|---|---|---|"]
for r in rows:
    sp = r["split"]
    lines.append(f"| {sp} | {fmt([onehot[sp]['ensemble'], *onehot[sp]['ensemble_ci']])} | {fmt(r['blosum62'])} | "
                 f"{fmt(r['esm_static'])} | {fmt(esm[('8M', sp)]['ensemble'])} | {fmt(esm[('650M', sp)]['ensemble'])} |")
lines.append("\nAll columns are the same five-network ensemble; only the encoding of the four positions differs.")
open("results/controls_table.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
