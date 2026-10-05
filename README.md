# gb1-fitness

Predicting protein fitness from sequence on the GB1 landscape, with a focus on
extrapolation: can a model trained on simple mutants predict complex ones, and
does it know when it is wrong?

**Status:** work in progress. No results yet.

## Data

GB1 from the [FLIP benchmark](https://github.com/J-SNACKKB/FLIP/tree/main/splits/gb1):
about 160,000 variants of protein GB1 mutated at four positions (V39, D40, G41, V54),
each with a fitness value from a binding assay (Wu et al., eLife 2016).
The data is not stored in this repository. To download it:

```bash
curl -L -o data/splits.zip https://github.com/J-SNACKKB/FLIP/raw/main/splits/gb1/splits.zip
unzip -o data/splits.zip -d data
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Plan

1. Explore the data and fit a linear baseline in NumPy
2. Train a small neural network in PyTorch on the random split
3. Add a validation set, early stopping and fixed seeds
4. Evaluate across splits with Spearman correlation and bootstrap confidence intervals
5. Use an ensemble to estimate uncertainty and check whether it tracks error
6. Write up the results and limitations
