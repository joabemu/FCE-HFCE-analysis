# FCE / HFCE analysis and extensions — reproducibility code

This repository contains the code used for the experimental part of the manuscript:

**Fuzzy combination entropy: Properties, heterogeneous relations, and outlier scoring**

It provides:

- reproduction of the published HFCE AUC results;
- product-pairwise FCE (`lambda = 1`);
- regularized RFCE (`delta = 1e-3` and `delta = 1e-2`);
- smooth RFC modulation (`alpha = 4`);
- the controlled equal-cardinality experiment used to demonstrate the identifiability difference between cardinality-based and pairwise representations;
- an optimized CPU script for the large Musk dataset.

## 1. Data

The 20 benchmark datasets are **not redistributed here**. Download the released HFCE repository/data associated with Su et al. and place the original `.mat` files in:

```text
Datasets/
```

Each `.mat` file must contain the variable `trandata`, with the class label in the last column, exactly as in the released HFCE data.

The scripts expect the original HFCE filenames.

## 2. Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
```

## 3. Repository layout

```text
src/
    reproduce_hfce.py
    run_variants.py
    run_musk_optimized.py
    synthetic_pairwise.py
results/
    hfce_20datasets_main_variants_auc.csv
    synthetic/
docs/
    IMPLEMENTATION_NOTES.md
requirements.txt
README.md
```

Before running `reproduce_hfce.py` or `run_variants.py`, copy/symlink the original HFCE `Datasets` directory into `src/Datasets/`, or edit the data path in the scripts.

## 4. Reproduce original HFCE results

From `src/`:

```bash
python reproduce_hfce.py Aud Chess Lym
```

or, for all datasets:

```bash
python reproduce_hfce.py
```

The script follows implementation details of the released HFCE code so that the published AUC values are reproduced to the reported precision.

## 5. Run analytical variants

From `src/`:

```bash
python run_variants.py Aud Glass Arr
```

or:

```bash
python run_variants.py
```

Variants evaluated:

- `HFCE`
- `PAIR1`: product-pairwise extension, `lambda=1`
- `REG1e-3`: regularized RFCE, `delta=1e-3`
- `REG1e-2`: regularized RFCE, `delta=1e-2`
- `SMOOTH4`: smooth modulation, `alpha=4`
- `PAIR1_SMOOTH4`

For the very large Musk dataset, use the optimized script below if the generic CPU version is too slow.

## 6. Musk

```bash
python run_musk_optimized.py --data ../Datasets/musk.mat
```

The run used in the manuscript yielded AUC = 1.0 for HFCE, pairwise `lambda=1`, regularized RFCE `delta=0.01`, and smooth modulation.

## 7. Controlled equal-cardinality experiment

```bash
python synthetic_pairwise.py --output-dir ../results/synthetic
```

The experiment uses

```text
F(t) = (1, 1-t, t, 0),  0 <= t <= 0.5
```

for which the sigma-count is exactly 2 for every `t`.

Therefore:

```text
P_card = 1
P_pair = 1 + t(1-t)
```

The controlled ranking diagnostic gives:

```text
cardinality-only AUC = 0.5
pairwise extra-information AUC = 1.0
```

This is a diagnostic representation experiment, not a claim that pairwise FCE universally improves outlier detection.

## 8. Reproducibility note

See `docs/IMPLEMENTATION_NOTES.md` for implementation details of the released HFCE code that are necessary to reproduce the published numbers.

## 9. Citation

A formal citation will be added after publication. Until then, please cite the manuscript and the original HFCE article from which the benchmark data and baseline implementation originate.

## 10. License

Before making the repository public, add the license you want to use for **your own code**. Do not copy a license from the original HFCE repository unless its terms explicitly allow that reuse.
