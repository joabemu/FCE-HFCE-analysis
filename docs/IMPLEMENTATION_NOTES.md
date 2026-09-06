# Implementation notes

The reproduction follows the authors' **released HFCE implementation**, not only the printed equations.

Important details observed in the released code:

1. Numerical relations use `std(attribute)/sigma` with sample standard deviation (`ddof=1`), matching PyTorch's default correction used by the original code.
2. The original implementation adds `1e-4` to each single/fused relation before the weight, FCE, RFC, and leave-one-out computations.
3. Attribute importance in the released code is computed as the sum of two logarithmic terms. The second term is a logarithm of the complement row sum, which differs from the linear expression printed as CE in the paper. The reproduction scripts intentionally follow the released code because that is what reproduces the published AUC values.
4. The released datasets reproduce the published AUC values to three decimals on all 20 datasets. Two metadata discrepancies were observed in the dataset summary: the released Audiology file contains 57 outliers rather than 53, and the released Lymphography file contains 18 conditional attributes rather than 8. We do not alter the released files.
5. `run_variants.py` uses the pairwise correction on the underlying proper relation before the `1e-4` stabilization. The separate optimized Musk script uses the stabilized matrix for that correction; Musk has AUC 1.0 for both original and pairwise variants, so this does not affect the reported Musk AUC. For publication, the pre-stabilization interpretation in `run_variants.py` is the preferred semantics for the pairwise extension.

The scripts are provided to make every implementation choice explicit and reproducible.
