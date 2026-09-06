"""
Optimized CPU implementation used for the Musk dataset.

This script reproduces the original HFCE score and evaluates:
- product-pairwise FCE (lambda = 1),
- regularized RFCE (delta = 0.01),
- smooth modulation (alpha = 4).

Usage:
    python run_musk_optimized.py --data ../Datasets/musk.mat

The .mat file must contain the variable 'trandata', with the class label
in the final column, as in the released HFCE repository.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score

EPS = np.float32(1e-4)

def main(data_path, output_path):
    D = loadmat(data_path)["trandata"]
    X = D[:, :-1].astype(np.float32)
    y = D[:, -1].astype(int)

    n, m = X.shape
    sigma = 0.2
    stds = X.std(axis=0, ddof=1)
    ID = ((X >= 1).all(axis=0)) & (X.max(axis=0) != X.min(axis=0))

    def relation(c):
        v = X[:, c]
        if ID[c]:
            return (v[:, None] == v[None, :]).astype(np.float32)
        radius = float(stds[c] / sigma)
        R = 1.0 - np.abs(v[:, None] - v[None, :])
        R[R < radius] = 0.0
        return R.astype(np.float32)

    def importance(R):
        s = R.sum(axis=1, dtype=np.float64)
        return (
            -np.mean(np.log2(s / n + 1e-4))
            -np.mean(np.log2((n - s) / n + 1e-4))
        )

    def all_scale(R):
        # Authors' released code adds 1e-4 before FCE/RFC/weights.
        A = R + EPS
        s = A.sum(axis=1, dtype=np.float64)
        diag = np.diag(A).astype(np.float64)
        tp = n * (n - 1) / 2.0
        tpx = (n - 1) * (n - 2) / 2.0

        # Original cardinality-based FCE
        p = 0.5 * s * (s - 1)
        F = np.mean(1 - p / tp)

        # Vectorized leave-one-out pair-count sums
        base = np.sum(s * s - s)
        Ats = A.T.dot(s.astype(np.float32)).astype(np.float64)
        col_r2 = np.einsum("ij,ij->j", A, A, dtype=np.float64)
        col_r = A.sum(axis=0, dtype=np.float64)
        sumP = 0.5 * (base - 2 * Ats + col_r2 + col_r)
        zdiag = s - diag
        sumP -= 0.5 * zdiag * (zdiag - 1)
        Fm = 1 - sumP / ((n - 1) * tpx)

        total = A.sum(dtype=np.float64)
        rfc = s - (total - 2 * s + diag) / (n - 1)
        w = np.sqrt(s / n)

        rf = np.clip(1 - Fm / F, 0, 1)
        g = np.where(
            rfc > 0,
            (n - np.abs(rfc)) / (2 * n),
            np.sqrt((n + np.abs(rfc)) / (2 * n)),
        )
        od = rf * g

        # Regularized RFCE, delta = 0.01
        rf_reg = np.maximum(0, (F - Fm) / (F + 0.01))
        od_reg = rf_reg * g

        # Smooth modulation, alpha = 4
        L = 1 / (2 * n)
        U = np.sqrt(1 - 1 / n)
        g_smooth = (U + L) / 2 - (U - L) / 2 * np.tanh(4 * rfc / n)
        od_smooth = rf * g_smooth

        # Product-pairwise lambda = 1
        # This optimized Musk run applies the quadratic correction to A=R+EPS,
        # matching the exact run used for the reported Musk AUC. Since all four
        # Musk variants have AUC 1.0, this implementation detail does not affect
        # the reported Musk AUC.
        h = A * (1 - A)
        q = h.sum(axis=1, dtype=np.float64)
        totalq = q.sum()
        colh = h.sum(axis=0, dtype=np.float64)
        hd = np.diag(h).astype(np.float64)
        P_pair = p + 0.5 * q
        F_pair = np.mean(1 - P_pair / tp)
        sumq_except = totalq - colh - q + hd
        sumP_pair = sumP + 0.5 * sumq_except
        Fm_pair = 1 - sumP_pair / ((n - 1) * tpx)
        rf_pair = np.clip(1 - Fm_pair / F_pair, 0, 1)
        od_pair = rf_pair * g

        return od, od_reg, od_smooth, od_pair, w

    start = time.time()
    imps = np.empty(m)
    acc = {k: np.zeros(n) for k in ["HFCE", "REG1e-2", "SMOOTH4", "PAIR1"]}

    # Single attributes
    for c in range(m):
        R = relation(c)
        imps[c] = importance(R)
        od, odr, ods, odp, w = all_scale(R)
        for key, z in [
            ("HFCE", od),
            ("REG1e-2", odr),
            ("SMOOTH4", ods),
            ("PAIR1", odp),
        ]:
            acc[key] += (1 - z) * w

    # Attribute-subset sequence. Musk uses Min fusion.
    order = np.argsort(imps, kind="stable")
    cum = None
    for c in order:
        R = relation(c)
        cum = R.copy() if cum is None else np.minimum(cum, R, out=cum)
        od, odr, ods, odp, w = all_scale(cum)
        for key, z in [
            ("HFCE", od),
            ("REG1e-2", odr),
            ("SMOOTH4", ods),
            ("PAIR1", odp),
        ]:
            acc[key] += (1 - z) * w

    aucs = {
        key: roc_auc_score(y, 1 - value / (2 * m))
        for key, value in acc.items()
    }
    aucs["seconds"] = time.time() - start
    print(aucs)
    pd.DataFrame([aucs]).to_csv(output_path, index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to musk.mat")
    ap.add_argument("--output", default="musk_variants_auc.csv")
    args = ap.parse_args()
    main(Path(args.data), Path(args.output))
