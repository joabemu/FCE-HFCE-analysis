"""
Controlled equal-cardinality experiment used in the manuscript.

The family F(t)=(1,1-t,t,0) has sigma-count 2 for every t.
The cardinality-based pair count is therefore constant, whereas the
product-pairwise count is 1+t(1-t).

Usage:
    python synthetic_pairwise.py --output-dir ../results/synthetic
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def main(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic family
    t = np.linspace(0, 0.5, 101)
    mu = np.column_stack([np.ones_like(t), 1-t, t, np.zeros_like(t)])
    sigma_count = np.full_like(t, 2.0)
    p_card = np.ones_like(t)
    qf = np.sum(mu * (1-mu), axis=1)
    p_pair = p_card + 0.5*qf

    pd.DataFrame({
        "t": t,
        "mu1": mu[:,0],
        "mu2": mu[:,1],
        "mu3": mu[:,2],
        "mu4": mu[:,3],
        "sigma_count": sigma_count,
        "P_card": p_card,
        "quadratic_fuzziness": qf,
        "P_pair_product": p_pair,
        "pairwise_extra_information": p_pair-p_card,
    }).to_csv(output_dir/"fixed_cardinality_family.csv", index=False)

    # Controlled ranking diagnostic
    rng = np.random.default_rng(20260906)
    n_regular, n_diffuse = 80, 20
    t_regular = rng.uniform(0.00, 0.08, n_regular)
    t_diffuse = rng.uniform(0.35, 0.50, n_diffuse)
    tt = np.concatenate([t_regular, t_diffuse])
    y = np.concatenate([
        np.zeros(n_regular, dtype=int),
        np.ones(n_diffuse, dtype=int)
    ])

    mu2 = np.column_stack([np.ones_like(tt), 1-tt, tt, np.zeros_like(tt)])
    p_card2 = np.ones_like(tt)
    qf2 = np.sum(mu2 * (1-mu2), axis=1)
    p_pair2 = p_card2 + 0.5*qf2

    auc_card = roc_auc_score(y, p_card2)
    auc_pair = roc_auc_score(y, p_pair2-p_card2)

    pd.DataFrame({
        "class": y,
        "t": tt,
        "sigma_count": 2.0,
        "P_card": p_card2,
        "quadratic_fuzziness": qf2,
        "P_pair_product": p_pair2,
        "pairwise_extra_information": p_pair2-p_card2,
    }).to_csv(output_dir/"controlled_ranking_task.csv", index=False)

    lambdas = [0, .25, .5, .75, 1.0]
    rows = []
    for lam in lambdas:
        score = p_card2 + lam*0.5*qf2
        rows.append({"lambda":lam, "AUC":roc_auc_score(y, score)})
    pd.DataFrame(rows).to_csv(output_dir/"lambda_auc.csv", index=False)

    print(f"Cardinality-only AUC: {auc_card:.3f}")
    print(f"Pairwise extra-information AUC: {auc_pair:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="../results/synthetic")
    args = ap.parse_args()
    main(args.output_dir)
