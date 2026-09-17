"""
Controlled stability diagnostics used in the manuscript.

This script anchors the two diagnostics to boundary regimes observed in the
released HFCE benchmark data:
  1) low-FCE sensitivity of original vs regularized RFCE;
  2) zero-crossing sensitivity of original vs smooth RFC modulation.

It assumes that reproduce_hfce.py and run_variants.py are in the same src/
directory and that the released HFCE .mat datasets are in src/Datasets/.

Usage:
    python stability_diagnostics.py
"""
import numpy as np
import pandas as pd

from reproduce_hfce import relation_col, importance, PARAMS
from run_variants import core_quantities, load, g_orig, g_smooth

SCAN_DATASETS = ["Glass", "Iono", "Iris", "Heart", "Hepa"]
DELTA = 1e-2

def collect_scales(name):
    X, _ = load(name)
    sigma, fusion = PARAMS[name]
    n, m = X.shape
    nominal = (X >= 1).all(axis=0) & (X.max(axis=0) != X.min(axis=0))

    relations, imps, scales = [], [], []
    for k in range(m):
        R = relation_col(X[:, k], bool(nominal[k]), sigma)
        relations.append(R)
        imps.append(importance(R))
        scales.append(("single", k, R.copy()))

    order = np.argsort(imps, kind="stable")
    pref = None
    for ell, idx in enumerate(order, 1):
        R = relations[idx]
        if pref is None:
            pref = R.copy()
        elif fusion == "Min":
            np.minimum(pref, R, out=pref)
        elif fusion == "Max":
            np.maximum(pref, R, out=pref)
        else:
            pref *= R
        scales.append(("subset", ell, pref.copy()))

    return n, scales

# ---- Diagnostic 1: smallest positive FCE observed in Heart ----
n_heart, heart_scales = collect_scales("Heart")
heart_items = []
for scale_type, scale_id, R in heart_scales:
    F, Fminus, _, _, rfc, _ = core_quantities(R, n_heart)
    if F > 0:
        heart_items.append((F, scale_type, scale_id, Fminus, rfc))
heart_items.sort(key=lambda x: x[0])
F, scale_type, scale_id, Fminus, rfc = heart_items[0]

active = np.where(Fminus < F)[0]
if len(active) == 0:
    raise RuntimeError("No active RFCE object found at selected Heart scale.")
i = active[np.argmin(F - Fminus[active])]
numerator = float(F - Fminus[i])
eps_rfce = 0.25 * numerator

orig_sensitivity = 1.0 / F
reg_sensitivity = 1.0 / (F + DELTA)
sensitivity_reduction = orig_sensitivity / reg_sensitivity

# symmetric numerical change around active Fminus
def rfce_original(fm):
    return max(0.0, (F - fm) / F)

def rfce_regularized(fm):
    return max(0.0, (F - fm) / (F + DELTA))

orig_minus = rfce_original(float(Fminus[i]) - eps_rfce)
orig_base = rfce_original(float(Fminus[i]))
orig_plus = rfce_original(float(Fminus[i]) + eps_rfce)
reg_minus = rfce_regularized(float(Fminus[i]) - eps_rfce)
reg_base = rfce_regularized(float(Fminus[i]))
reg_plus = rfce_regularized(float(Fminus[i]) + eps_rfce)

# ---- Diagnostic 2: closest-to-zero RFC among five benchmark datasets ----
candidates = []
for name in SCAN_DATASETS:
    n, scales = collect_scales(name)
    for scale_type2, scale_id2, R in scales:
        F0, Fm0, _, _, rfc2, _ = core_quantities(R, n)
        j = int(np.argmin(np.abs(rfc2)))
        candidates.append((
            abs(float(rfc2[j])), name, n, scale_type2, scale_id2, float(rfc2[j])
        ))
candidates.sort(key=lambda x: x[0])
epsilon, ref_dataset, n_ref, ref_type, ref_scale, observed_rfc = candidates[0]

r = np.array([-epsilon, epsilon], dtype=float)
g0 = g_orig(r, n_ref)
gs = g_smooth(r, n_ref)

jump_original = abs(float(g0[0] - g0[1]))
jump_smooth = abs(float(gs[0] - gs[1]))
jump_reduction = jump_original / jump_smooth

summary = pd.DataFrame([
    {
        "component": "RFCE",
        "dataset_reference": "Heart",
        "scale_type": scale_type,
        "scale_id": scale_id,
        "boundary_quantity": "FCE",
        "boundary_value": F,
        "original_metric": orig_sensitivity,
        "modified_metric": reg_sensitivity,
        "reduction_factor": sensitivity_reduction,
        "perturbation_magnitude": eps_rfce,
        "notes": "metric = local sensitivity to FCE_minus; delta=0.01",
    },
    {
        "component": "Modulation",
        "dataset_reference": ref_dataset,
        "scale_type": ref_type,
        "scale_id": ref_scale,
        "boundary_quantity": "abs(RFC)",
        "boundary_value": epsilon,
        "original_metric": jump_original,
        "modified_metric": jump_smooth,
        "reduction_factor": jump_reduction,
        "perturbation_magnitude": epsilon,
        "notes": "metric = |g(-eps)-g(+eps)|; alpha=4, sn=n, L=1/(2n), U=sqrt(1-1/n)",
    },
])

summary.to_csv("../results/stability_diagnostics.csv", index=False)

print(summary.to_string(index=False))
print()
print("RFCE numerical perturbation:")
print(f"FCE={F:.12g}, object={i}, eps={eps_rfce:.12g}")
print(f"original: {orig_minus:.12g}, {orig_base:.12g}, {orig_plus:.12g}")
print(f"regularized: {reg_minus:.12g}, {reg_base:.12g}, {reg_plus:.12g}")
print()
print("Smooth zero-crossing:")
print(f"reference={ref_dataset}, eps={epsilon:.12g}")
print(f"original: g(-eps)={g0[0]:.12g}, g(+eps)={g0[1]:.12g}")
print(f"smooth:   g(-eps)={gs[0]:.12g}, g(+eps)={gs[1]:.12g}")
