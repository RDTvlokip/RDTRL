"""Tour 59 (29/09/2026) : H59-24. Taux d'echappement lambda(delta) de la
falaise a eps 1e-10, sur les runs de lancer_tour59_hasard.sh (10 phases
par delta, censure a droite a PAS pas).

Estimateur du maximum de vraisemblance d'un taux constant avec censure :
    lambda = n_echappements / somme(min(t_i, PAS)),
intervalle de confiance de Poisson approche (exact de Garwood) sur n.
Puis ln(lambda) contre delta : pente d ln(lambda)/d delta, comparee a la
prediction du carnet (~9e9, soit x100 par 5e-10).
"""

import glob
import re
from collections import defaultdict

import numpy as np
from scipy import stats

PAS = 30000


def garwood(n, alpha=0.05):
    lo = 0.0 if n == 0 else stats.chi2.ppf(alpha / 2, 2 * n) / 2
    hi = stats.chi2.ppf(1 - alpha / 2, 2 * n + 2) / 2
    return lo, hi


if __name__ == "__main__":
    import sys
    # argument optionnel : prefixe(s) des fichiers, separes par des virgules (hz par defaut ; hzo = test hors echantillon)
    prefixes = (sys.argv[1] if len(sys.argv) > 1 else "hz").split(",")
    par_delta = defaultdict(list)
    for f in [g for p in prefixes for g in glob.glob(f"D:/tmp/rdtrl_t59_{p}_k*_1e-10_*.txt")]:
        m = re.search(r"_k(\d+)_1e-10_([0-9.e+-]+)\.txt$", f)
        txt = open(f).read()
        if not m or "bascule" not in txt:
            continue
        k, off = int(m.group(1)), float(m.group(2))
        pas = int(re.search(r"pas=(\d+)", txt).group(1))
        b = int(re.search(r"bascule=(-?\d+)", txt).group(1))
        par_delta[off].append((k, b if b >= 0 else None, pas))
    xs, ys = [], []
    print("offset      n_runs  echapp.  exposition    lambda (par pas)   IC 95 %              temps d'echappement")
    for off in sorted(par_delta):
        L = par_delta[off]
        n = sum(1 for _, b, _ in L if b is not None)
        expo = sum((b if b is not None else pas) for _, b, pas in L)
        lam = n / expo if expo else float("nan")
        lo, hi = garwood(n)
        temps = sorted(b for _, b, _ in L if b is not None)
        print(f"{off:.2e}  {len(L):5d}  {n:6d}  {expo:10d}    {lam:.3e}        [{lo / expo:.2e} ; {hi / expo:.2e}]   {temps}")
        if n > 0:
            xs.append(off); ys.append(np.log(lam))
    if len(xs) >= 3:
        p = np.polyfit(xs, ys, 1)
        r2 = np.corrcoef(xs, ys)[0, 1] ** 2
        print(f"\nln(lambda) = {p[0]:.3e} * delta_offset + {p[1]:.2f}   R2={r2:.3f}   (prediction du carnet : pente ~ 9e9)")
