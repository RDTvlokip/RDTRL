"""Tour 61 (07/10/2026) : le rapport delta'_med(r4)/delta'_med(d3) (0,961 a pli-1e-10,
0,967 a pli-1e-12) differe-t-il de 1 de facon significative ? Rapport calcule PAR PHASE
(16 phases, reduction numba, 2 000 000 pas apres 10 000 de mise en regime), moyenne et
erreur standard. Si la mediane de l'etat calme etait le noeud d'un delta decale, les deux
observables (d3 et r4) donneraient le meme delta', rapport = 1.
"""

import sys
import numpy as np
from mpmath import mp, mpf

sys.path.insert(0, ".")
import reduction_numba_tour59 as redlib
from hasard_reduction_tour59 import base_state, warmed
from verifier_tour61_decomposition_d3 import PLI, d3_noeud, r4_noeud, pente_fn

mp.dps = 40

if __name__ == "__main__":
    ks = [int(a) for a in np.random.default_rng(7).integers(0, 4444, 16)]
    s1 = base_state()
    rng = np.random.default_rng(11)
    print("offset     phases  rapport r4/d3 (moy +- erreur standard)   min / max par phase    ecart a 1 en erreurs standard")
    for off in ("-1e-10", "-1e-12"):
        delta = float(PLI + mpf(off))
        dn, rn = float(d3_noeud(PLI + mpf(off))), float(r4_noeud(PLI + mpf(off)))
        sl3, sl4 = pente_fn(d3_noeud, PLI + mpf(off)), pente_fn(r4_noeud, PLI + mpf(off))
        rap = []
        for k in ks:
            w = warmed(s1, k, rng, 1e-9)
            redlib.avancer_obs(w, delta, 10000, np.zeros((1, 7)))
            rec = np.zeros((2_000_000, 7))
            first, maxrb, rb, done = redlib.avancer_obs(w, delta, 2_000_000, rec)
            if maxrb > 0.9:
                continue
            ex = 26.0 * np.exp(-rec[:, 1]); d3 = ex / (1.0 + ex)
            r4 = 1.0 / (1.0 + np.exp(rec[:, 2]) + w["Se"])
            rap.append(((np.median(r4) - rn) / sl4) / ((np.median(d3) - dn) / sl3))
        rap = np.array(rap)
        se = rap.std(ddof=1) / np.sqrt(len(rap))
        print(f"{off:8s}   {len(rap):3d}     {rap.mean():.4f} +- {se:.4f}                       {rap.min():.4f} / {rap.max():.4f}      {(rap.mean() - 1) / se:+.1f}", flush=True)
