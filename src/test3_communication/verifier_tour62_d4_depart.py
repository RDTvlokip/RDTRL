"""Tour 62 (07/10/2026) : la valeur de d4 = 1 - s4 au depart (1,8e-12, derive en t^-0,86 sous le
plancher d'Adam) influence-t-elle s = (moy d3 - noeud ideal)/pente et le rapport des medianes
delta'_r4/delta'_d3 ? Comparaison de deux departs de la ligne 4 de la reduction numba :
  A : etat de phase tel quel (d4 ~ 1,8e-12) ;
  B : z4 = e4 - o4 force a son noeud ideal (d4 = 3,13e-17).
12 phases, 2 000 000 pas apres 10 000 de mise en regime. NB : B n'est pas un etat d'equilibre
des moments d'Adam de la ligne 4 (m, v non modifies), mais les gradients de la ligne 4 sont
< 1e-14 dans les deux cas, donc sous le plancher eps 1e-10 : leur effet sur la ligne 4 est negligeable.
"""

import sys
import numpy as np
from mpmath import mp, mpf

sys.path.insert(0, ".")
import reduction_numba_tour59 as redlib
from hasard_reduction_tour59 import base_state, warmed
from verifier_tour61_decomposition_d3 import PLI, d3_noeud, r4_noeud, d4_noeud, pente_fn

mp.dps = 50

if __name__ == "__main__":
    ks = [int(a) for a in np.random.default_rng(7).integers(0, 4444, 12)]
    s1 = base_state()
    print("offset    depart   d4 reel au debut de la trace   s (moy d3 - noeud)/pente   rapport medianes r4/d3 (moy +- ES)")
    for off in sys.argv[1:]:
        delta = float(PLI + mpf(off))
        dn, rn = float(d3_noeud(PLI + mpf(off))), float(r4_noeud(PLI + mpf(off)))
        sl3, sl4 = pente_fn(d3_noeud, PLI + mpf(off)), pente_fn(r4_noeud, PLI + mpf(off))
        d4n = float(d4_noeud(PLI + mpf(off)))
        for dep in ("A", "B"):
            rng = np.random.default_rng(11)
            ss, rap, d4s = [], [], []
            for k in ks:
                w = warmed(s1, k, rng, 1e-9)
                if dep == "B":
                    w["x"][2] = w["x"][3] + float(np.log(26.0 * (1.0 - d4n) / d4n))
                redlib.avancer_obs(w, delta, 10000, np.zeros((1, 7)))
                z4 = float(w["x"][2] - w["x"][3])
                d4s.append(26.0 * np.exp(-z4) / (1.0 + 26.0 * np.exp(-z4)))
                rec = np.zeros((2_000_000, 7))
                first, maxrb, rb, done = redlib.avancer_obs(w, delta, 2_000_000, rec)
                if maxrb > 0.9:
                    continue
                ex = 26.0 * np.exp(-rec[:, 1]); d3 = ex / (1.0 + ex)
                r4 = 1.0 / (1.0 + np.exp(rec[:, 2]) + w["Se"])
                ss.append((d3.mean() - dn) / sl3)
                rap.append(((np.median(r4) - rn) / sl4) / ((np.median(d3) - dn) / sl3))
            rap = np.array(rap)
            print(f"{off:>8s}    {dep}      {np.median(d4s):.3e}                    {np.mean(ss):+.4e}              {rap.mean():.5f} +- {rap.std(ddof=1) / np.sqrt(len(rap)):.5f}", flush=True)
