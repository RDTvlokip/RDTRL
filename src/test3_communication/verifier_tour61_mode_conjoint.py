"""Tour 61 (07/10/2026) : l'etat calme (mode conjoint de (z3, u)) est-il le noeud d'un
delta decale ? Les medianes marginales de d3 et de r4 donnent des delta' dont le rapport
vaut 0,961 / 0,967 (pas 1). Le mode conjoint (pic de l'histogramme 2D lisse de
(z3 = e3 - o3, u = l3 - l4)) n'est pas affecte par l'asymetrie differente des deux marginales.

delta'_x(mode) = (x(mode) - noeud_x(delta)) / (d noeud_x / d delta), x = d3, r4.
Memes 16 phases que verifier_tour61_rapport_phases.py (reduction numba, 2e6 pas).
"""

import sys
import numpy as np
from mpmath import mp, mpf
from scipy.ndimage import gaussian_filter

sys.path.insert(0, ".")
import reduction_numba_tour59 as redlib
from hasard_reduction_tour59 import base_state, warmed
from verifier_tour61_decomposition_d3 import PLI, d3_noeud, r4_noeud, pente_fn

mp.dps = 40


def mode_2d(z, u, nb=240, sigma=2.0):
    zl, zh = np.quantile(z, [0.001, 0.999]); ul, uh = np.quantile(u, [0.001, 0.999])
    H, ze, ue = np.histogram2d(z, u, bins=nb, range=[[zl, zh], [ul, uh]])
    Hs = gaussian_filter(H, sigma)
    i, j = np.unravel_index(np.argmax(Hs), Hs.shape)
    return 0.5 * (ze[i] + ze[i + 1]), 0.5 * (ue[j] + ue[j + 1]), (ze[1] - ze[0], ue[1] - ue[0])


if __name__ == "__main__":
    ks = [int(a) for a in np.random.default_rng(7).integers(0, 4444, 16)]
    s1 = base_state()
    rng = np.random.default_rng(11)
    print("offset     phases  delta'_d3(mode)    delta'_r4(mode)    rapport r4/d3 (moy +- ES)   | rappel medianes : rapport")
    rappel = {"-1e-10": 0.961, "-1e-12": 0.967}
    for off in ("-1e-10", "-1e-12"):
        delta = float(PLI + mpf(off))
        dn, rn = float(d3_noeud(PLI + mpf(off))), float(r4_noeud(PLI + mpf(off)))
        sl3, sl4 = pente_fn(d3_noeud, PLI + mpf(off)), pente_fn(r4_noeud, PLI + mpf(off))
        a3, a4, rap = [], [], []
        for k in ks:
            w = warmed(s1, k, rng, 1e-9)
            redlib.avancer_obs(w, delta, 10000, np.zeros((1, 7)))
            rec = np.zeros((2_000_000, 7))
            first, maxrb, rb, done = redlib.avancer_obs(w, delta, 2_000_000, rec)
            if maxrb > 0.9:
                continue
            zm, um, _ = mode_2d(rec[:, 1], rec[:, 2])
            ex = 26.0 * np.exp(-zm); d3m = ex / (1.0 + ex)
            r4m = 1.0 / (1.0 + np.exp(um) + w["Se"])
            x3, x4 = (d3m - dn) / sl3, (r4m - rn) / sl4
            a3.append(x3); a4.append(x4); rap.append(x4 / x3)
        rap = np.array(rap)
        print(f"{off:8s}   {len(rap):3d}    {np.mean(a3):+.3e}      {np.mean(a4):+.3e}      {rap.mean():.4f} +- {rap.std(ddof=1) / np.sqrt(len(rap)):.4f}"
              f"          | {rappel[off]}   (min/max par phase {rap.min():.3f} / {rap.max():.3f})", flush=True)
