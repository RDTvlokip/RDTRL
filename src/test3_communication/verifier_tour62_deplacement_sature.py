"""Tour 62 (07/10/2026) : la mediane de l'etat calme reste-t-elle a distance FIXE du point de
pli (D_c = mediane_c - valeur_c au pli, c = d3, r4) quand x = delta_c - delta -> 0 ?
Si oui, le rapport delta'_r4/delta'_d3 se deduit sans ajustement de (D_d, D_r) a un seul x_ref
et de la fermeture exacte du noeud :
  rapport(x) = [(D_r - (r4_noeud(x) - r4_pli)) / pente_r(x)] / [(D_d - (d3_noeud(x) - d3_pli)) / pente_d(x)].
Reduction numba, 16 phases x 2 000 000 pas apres 10 000 de mise en regime.
Usage : python verifier_tour62_deplacement_sature.py <offsets...> ; x_ref = le plus petit |offset| donne.
"""

import sys
import numpy as np
from mpmath import mp, mpf

sys.path.insert(0, ".")
import reduction_numba_tour59 as redlib
from hasard_reduction_tour59 import base_state, warmed
from verifier_tour61_decomposition_d3 import PLI, d3_noeud, r4_noeud, pente_fn

mp.dps = 60

if __name__ == "__main__":
    offs = sys.argv[1:]
    ks = [int(a) for a in np.random.default_rng(7).integers(0, 4444, 16)]
    s1 = base_state()
    # point de pli : noeud a x = 1e-30 (ecart au vrai point de pli ~ sqrt(1e-30) 3e-1 de ... negligeable : 1e-15 en u)
    d3_pli = float(d3_noeud(PLI - mpf("1e-30"))); r4_pli = float(r4_noeud(PLI - mpf("1e-30")))
    res = {}
    for off in offs:
        delta = float(PLI + mpf(off))
        rng = np.random.default_rng(11)
        dn, rn = float(d3_noeud(PLI + mpf(off))), float(r4_noeud(PLI + mpf(off)))
        sl3, sl4 = pente_fn(d3_noeud, PLI + mpf(off)), pente_fn(r4_noeud, PLI + mpf(off))
        Dd, Dr, rap = [], [], []
        for k in ks:
            w = warmed(s1, k, rng, 1e-9)
            redlib.avancer_obs(w, delta, 10000, np.zeros((1, 7)))
            rec = np.zeros((2_000_000, 7))
            first, maxrb, rb, done = redlib.avancer_obs(w, delta, 2_000_000, rec)
            if maxrb > 0.9:
                continue
            ex = 26.0 * np.exp(-rec[:, 1]); d3 = ex / (1.0 + ex)
            r4 = 1.0 / (1.0 + np.exp(rec[:, 2]) + w["Se"])
            md3, mr4 = np.median(d3), np.median(r4)
            Dd.append(md3 - d3_pli); Dr.append(mr4 - r4_pli)
            rap.append(((mr4 - rn) / sl4) / ((md3 - dn) / sl3))
        res[off] = dict(Dd=np.mean(Dd), Dr=np.mean(Dr), rap=np.mean(rap), se=np.std(rap, ddof=1) / np.sqrt(len(rap)),
                        dn=dn, rn=rn, sl3=sl3, sl4=sl4)
        print(f"{off:>8s}   D_d = {res[off]['Dd']:+.5e}   D_r = {res[off]['Dr']:+.5e}   D_r/D_d = {res[off]['Dr'] / res[off]['Dd']:.5f}   rapport mesure {res[off]['rap']:.5f} +- {res[off]['se']:.5f}", flush=True)
    ref = min(offs, key=lambda o: abs(float(o)))
    Dd0, Dr0 = res[ref]["Dd"], res[ref]["Dr"]
    print(f"\nconstantes figees a x_ref = {abs(float(ref)):.0e} : D_d = {Dd0:+.5e}, D_r = {Dr0:+.5e}")
    print("offset     D_d(x)/D_d(ref)   D_r(x)/D_r(ref)   rapport predit (2 constantes)   rapport mesure     ecart")
    for off in offs:
        q = res[off]
        pred = ((Dr0 - (q["rn"] - r4_pli)) / q["sl4"]) / ((Dd0 - (q["dn"] - d3_pli)) / q["sl3"])
        print(f"{off:>8s}   {q['Dd'] / Dd0:12.4f}      {q['Dr'] / Dr0:12.4f}        {pred:.5f}                  {q['rap']:.5f}        {pred - q['rap']:+.5f}")
