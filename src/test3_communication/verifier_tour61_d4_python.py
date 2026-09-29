"""Tour 61 (29/09/2026) : E61-1, troisieme coordonnee independante pour la
mediane de l'etat calme : d4 = 1 - s4, issu du logit e4 - o4 de la ligne 4 de
l'emetteur. Si la mediane est le noeud d'un delta decale, delta'_d4 = delta'_d3.

Pourquoi pas la reduction numba : au noeud d4 = 26 exp(-X4)/(1 + 26 exp(-X4)) avec
X4 = (1+delta) r4/beta ~ 41,2, soit d4 ~ 3e-17, SOUS la resolution de la double
precision (1,1e-16) : la reconstruction d4 = 1 - rb/r4 est un bruit de soustraction
(premiere tentative, ratios absurdes de 1e6 a 4e7, non retenue). reduction_deux_lignes_tour59
(red2.py, pur Python) calcule d4 directement (C exp(-z4)/(1 + C exp(-z4))) dans sa trace.

Protocole : etat chaud 40 000 pas du reseau complet (eps 1e-10, mur 23), puis
20 000 + k pas a pli-1e-6 (k aleatoire), 10 000 pas de mise en regime au delta voulu,
NREC pas traces. Phases echappees ecartees. Plus lent que numba : NPH phases, NREC petit.
"""

import sys
import numpy as np
from mpmath import mp, mpf

sys.path.insert(0, ".")
from reduction_deux_lignes_tour59 import charger, simuler
from verifier_tour61_decomposition_d3 import PLI, d3_noeud, r4_noeud, d4_noeud, pente_fn

mp.dps = 40
CHAUD = "D:/tmp/rdtrl_tour59_chaud_mur23_3_4_eps1e-10_20000_20000.pt"


if __name__ == "__main__":
    NPH, NREC = 16, 300000
    offs = ["-1e-9", "-1e-10", "-1e-12"]
    s_chaud = charger(CHAUD)
    rng = np.random.default_rng(3)
    print(f"{NPH} phases x {NREC} pas traces (pur Python) ; delta' = (mediane - noeud)/pente du noeud, par observable")
    print("offset     d4 noeud     delta'_med(d3)   delta'_med(r4)   delta'_med(d4)   rapport r4/d3   rapport d4/d3   phases echappees")
    for off in offs:
        delta = float(PLI + mpf(off))
        dn3, rn, dn4 = float(d3_noeud(PLI + mpf(off))), float(r4_noeud(PLI + mpf(off))), float(d4_noeud(PLI + mpf(off)))
        sl3, sl4r, sl4 = pente_fn(d3_noeud, PLI + mpf(off)), pente_fn(r4_noeud, PLI + mpf(off)), pente_fn(d4_noeud, PLI + mpf(off))
        acc, esc = [], 0
        for _ in range(NPH):
            k = int(rng.integers(0, 4444))
            _, _, fin = simuler(s_chaud, float(PLI - mpf("1e-6")), 1e-10, 20000 + k, trace=False, seuil_stop=False)
            _, _, fin = simuler(fin, delta, 1e-10, 10000, trace=False, seuil_stop=False)
            out, b, fin = simuler(fin, delta, 1e-10, NREC, trace=True, seuil_stop=False)
            d3, r4, d4 = out[:, 0], out[:, 1], out[:, 4]
            if b >= 0 or d3.max() > 0.5:  # echappee (R_b > 0,9 pendant la trace, ou d3 effondre pendant la mise en regime)
                esc += 1
                continue
            acc.append(((np.median(d3) - dn3) / sl3, (np.median(r4) - rn) / sl4r, (np.median(d4) - dn4) / sl4,
                        np.median(d4), d4.max() / d4.min()))
        if not acc:
            print(f"{off:8s} toutes echappees")
            continue
        a = np.array(acc).mean(axis=0)
        print(f"{off:8s}  {dn4:.3e}   {a[0]:+.3e}      {a[1]:+.3e}      {a[2]:+.3e}     {a[1] / a[0]:+7.3f}        {a[2] / a[0]:+7.3f}         {esc}"
              f"    mediane d4 = {a[3]:.3e} (max/min sur une trace : {a[4]:.4f}) ; d4 mediane / d4 noeud = {a[3] / dn4:.3e}", flush=True)
