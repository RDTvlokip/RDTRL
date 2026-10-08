"""Tour 63 (08/10/2026) : pourquoi les 16 phases de la reduction a eps 1e-6 (offsets -1e-11, -1e-13)
sont-elles toutes "echappees" (R_b > 0,9) ? Une phase, suivie pas a pas : etat apres chauffe sous eps,
puis R_b, d3, u, sqrt(v) a quelques pas, et comparaison avec l'etat de chauffe du RESEAU COMPLET a eps 1e-6
(cache D:/tmp/rdtrl_tour59_chaud_mur23_3_4_eps1e-06_20000_20000.pt : d3 = 2,6159e-3 a pli-1e-6).
"""

import sys
import numpy as np
from mpmath import mp, mpf

sys.path.insert(0, ".")
import reduction_numba_tour59 as redlib
from hasard_reduction_tour59 import base_state
from verifier_tour61_decomposition_d3 import PLI, d3_noeud

mp.dps = 60

if __name__ == "__main__":
    eps = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-6
    off = sys.argv[2] if len(sys.argv) > 2 else "-1e-11"
    s1 = base_state()

    def d3_de(w):
        z3 = w["x"][0] - w["x"][1]
        ex = 26.0 * np.exp(-z3)
        return ex / (1.0 + ex)

    print(f"eps = {eps:g} ; d3 de l'etat de phase 1 (eps 1e-10, 20 000 pas) = {d3_de(s1):.6e} ; noeud a pli-1e-6 = {float(d3_noeud(PLI - mpf('1e-6'))):.6e}")
    w = redlib.copie(s1)
    for etape, n in (("chauffe 5 000", 5000), ("chauffe 15 000", 15000), ("chauffe 20 000", 20000), ("chauffe 100 000", 100000)):
        r = redlib.avancer(w, float(PLI - mpf("1e-6")), n, eps=eps, cross=2.0, stop=False)
        print(f"   apres {etape:16s} (cumul eps={eps:g}) : d3 = {d3_de(w):.6e}   R_b dernier = {r[2]:.6f}   max R_b = {r[1]:.6f}")
    delta = float(PLI + mpf(off))
    print(f"\npuis delta = pli{off} (noeud d3 = {float(d3_noeud(PLI + mpf(off))):.6e})")
    for n in (1000, 10000, 100000, 1000000):
        r = redlib.avancer(w, delta, n, eps=eps, cross=2.0, stop=False)
        print(f"   +{n:8d} pas : d3 = {d3_de(w):.6e}   R_b = {r[2]:.6f}   max R_b = {r[1]:.6f}")
