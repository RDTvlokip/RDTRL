"""Tour 68 : etat regle de la reduction sans salve, depuis l'etat de base BRUT (pas de chauffe), comme dans sa verification.

Usage : python verifier_tour68_etat_regle.py <module> <eps> <x> <pas> [<pas_mesure_derive>]
Sort : d3 final, g_l3 et g_l4 (convention J) et m[4], m[5] a la fin, derive de l3 + l4 par pas sur les derniers pas_mesure_derive pas
(2 lr g_l/eps attendu), et la racine g_l3 = g_l4 (z4 a la valeur de base) pour comparer.
"""

import importlib
import json
import sys
import math

import numpy as np
from mpmath import mp, mpf

sys.path.insert(0, ".")
from verifier_tour61_decomposition_d3 import PLI
from verifier_tour68_point_fixe_gl3_gl4 import racine, d3_de
from verifier_tour68_somme_gradient_queue import g_reduction

mp.dps = 40

if __name__ == "__main__":
    mod, eps, x, pas = sys.argv[1], float(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    der = int(sys.argv[5]) if len(sys.argv) > 5 else 100000
    redlib = importlib.import_module(mod)
    d = json.load(open("etat_base_reduction_mur23.json", encoding="utf-8"))
    s = dict(x=np.array(d["x"]), m=np.array(d["m"]), v=np.array(d["v"]), Se=d["Se"], Sed=d["Sed"], t=d["t"])
    delta = float(PLI - mpf(x))
    redlib.avancer_obs(s, delta, pas - der, np.zeros((1, 7)), eps=eps)
    sum0 = s["x"][4] + s["x"][5]
    first, maxrb, rb, done = redlib.avancer_obs(s, delta, der, np.zeros((1, 7)), eps=eps)
    sum1 = s["x"][4] + s["x"][5]
    z3 = s["x"][0] - s["x"][1]
    ex = 26.0 * math.exp(-z3); d3 = ex / (1.0 + ex)
    g4, g5 = g_reduction(s, delta)
    z4b = mpf(d["x"][2]) - mpf(d["x"][3])
    u2 = racine(PLI - mpf(x), mpf(d["Se"]), mpf(d["Sed"]), "egal", z4b)
    d3r, _ = d3_de(u2, PLI - mpf(x), mpf(d["Se"]))
    print(f"{mod} eps {eps:.0e} x {x} pas {pas} : maxRb {maxrb:.3g}")
    print(f"   d3 final {d3:.10e}   racine g_l3=g_l4 (z4 base) {float(d3r):.10e}   diff {d3 - float(d3r):+.2e}")
    print(f"   g_l3 {g4:+.5e}  g_l4 {g5:+.5e}  m4 {s['m'][4]:+.5e}  m5 {s['m'][5]:+.5e}  z4 {s['x'][2] - s['x'][3]:.4f}")
    print(f"   derive de l3 + l4 par pas : {(sum1 - sum0) / der:+.5e}   attendu 2 lr g/eps = {2 * redlib.LR * g4 / eps:+.5e}")
