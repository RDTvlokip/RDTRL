"""Tour 71 : que se passe-t-il pendant la chauffe (pli - 1e-6, 20 000 pas depuis l'etat de base) selon lr ? rb = (1 - d4) r4 depasse-t-il 0,9
de facon transitoire (un depassement qui revient) ou durable (effondrement) ?

Usage : python verifier_tour71_trace_chauffe.py <module> <eps/lr> [<pas>]
Sort : premier pas ou rb > 0,9, rb max, rb au dernier pas, nombre de pas ou rb > 0,9, et rb a 100, 1000, 10000 pas.
"""

import importlib
import json
import os
import sys

import numpy as np
from mpmath import mp, mpf

sys.path.insert(0, ".")
from verifier_tour61_decomposition_d3 import PLI

mp.dps = 40

if __name__ == "__main__":
    mod, q = sys.argv[1], float(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    redlib = importlib.import_module(mod)
    lr = redlib.LR
    eps = q * lr
    d = json.load(open("etat_base_reduction_mur23.json", encoding="utf-8"))
    s = dict(x=np.array(d["x"]), m=np.array(d["m"]), v=np.array(d["v"]), Se=d["Se"], Sed=d["Sed"], t=d["t"])
    if os.environ.get("TOUR71_ECHELLE"):
        # etat de base fait a lr 0,05 ; sqrt(v) et m vont comme lr a eps/lr fixe : on les met a l'echelle de lr avant de partir
        c = lr / 0.05
        s["m"] = s["m"] * c; s["v"] = s["v"] * c * c
    rec = np.zeros((n, 7))
    first, maxrb, rb, done = redlib.avancer(s, float(PLI - mpf("1e-6")), n, eps=eps, cross=2.0, stop=False, rec=rec)
    r = rec[:, 0]
    sup = np.nonzero(r > 0.9)[0]
    print(f"{mod} lr {lr} eps/lr {q:.2e} : {n} pas a pli-1e-6")
    print(f"   rb initial {r[0]:.5f}  max {r.max():.5f}  final {r[-1]:.5f}  premier rb > 0,9 : {sup[0] if len(sup) else 'jamais'}  nb de pas > 0,9 : {len(sup)}  dernier : {sup[-1] if len(sup) else '-'}")
    print("   rb a t = 0,1,2,3,10,100,1000,10000 :", [round(float(r[t]), 4) for t in (0, 1, 2, 3, 10, 100, 1000, min(10000, n - 1))])
