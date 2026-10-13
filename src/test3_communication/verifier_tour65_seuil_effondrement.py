"""Tour 65 (vraie critique de dipankarsarkar) : l'eps d'effondrement de la reduction a x = pli - delta fixe
est-il au meme eps/lr pour lr 0,05 et lr 0,02 (symetrie d'echelle d'Adam) ?

Usage : python verifier_tour65_seuil_effondrement.py <lr> <x> <eps> [<eps> ...]    (x en valeur absolue, ex. 1e-13)
Pour chaque eps : 16 phases (memes ks que le tour 64), chauffe 20 000 + k pas a pli-1e-6, 10 000 pas au delta
voulu, puis 2 000 000 pas ; une phase est "effondree" si r4 > 0,9 a un moment (maxrb). On separe : effondree
pendant la chauffe+mise en place (avant la mesure) ou pendant les 2e6 pas. Sortie : nombre de phases effondrees.
"""

import importlib
import sys
import numpy as np
from mpmath import mp, mpf

sys.path.insert(0, ".")
from hasard_reduction_tour59 import base_state
from verifier_tour61_decomposition_d3 import PLI

mp.dps = 60

if __name__ == "__main__":
    lr = sys.argv[1]
    x = sys.argv[2]
    ks = [int(a) for a in np.random.default_rng(7).integers(0, 4444, 16)]
    s1 = base_state()
    delta = float(PLI - mpf(x))
    redlib = importlib.import_module("reduction_numba_tour59" if lr == "0.05" else "reduction_numba_tour63_lr002")
    for eps in map(float, sys.argv[3:]):
        rng = np.random.default_rng(11)
        avant = apres = chauffe = 0
        premiers = []
        for k in ks:
            w = redlib.copie(s1)
            _, mxw, _, _ = redlib.avancer(w, float(PLI - mpf("1e-6")), 20000 + k, eps=eps, cross=2.0, stop=False)
            w["x"] = w["x"] * (1.0 + 1e-9 * rng.standard_normal(6))
            _, mx0, _, _ = redlib.avancer_obs(w, delta, 10000, np.zeros((1, 7)), eps=eps)
            if mxw > 0.9:
                chauffe += 1
            if mx0 > 0.9:
                avant += 1
                continue
            rec = np.zeros((2_000_000, 7))
            first, maxrb, rb, done = redlib.avancer_obs(w, delta, 2_000_000, rec, eps=eps)
            if maxrb > 0.9:
                apres += 1
                premiers.append(first)
        print(f"lr {lr} x {x} eps {eps:9.3e}  eps/lr {eps / float(lr):9.3e}  effondrees avant mesure {avant:2d} (dont deja pendant la chauffe a pli-1e-6 : {chauffe:2d}) / pendant {apres:2d}  / 16 (total {avant + apres:2d})"
              f"  premier pas median {np.median(premiers) if premiers else float('nan'):.0f}", flush=True)
