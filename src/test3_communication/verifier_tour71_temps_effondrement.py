"""Tour 71 (vraie critique de dipankarsarkar, 17/10/2026) : le temps d'effondrement diverge-t-il en 1 / sqrt(lr - lr_c) quand lr
descend vers le plafond (entre 0,3 et 0,35 a eps/lr = 6e-7) ?

Usage : python verifier_tour71_temps_effondrement.py <module> <x> <eps/lr> <tmax> [<phases>]
Protocole des tours 64-70 : etat de base JSON, chauffe 20 000 + k pas a pli - 1e-6 (meme eps), perturbation 1e-9, puis on bascule a
delta = pli - x ; on court jusqu'a rb > 0,9 (premier pas t_c compte depuis la bascule) ou tmax pas. Sortie : par phase le temps
d'effondrement (ou > tmax), et le resume : nombre effondre, mediane et moyenne des temps observes, ecart-type / moyenne (CV),
rapport 2e plus petit / plus petit, et, si au moins 4 temps, le test de Lilliefors (exponentielle) approxime : KS contre
l'exponentielle de meme moyenne (valeur critique 0,25 a 5 % pour n = 16, tabulee Lilliefors).
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


def etat_base():
    d = json.load(open("etat_base_reduction_mur23.json", encoding="utf-8"))
    return dict(x=np.array(d["x"]), m=np.array(d["m"]), v=np.array(d["v"]), Se=d["Se"], Sed=d["Sed"], t=d["t"])


if __name__ == "__main__":
    mod, x, q, tmax = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(float(sys.argv[4]))
    nph = int(sys.argv[5]) if len(sys.argv) > 5 else 16
    redlib = importlib.import_module(mod)
    lr = redlib.LR
    eps = q * lr
    s1 = etat_base()
    ks = [int(a) for a in np.random.default_rng(7).integers(0, 4444, 16)][:nph]
    delta = float(PLI - mpf(x))
    rng = np.random.default_rng(11)
    temps, avant = [], 0
    for k in ks:
        w = redlib.copie(s1)
        if os.environ.get("TOUR71_SANS_CHAUFFE"):
            # on part de l'etat de base brut et on mesure le temps d'effondrement a delta directement (x peut valoir 1e-6)
            mxw = 0.0
        else:
            _, mxw, _, _ = redlib.avancer(w, float(PLI - mpf("1e-6")), 20000 + k, eps=eps, cross=2.0, stop=False)
        w["x"] = w["x"] * (1.0 + 1e-9 * rng.standard_normal(6))
        if mxw > 0.9:
            avant += 1
            temps.append(-2)
            continue
        first, maxrb, rb, done = redlib.avancer_obs(w, delta, tmax, np.zeros((1, 7)), eps=eps, cross=0.9, stop=True)
        temps.append(first if first >= 0 else -1)
    t = np.array([a for a in temps if a >= 0], float)
    nc = len([a for a in temps if a >= 0 or a == -2])
    print(f"{mod} (lr {lr}) x {x} eps/lr {q:.2e} tmax {tmax:.1e} : effondrees {nc}/{len(ks)} (pendant la chauffe : {avant})")
    print("   temps par phase :", [("chauffe" if a == -2 else ("> tmax" if a < 0 else a)) for a in temps])
    if len(t) >= 2:
        cv = t.std(ddof=1) / t.mean()
        ts = np.sort(t)
        ks_stat = np.max(np.abs(np.arange(1, len(ts) + 1) / len(ts) - (1 - np.exp(-ts / ts.mean()))))
        print(f"   mediane {np.median(t):.0f}  moyenne {t.mean():.0f}  CV {cv:.2f}  min {ts[0]:.0f}  max {ts[-1]:.0f}  KS/exponentielle {ks_stat:.3f}")
