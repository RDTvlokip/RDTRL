"""Tour 66 (vraie critique de dipankarsarkar) : son test propre, la symetrie du pli
(lr, eps, x, d) -> (c lr, c^2 eps, c^2 x, c d), c = 0,4 ; et mon mecanisme : les salves sont le bord de
stabilite d'Adam sur le mode raide (u = l3 - l4 a 99 %, z3 a 13 %), le pli est le mode mou (z3 a 99 %).

Usage : python verifier_tour66_paire_pli.py <lr:eps:x> [...]    (x = pli - delta, ex. 0.02:1.6e-9:1.6e-14)
Reduction numba, 16 phases (memes ks que les tours 64-65), chauffe 20 000 + k pas a pli-1e-6, 10 000 pas au
delta voulu, 2 000 000 pas enregistres. Par etat : sd(d3), sd(u), D_d et D_r contre le noeud au delta mesure,
tilt D_r/D_d, medianes de sqrt(v_e3) et sqrt(v_l3), leur rapport (a comparer a la composante z3/u du mode
raide, 0,1341), et le nombre de bord de stabilite S = lr * 2 |J_uu| / (mediane sqrt(v_l3) + eps), J_uu =
-1,1204e-4 (jacobien, verifier_tour66_forme_normale.py ; le bord d'Adam a b1 = 0,9 est 2(1+b1)/(1-b1) = 38).
"""

import importlib
import os
import sys
import numpy as np
from mpmath import mp, mpf

sys.path.insert(0, ".")
from hasard_reduction_tour59 import base_state
from verifier_tour61_decomposition_d3 import PLI, d3_noeud, r4_noeud

mp.dps = 60
J_UU = 1.1204e-4

if __name__ == "__main__":
    ks = [int(a) for a in np.random.default_rng(7).integers(0, 4444, 16)]
    s1 = base_state()
    print("lr    eps       x         sd_d3       sd_u        D_d(noeud)   D_r(noeud)   tilt     med sv_e3   med sv_l3   sv_e3/sv_l3   S      ech.")
    for arg in sys.argv[1:]:
        lr, eps, x = arg.split(":")
        redlib = importlib.import_module(os.environ.get("TOUR66_REDLIB") or ("reduction_numba_tour59" if lr == "0.05" else "reduction_numba_tour63_lr002"))
        if redlib.LR != float(lr):
            raise SystemExit(f"le module {redlib.__name__} a LR = {redlib.LR}, pas {lr}")
        epsf = float(eps)
        delta = float(PLI - mpf(x))
        d3n = float(d3_noeud(PLI - mpf(x))); r4n = float(r4_noeud(PLI - mpf(x)))
        rng = np.random.default_rng(11)
        sd, sdu, Dd, Dr, ve, vl, esc = [], [], [], [], [], [], 0
        for k in ks:
            w = redlib.copie(s1)
            redlib.avancer(w, float(PLI - mpf("1e-6")), 20000 + k, eps=epsf, cross=2.0, stop=False)
            w["x"] = w["x"] * (1.0 + 1e-9 * rng.standard_normal(6))
            redlib.avancer_obs(w, delta, 10000, np.zeros((1, 7)), eps=epsf)
            rec = np.zeros((2_000_000, 7))
            first, maxrb, rb, done = redlib.avancer_obs(w, delta, 2_000_000, rec, eps=epsf)
            if maxrb > 0.9:
                esc += 1
                continue
            ex = 26.0 * np.exp(-rec[:, 1]); d3 = ex / (1.0 + ex)
            r4 = 1.0 / (1.0 + np.exp(rec[:, 2]) + w["Se"])
            sd.append(d3.std()); sdu.append(rec[:, 2].std())
            Dd.append(np.median(d3) - d3n); Dr.append(np.median(r4) - r4n)
            ve.append(np.median(rec[:, 3])); vl.append(np.median(rec[:, 4]))
        if not sd:
            print(f"{lr:5s} {epsf:8.2e}  {x:8s}  16/16 effondrees", flush=True)
            continue
        sd, sdu, Dd, Dr, ve, vl = map(np.mean, (sd, sdu, Dd, Dr, ve, vl))
        S = float(lr) * 2 * J_UU / (vl + epsf)
        print(f"{lr:5s} {epsf:8.2e}  {x:8s}  {sd:.4e}  {sdu:.4e}  {Dd:+.4e}  {Dr:+.4e}  {Dr / Dd:7.4f}  {ve:.4e}  {vl:.4e}  {ve / vl:.4f}        {S:6.2f}  {esc}", flush=True)
