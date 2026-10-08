"""Tour 64 (08/10/2026, vraie critique de dipankarsarkar) : l'inclinaison L = (D_r/D_d)/rho et le
deplacement D_d sont-ils des fonctions de r = eps/sqrt(v3) seul (sqrt(v3) : echelle d'Adam de la ligne 3),
ou aussi du gain lr/sqrt(v3) ? Il demande le rapport de la mediane de sqrt(v) de la ligne 3 entre
(lr 0,05, eps 1e-9) et (lr 0,02, eps 1e-10) ; "r seul" exige plus de 10 (borne tiree de l'interpolation
de L, voir verifier_tour64_cartographie_lr.py).

Usage : python verifier_tour64_gain_et_r.py <lr:eps> [<lr:eps> ...]   (lr dans {0.02, 0.05})
Reduction numba (lr 0,05 : reduction_numba_tour59 ; lr 0,02 : reduction_numba_tour63_lr002), x = 1e-13,
16 phases x 2 000 000 pas apres 20 000 + k pas de chauffe a pli-1e-6 et 10 000 pas au delta voulu.
Par run : sd(d3) sature, D_d, D_r, L, mediane et moyenne de sqrt(v) pour e3 (col. 3 de la trace) et l3
(col. 4), r_med = eps/mediane sqrt(v_e3), gain = lr/mediane sqrt(v_e3), et sqrt(v_o3)/sqrt(v_e3) final.
"""

import importlib
import sys
import numpy as np
from mpmath import mp, mpf

sys.path.insert(0, ".")
from hasard_reduction_tour59 import base_state
from verifier_tour61_decomposition_d3 import PLI, d3_noeud, r4_noeud, pente_fn

mp.dps = 60
OFF = "-1e-13"

if __name__ == "__main__":
    ks = [int(a) for a in np.random.default_rng(7).integers(0, 4444, 16)]
    s1 = base_state()
    d3_pli = float(d3_noeud(PLI - mpf("1e-30"))); r4_pli = float(r4_noeud(PLI - mpf("1e-30")))
    delta = float(PLI + mpf(OFF))
    sl3, sl4 = pente_fn(d3_noeud, PLI + mpf(OFF)), pente_fn(r4_noeud, PLI + mpf(OFF))
    rho = sl4 / sl3
    print("lr     eps      sd_d3        |D_d|        L         med sqrt(v_e3)   moy sqrt(v_e3)   med sqrt(v_l3)   r_med=eps/sqrt(v)  gain=lr/sqrt(v)   vo3/ve3   ech.")
    for arg in sys.argv[1:]:
        lr, eps = arg.split(":")
        redlib = importlib.import_module("reduction_numba_tour59" if lr == "0.05" else "reduction_numba_tour63_lr002")
        eps = float(eps)
        rng = np.random.default_rng(11)
        sd, Dd, Dr, vm, vmean, vl, ratio_o, esc = [], [], [], [], [], [], [], 0
        for k in ks:
            w = redlib.copie(s1)
            redlib.avancer(w, float(PLI - mpf("1e-6")), 20000 + k, eps=eps, cross=2.0, stop=False)
            w["x"] = w["x"] * (1.0 + 1e-9 * rng.standard_normal(6))
            redlib.avancer_obs(w, delta, 10000, np.zeros((1, 7)), eps=eps)
            rec = np.zeros((2_000_000, 7))
            first, maxrb, rb, done = redlib.avancer_obs(w, delta, 2_000_000, rec, eps=eps)
            if maxrb > 0.9:
                esc += 1
                continue
            ex = 26.0 * np.exp(-rec[:, 1]); d3 = ex / (1.0 + ex)
            r4 = 1.0 / (1.0 + np.exp(rec[:, 2]) + w["Se"])
            sd.append(d3.std()); Dd.append(np.median(d3) - d3_pli); Dr.append(np.median(r4) - r4_pli)
            vm.append(np.median(rec[:, 3])); vmean.append(rec[:, 3].mean()); vl.append(np.median(rec[:, 4]))
            ratio_o.append(np.sqrt(w["v"][1]) / np.sqrt(w["v"][0]))
        sd, Dd, Dr, vm, vmean, vl = map(np.array, (sd, Dd, Dr, vm, vmean, vl))
        L = (Dr.mean() / Dd.mean()) / rho
        print(f"{lr:5s} {eps:7.0e}  {sd.mean():.4e}  {abs(Dd.mean()):.4e}  {L:.5f}   {vm.mean():.4e}      {vmean.mean():.4e}      {vl.mean():.4e}      "
              f"{eps / vm.mean():.4e}        {float(lr) / vm.mean():.4e}     {np.mean(ratio_o):.4f}   {esc}", flush=True)
