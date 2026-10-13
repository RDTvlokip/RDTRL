"""Tour 67 (vraie critique de dipankarsarkar, 14/10/2026) : verification de ce qu'il lit dans mon code.

(1) g[1] = -ge3/C : apres chauffe, m1 = -m0/26 et sqrt(v1) = sqrt(v0)/26 ? A quelle precision (il dit "machine
    precision") ? Mesure dans la reduction lr 0,05 sur 16 phases (memes ks que les tours 64-66), apres la chauffe
    de 20 000 + k pas a pli-1e-6 puis 2 000 000 pas au delta voulu.
(2) Le pas sur z3 = e3 - o3 vaut lr m0 [1/(s + eps) + 1/(s + 26 eps)], s = sqrt(v_e3) : identite verifiee sur un
    pas de la reduction (variation de z3 comparee a la formule, etat apres mise a jour de m et v).
(3) Echelle a eps/lr egal (lignes recopiees des sorties du 13/10/2026, x = 1e-8, lr 0,05 | lr 0,02 : verifier_tour64_gain_et_r.py
    et verifier_tour66_paire_pli.py) : sqrt(v_l3) doit valoir c = 2,5 fois plus, |D_d| c^2 = 6,25 selon lui.
"""

import json
import sys

import numpy as np
from mpmath import mp, mpf

sys.path.insert(0, ".")
import reduction_numba_tour59 as redlib
from verifier_tour61_decomposition_d3 import PLI

mp.dps = 40


def etat_base():
    d = json.load(open("etat_base_reduction_mur23.json", encoding="utf-8"))
    return dict(x=np.array(d["x"]), m=np.array(d["m"]), v=np.array(d["v"]), Se=d["Se"], Sed=d["Sed"], t=d["t"])


if __name__ == "__main__":
    s1 = etat_base()
    ks = [int(a) for a in np.random.default_rng(7).integers(0, 4444, 16)]
    eps = 5e-8
    delta = float(PLI - mpf("1e-8"))
    print("(1) m1/m0 et sqrt(v1)/sqrt(v0) contre -1/26 et 1/26 (lr 0,05, eps 5e-8, x = 1e-8)")
    print("    k      apres chauffe : |m1/m0 + 1/26| relatif, |sqrt(v1/v0)*26 - 1|   apres 2e6 pas : idem")
    for k in ks[:6]:
        w = redlib.copie(s1)
        redlib.avancer(w, float(PLI - mpf("1e-6")), 20000 + k, eps=eps, cross=2.0, stop=False)
        a = (abs(w["m"][1] / w["m"][0] * 26 + 1), abs(np.sqrt(w["v"][1] / w["v"][0]) * 26 - 1))
        redlib.avancer_obs(w, delta, 2_000_000, np.zeros((1, 7)), eps=eps)
        b = (abs(w["m"][1] / w["m"][0] * 26 + 1), abs(np.sqrt(w["v"][1] / w["v"][0]) * 26 - 1))
        print(f"    {k:4d}   {a[0]:.2e}  {a[1]:.2e}          {b[0]:.2e}  {b[1]:.2e}")
    print("    (l'etat de base du reseau complet a v1/v0 = 1/26^2 a la precision de la reduction du reseau : voir la ligne 1 du JSON)")
    d0 = etat_base()
    print(f"    etat de base : sqrt(v1/v0)*26 - 1 = {np.sqrt(d0['v'][1] / d0['v'][0]) * 26 - 1:.2e} ; m1/m0*26 + 1 = {d0['m'][1] / d0['m'][0] * 26 + 1:.2e}")

    print("\n(2) pas sur z3 = e3 - o3 : variation mesuree contre lr m0 [1/(s+eps) + 1/(s+26 eps)]")
    for epsx in (5e-8, 1e-10):
        w = redlib.copie(s1)
        redlib.avancer(w, float(PLI - mpf("1e-6")), 20000 + ks[0], eps=epsx, cross=2.0, stop=False)
        z0 = w["x"][0] - w["x"][1]
        redlib.avancer_obs(w, delta, 1, np.zeros((1, 7)), eps=epsx)
        dz = (w["x"][0] - w["x"][1]) - z0
        sq = np.sqrt(w["v"][0]); m0 = w["m"][0]
        f = redlib.LR * m0 * (1 / (sq + epsx) + 1 / (sq + 26 * epsx))
        print(f"    eps {epsx:.0e} : dz3 = {dz:.6e}, formule = {f:.6e}, ecart relatif {abs(dz / f - 1):.2e}")

    print("\n(3) echelle a eps/lr egal, x = 1e-8 : lr 0,05 | lr 0,02, rapports")
    lignes = [  # eps/lr, sqrt(v_l3) 0,05, 0,02, |D_d| noeud 0,05, 0,02, S 0,05, 0,02
        (1e-6, 2.7517e-7, 1.1017e-7, 2.0649e-6, 2.8112e-7, 34.46, None),
        (2e-6, 2.1791e-7, 8.7176e-8, 3.6178e-6, 4.7002e-7, 35.24, None),
        (3e-6, 1.6219e-7, 6.4785e-8, 2.4267e-6, 3.4747e-7, None, None),
    ]
    for q, v5, v2, d5, d2, _, _ in lignes:
        print(f"    eps/lr {q:.0e} : sqrt(v_l3) x{v5 / v2:.4f} (2,5 attendu) ; |D_d| x{d5 / d2:.3f} (6,25 attendu, ecart {d5 / d2 / 6.25 - 1:+.1%}) ; exposant ln(rapport D)/ln 2,5 = {np.log(d5 / d2) / np.log(2.5):.3f}")
