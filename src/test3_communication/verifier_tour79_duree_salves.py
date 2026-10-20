"""Tour 79 : durees, amplitudes et remplissages des salves en fonction de lr, a eps/lr fixe (moments de l'etat de base mis a l'echelle de lr/0,05).

Hypothese H79-4 : sqrt(v_e3) n'est pas exactement proportionnel a lr (exposant 0,986 a x = 1e-6) parce que la dynamique non lineaire des salves (amplitude de saturation de u proportionnelle a lr,
temps de croissance en ln(A_sat/A_0)) n'est pas invariante d'echelle, alors que le franchissement de S_c = 38 l'est.

Usage : python verifier_tour79_duree_salves.py <module> <eps> <x> [<nrec>]     (TOUR78_ECHELLE=1 pour les moments a l'echelle)
Pour chaque salve (groupe de pas avec E = max(|D2u|/4) > 3e-4, coupe a 30 pas) : duree, amplitude maximale de E, rapport de sqrt(v) 40 pas apres la fin sur sqrt(v) au debut pour l3 et e3.
"""

import importlib
import os
import sys

import numpy as np
from mpmath import mpf

sys.path.insert(0, ".")
from verifier_tour75_salves_depart import etat_base, PLI


def main(nom, eps, xs, nrec):
    redlib = importlib.import_module(nom)
    lr = redlib.LR
    S1 = etat_base()
    delta = float(PLI - mpf(xs))
    ks = [int(a) for a in np.random.default_rng(7).integers(0, 4444, 16)]
    rng = np.random.default_rng(11)
    w = redlib.copie(S1)
    if os.environ.get("TOUR78_ECHELLE"):
        w["m"] = w["m"] * (lr / 0.05)
        w["v"] = w["v"] * (lr / 0.05) ** 2
    redlib.avancer(w, float(PLI - mpf("1e-6")), 20000 + ks[0], eps=eps, cross=2.0, stop=False)
    w["x"] = w["x"] * (1.0 + 1e-9 * rng.standard_normal(6))
    redlib.avancer_obs(w, delta, 10000, np.zeros((1, 7)), eps=eps)
    rec = np.zeros((nrec, 7))
    first, maxrb, rb, done = redlib.avancer_obs(w, delta, nrec, rec, eps=eps)
    if maxrb > 0.9:
        print(f"{nom} lr = {lr} : effondre")
        return
    u = rec[:, 2]; s_e3 = rec[:, 3]; s_l3 = rec[:, 4]
    D2 = u[2:] - 2 * u[1:-1] + u[:-2]
    A = np.abs(D2) / 4.0
    E = np.maximum(A[:-1], A[1:])
    idx = np.nonzero(E > 3e-4)[0]
    coupe = np.nonzero(np.diff(idx) > 30)[0]
    debuts = np.concatenate([[0], coupe + 1]); fins = np.concatenate([coupe, [len(idx) - 1]])
    dur, amp, rl, re_, per = [], [], [], [], []
    g_prec = None
    for g, (a, b) in enumerate(zip(debuts, fins)):
        ia, ib = idx[a], idx[b]
        if ib + 41 >= len(E) or ia < 5:
            continue
        dur.append(ib - ia + 1)
        amp.append(E[ia:ib + 1].max())
        rl.append(s_l3[ib + 40] / s_l3[ia - 1])
        re_.append(s_e3[ib + 40] / s_e3[ia - 1])
        if g_prec is not None:
            per.append(ia - g_prec)
        g_prec = ia
    print(f"{nom} lr = {lr} eps = {eps:.1e} x = {xs} : {len(dur)} salves ; duree mediane {np.median(dur):.0f} pas ; amplitude maximale mediane {np.median(amp):.4e} ; periode mediane {np.median(per):.0f} pas")
    print(f"   remplissage (sqrt v apres/avant) : l3 {np.median(rl):.4f} ; e3 {np.median(re_):.4f} ; rapport e3/l3 des remplissages {np.median(re_) / np.median(rl):.4f}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 400000)
