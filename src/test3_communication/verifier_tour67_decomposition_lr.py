"""Tour 67 : D_d(lr, q) = a(q) lr^p + D0(q) ? (q = eps/lr, x = 1e-8, D_d contre le noeud au delta mesure ;
lignes recopiees des sorties du 14/10/2026 de verifier_tour67_zero_dd.py). dipankarsarkar : D_d = lr^2 G(q)/lambda,
donc p = 2 et D0 = 0 ; mes lignes donnent un zero de D_d qui descend a bas lr (6,20 ; 6,10 ; 5,76 ; <4,6e-7 pour
lr 0,05 ; 0,02 ; 0,01 ; 0,005), donc D0 != 0 (terme qui ne va pas comme lr^p).
Ajuste, pour chaque q commun aux quatre lr (0,01 ; 0,02 ; 0,05 ; 0,1), D = a lr^p + D0 (3 parametres, 4 points) ;
puis ajuste p = 2 impose (a, D0) ; puis predit lr 0,005 (hors echantillon).
"""

import numpy as np
from scipy.optimize import curve_fit

Q = [5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5]  # en 1e-7
D = {  # lr -> D_d (noeud) aux 7 valeurs de q
    0.01: [3.3451e-9, 5.5795e-9, 7.7489e-9, 9.8140e-9, 1.1525e-8, 1.3568e-8, 1.5640e-8],
    0.02: [-1.8565e-8, -9.3236e-9, -3.4979e-10, 8.2468e-9, 1.5310e-8, 2.3824e-8, 3.2402e-8],
    0.05: [-1.7035e-7, -1.1210e-7, -5.4633e-8, 5.3655e-10, 4.7788e-8, 1.0487e-7, 1.6320e-7],
    0.1: [-6.2778e-7, -4.0806e-7, -1.8163e-7, 2.8626e-8, 2.5997e-7, 5.1214e-7, 7.8863e-7],
}
LR = np.array(sorted(D))


def f_libre(lr, a, p, d0):
    return a * lr ** p + d0


def f_p2(lr, a, d0):
    return a * lr ** 2 + d0


if __name__ == "__main__":
    print("q(1e-7)   p libre    a              D0 (libre)     |  p = 2 : a             D0            rms resid (libre | p=2)   D0/|D(0,01)|")
    res = []
    for i, q in enumerate(Q):
        y = np.array([D[l][i] for l in LR])
        # echelle : on ajuste D / 1e-8 pour le conditionnement
        try:
            pl, _ = curve_fit(f_libre, LR, y / 1e-8, p0=[y[-1] / 1e-8 / LR[-1] ** 2, 2.0, 0.0], maxfev=20000)
        except Exception as e:
            pl = [np.nan] * 3
        p2, _ = curve_fit(f_p2, LR, y / 1e-8, p0=[y[-1] / 1e-8 / LR[-1] ** 2, 0.0])
        rl = np.sqrt(np.mean((f_libre(LR, *pl) - y / 1e-8) ** 2)) * 1e-8
        r2 = np.sqrt(np.mean((f_p2(LR, *p2) - y / 1e-8) ** 2)) * 1e-8
        res.append((q, pl, p2))
        print(f"{q:5.1f}    {pl[1]:6.3f}   {pl[0] * 1e-8:+.3e}   {pl[2] * 1e-8:+.3e}  |  {p2[0] * 1e-8:+.3e}   {p2[1] * 1e-8:+.3e}      {rl:.1e} | {r2:.1e}          {p2[1] * 1e-8 / abs(y[0]):+.2f}")
    print("\nprediction hors echantillon, lr 0,005 (D = a lr^2 + D0, ajustement p = 2 sur lr 0,01-0,1) :")
    for q, pl, p2 in res:
        print(f"   q = {q:.1f}e-7 : D(0,005) predit = {f_p2(0.005, *p2) * 1e-8:+.3e}   (libre : {f_libre(0.005, *pl) * 1e-8:+.3e})")
