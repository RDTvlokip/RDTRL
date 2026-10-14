"""Tour 69 (vraie critique de dipankarsarkar, 15/10/2026) : ses ajustements zero = a + b lr^p sur mes zeros du tour 68.

Zeros (1e-7, D_d - 10,31e-9, interpolation lineaire sur la grille de pas 1e-8 en eps/lr, verifier_tour67_zero_corrige.py) :
lr 0,005 / 0,01 / 0,02 / 0,05 / 0,1 = 6,220 / 6,229 / 6,229 / 6,221 / 6,191 (arrondis a 1e-3).
Il ajuste lr 0,01 a 0,1 (quatre points) avec p = 1, 2, 3 (deux parametres a, b, residu max) puis predit lr 0,2 ; le "controle
sans ajustement" : (zero(0,02) - zero(0,05)) / (zero(0,05) - zero(0,1)). Je refais tout, avec (i) mes valeurs arrondies, (ii) un
ajustement a p libre (trois parametres pour quatre points), (iii) la sensibilite a l'incertitude d'arrondi (+-0,0005),
(iv) les zeros affines si le fichier d'entree optionnel est donne : python verifier_tour69_ajustement_zero.py [lr=zero ...].
"""

import sys
import itertools

import numpy as np
from scipy.optimize import curve_fit

ZEROS = {0.005: 6.220, 0.01: 6.229, 0.02: 6.229, 0.05: 6.221, 0.1: 6.191}


def ajuste(lrs, z, p):
    A = np.vstack([np.ones_like(lrs), lrs ** p]).T
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    res = z - A @ coef
    return coef, res


if __name__ == "__main__":
    zs = dict(ZEROS)
    for a in sys.argv[1:]:
        k, v = a.split("=")
        zs[float(k)] = float(v)
    lr4 = np.array([0.01, 0.02, 0.05, 0.1]); z4 = np.array([zs[l] for l in lr4])
    print("zeros utilises (1e-7) :", {k: zs[k] for k in sorted(zs)})
    print("\nlr 0,01 a 0,1, zero = a + b lr^p")
    print(" p    a        b          residu max   residus (0,01 ; 0,02 ; 0,05 ; 0,1)         zero a lr 0,2   residu a lr 0,005")
    for p in (1, 2, 3):
        (a, b), res = ajuste(lr4, z4, p)
        print(f" {p}   {a:.4f}  {b:+9.3f}    {np.abs(res).max():.4f}      {np.round(res, 4)}     {a + b * 0.2 ** p:.3f}          {zs[0.005] - (a + b * 0.005 ** p):+.4f}")
    # p libre
    try:
        f = lambda lr, a, b, p: a + b * lr ** p
        (a, b, p), _ = curve_fit(f, lr4, z4, p0=[6.23, -4.0, 2.0], maxfev=100000)
        print(f"\np libre (3 parametres, 4 points) : a = {a:.4f}  b = {b:+.3f}  p = {p:.3f}   zero a lr 0,2 = {a + b * 0.2 ** p:.3f}")
    except Exception as e:
        print("p libre : echec", e)
    r = (zs[0.02] - zs[0.05]) / (zs[0.05] - zs[0.1])
    print(f"\ncontrole sans ajustement : (z(0,02) - z(0,05)) / (z(0,05) - z(0,1)) = {zs[0.02] - zs[0.05]:.3f} / {zs[0.05] - zs[0.1]:.3f} = {r:.3f}  (lineaire : {(0.05 - 0.02) / (0.1 - 0.05):.3f}, p = 2 : {(0.05**2 - 0.02**2) / (0.1**2 - 0.05**2):.3f}, p = 3 : {(0.05**3 - 0.02**3) / (0.1**3 - 0.05**3):.3f})")
    # sensibilite aux arrondis : tire +-0,0005 sur chaque zero, regarde quel p ressort et le zero a lr 0,2
    rng = np.random.default_rng(3)
    out = []
    for _ in range(4000):
        zz = z4 + rng.uniform(-0.0005, 0.0005, 4)
        best = min((np.abs(ajuste(lr4, zz, p)[1]).max(), p) for p in (1, 2, 3))
        (a, b), _ = ajuste(lr4, zz, 2)
        out.append((best[1], a + b * 0.04))
    out = np.array(out)
    print(f"\narrondi +-0,0005 sur chaque zero, 4000 tirages : p retenu (plus petit residu max) : " + ", ".join(f"p = {p} : {np.mean(out[:, 0] == p):.0%}" for p in (1, 2, 3)) +
          f" ; zero a lr 0,2 avec p = 2 : {np.percentile(out[:, 1], 5):.3f} a {np.percentile(out[:, 1], 95):.3f}")
    # cinq points (0,005 a 0,1) puis comparaison a lr 0,2 mesure (argument 0.2=...)
    lr5 = np.array([l for l in sorted(zs) if l <= 0.1]); z5 = np.array([zs[l] for l in lr5])
    print(f"\najustement sur {len(lr5)} points (lr {lr5.min()} a {lr5.max()}) :")
    for p in (1, 2, 3):
        (a, b), res = ajuste(lr5, z5, p)
        pred = a + b * 0.2 ** p
        mes = f", mesure {zs[0.2]:.4f} (ecart {zs[0.2] - pred:+.4f})" if 0.2 in zs else ""
        print(f"   p = {p} : a = {a:.4f} b = {b:+.3f} residus {np.round(res, 4)} residu max {np.abs(res).max():.4f} ; predit a lr 0,2 : {pred:.4f}{mes}")
    # tous les sous-ensembles de trois lr, p = 2 libre non, juste le test : p = 2 predit-il le quatrieme ?
    print("\nvalidation croisee, p = 2 : on retire un lr (parmi 0,01 ; 0,02 ; 0,05 ; 0,1 ; 0,005) et on le predit avec les autres :")
    toutes = sorted(zs)
    for out_lr in toutes:
        lrs = np.array([l for l in toutes if l != out_lr and l >= 0.01]); zz = np.array([zs[l] for l in lrs])
        if len(lrs) < 3:
            continue
        (a, b), _ = ajuste(lrs, zz, 2)
        print(f"   sans {out_lr:5.3f} : predit {a + b * out_lr ** 2:.4f}, mesure {zs[out_lr]:.4f}, ecart {zs[out_lr] - (a + b * out_lr ** 2):+.4f}")
