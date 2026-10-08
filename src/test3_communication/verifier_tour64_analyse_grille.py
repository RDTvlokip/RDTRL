"""Tour 64 : analyse de la grille (lr, eps) mesuree par verifier_tour64_gain_et_r.py (11 runs, reduction numba,
x = 1e-13, 16 phases, lignes recopiees des sorties du 08/10/2026 ; les deux lignes (0.05,1e-9) et
(0.02,1e-10) viennent du premier appel, les neuf autres du second).

Questions : (1) R = mediane sqrt(v_e3)(0.05, 1e-9)/(0.02, 1e-10) (sa question) ; (2) sqrt(v)/lr et gain
lr/sqrt(v) sur les 11 etats (constants ?) ; (3) L et |D_d| se rangent-ils sur r = eps/sqrt(v_e3) ? : pour
chaque etat lr 0,02, valeur interpolee (lineaire en log r) de la colonne lr 0,05 au meme r, et ecart ;
(4) memes tests pour d'autres organisateurs (sd, eps/lr).
"""

import math
import numpy as np

# lr, eps, sd, |D_d|, L, mediane sqrt(v_e3)
T = [
    (0.05, 1e-10, 2.1995e-5, 6.0837e-6, 0.96799, 4.7367e-8),
    (0.05, 1e-9, 1.7417e-5, 7.1539e-6, 0.96620, 4.6509e-8),
    (0.05, 3e-9, 1.2380e-5, 8.3244e-6, 0.95593, 4.5509e-8),
    (0.05, 1e-8, 6.3665e-6, 8.6313e-6, 0.94988, 4.3719e-8),
    (0.02, 1e-10, 6.0720e-6, 5.8490e-6, 0.96540, 1.9090e-8),
    (0.02, 3e-10, 5.1226e-6, 5.9383e-6, 0.96418, 1.8856e-8),
    (0.02, 1e-9, 3.4520e-6, 5.7817e-6, 0.96219, 1.8398e-8),
    (0.02, 3e-9, 2.0445e-6, 4.8066e-6, 0.96714, 1.7774e-8),
    (0.02, 1e-8, 1.0577e-6, 1.8958e-6, 0.98660, 1.6416e-8),
]


def interp(xv, xs, ys):
    """lineaire en log x ; None hors plage"""
    xs = np.array(xs); ys = np.array(ys)
    if xv < xs.min() or xv > xs.max():
        return None
    return float(np.interp(math.log(xv), np.log(xs), ys))


if __name__ == "__main__":
    v05 = [t for t in T if t[0] == 0.05]; v02 = [t for t in T if t[0] == 0.02]
    R = [t for t in v05 if t[1] == 1e-9][0][5] / [t for t in v02 if t[1] == 1e-10][0][5]
    print(f"(1) R = sqrt(v_e3)(0.05, 1e-9)/sqrt(v_e3)(0.02, 1e-10) = {R:.3f}  (sa borne pour r seul : > 10)")
    print("\n(2) sqrt(v)/lr (1e-7) et gain lr/sqrt(v) (1e6) :")
    for lr, eps, sd, D, L, v in T:
        print(f"   lr {lr:4.2f} eps {eps:6.0e} : sqrt(v)/lr = {v / lr * 1e7:.3f}e-7   gain = {lr / v / 1e6:.3f}e6   r = eps/sqrt(v) = {eps / v:.4e}")
    g = [lr / v for lr, eps, sd, D, L, v in T]
    print(f"   gain : min {min(g):.3e} max {max(g):.3e} (rapport {max(g) / min(g):.3f}) ; r : min {min(e / v for lr, e, s, D, L, v in T):.2e} max {max(e / v for lr, e, s, D, L, v in T):.2e}")
    print("\n(3) L et |D_d| au MEME r : lr 0,02 contre colonne lr 0,05 interpolee (lineaire en log r)")
    r05 = [e / v for lr, e, s, D, L, v in v05]; L05 = [L for *_, L, v in v05]; D05 = [D for lr, e, s, D, L, v in v05]
    print("   lr 0,02 : eps       r        L(0,02)   L(0,05) interp   dL       |D_d|(0,02)   |D_d|(0,05) interp   dD/D")
    for lr, eps, sd, D, L, v in v02:
        r = eps / v
        l5, d5 = interp(r, r05, L05), interp(r, r05, D05)
        if l5 is None:
            print(f"   {eps:6.0e}   {r:.3e}   {L:.5f}   hors plage (r(0,05) de {min(r05):.1e} a {max(r05):.1e})")
        else:
            print(f"   {eps:6.0e}   {r:.3e}   {L:.5f}   {l5:.5f}        {L - l5:+.5f}   {D:.4e}    {d5:.4e}         {(D / d5 - 1) * 100:+.1f} %")
    print("\n   paires (lr 0,05, lr 0,02) au r le plus proche :")
    for lr5, e5, s5, D5, L5, v5 in v05:
        r5 = e5 / v5
        best = min(v02, key=lambda t: abs(math.log((t[1] / t[5]) / r5)))
        r2 = best[1] / best[5]
        print(f"   (0,05, {e5:6.0e}) r = {r5:.3e} L {L5:.5f} |D_d| {D5:.3e}   |  (0,02, {best[1]:6.0e}) r = {r2:.3e} L {best[4]:.5f} |D_d| {best[3]:.3e}   rapport des r {max(r5, r2) / min(r5, r2):.2f}  dL {best[4] - L5:+.5f}  d|D_d| {(best[3] / D5 - 1) * 100:+.1f} %")
    print("\n(4) autres organisateurs : L en fonction de eps/lr et de sd, par colonne")
    for lr, eps, sd, D, L, v in sorted(T, key=lambda t: t[1] / t[0]):
        print(f"   eps/lr = {eps / lr:.2e}   lr {lr:4.2f} eps {eps:6.0e}   sd {sd:.3e}   L {L:.5f}   |D_d| {D:.3e}")
