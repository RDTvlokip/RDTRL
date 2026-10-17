"""Tour 74 : modele du gain corrige du momentum, hors echantillon.

taux(t) = ln mu, mu - 1 = eps solution de eps^2 + eps (1 - b1)(1 - cJ) - cJ (1 - b1) = 0 (equation caracteristique du mode lent avec momentum),
cJ = exp(KA * taux73(t)) - 1, taux73(t) = lr* lambda_s [1/(s + eps_adam) + 1/(s + 26 eps_adam)], lambda_s = 7,30e-5 sqrt(x), s(t) = s(200) exp(-(t - 200)/2000).
KA = 0,983 : rapport ln A / taux73 mesure a t = 200 et 350 a x = 1e-6 (verifier_tour74_valeur_propre.py), le seul nombre calibre.
Ancre : t(delta = 1e-6) mesure (tol 0,02, au-dessus de lr*). Predit t(1e-7), t(1e-8), t(1e-9), t(1e-10), compare aux temps mesures par trajets.
Usage : python verifier_tour74_modele_corrige.py <q> <xw> [<KA>]
"""

import sys

import numpy as np
from mpmath import mpf

sys.path.insert(0, ".")
import verifier_tour72_retour_et_jacobien as v72
from verifier_tour61_decomposition_d3 import PLI, d3_noeud

B1 = 0.9


def taux_corrige(r73, ka):
    cj = np.expm1(ka * r73)
    e = 0.5 * (-(1 - B1) * (1 - cj) + np.sqrt(((1 - B1) * (1 - cj)) ** 2 + 4 * cj * (1 - B1)))
    return np.log1p(e)


if __name__ == "__main__":
    q = float(sys.argv[1])
    xw = sys.argv[2]
    ka = float(sys.argv[3]) if len(sys.argv) > 3 else 0.983
    v72.TOL_BISSECTION = 1e-15
    ls = v72.bissection(q, xw)
    d3j = float(d3_noeud(PLI - mpf(xw), -1)); z3_col = float(np.log(26 * (1 - d3j) / d3j))
    mes = {}
    for e in range(-6, -11, -1):
        r = v72.trajet(ls + 10.0 ** e, q, xw)
        ok = np.abs(r[:, 1] - z3_col) < 0.02
        mes[10.0 ** e] = int(np.nonzero(ok)[0][-1]) + 1 if np.any(ok) else 0
    r = v72.trajet(ls + 1e-7, q, xw)
    t0 = 200
    s0 = float(np.median(r[t0 - 20:t0 + 20, 3]))
    eps = q * ls
    lam = 7.30e-5 * float(mpf(xw)) ** 0.5

    def taux73(t):
        s = s0 * np.exp(-(t - t0) / 2000.0)
        return ls * lam * (1.0 / (s + eps) + 1.0 / (s + 26.0 * eps))

    print(f"q = {q:.3e} xw = {xw} : lr* = {ls:.15f}, s(200) = {s0:.4e}, taux73(200) = {taux73(200):.5f}, taux corrige(200) = {taux_corrige(taux73(200), ka):.5f}")
    print("   mesure t(delta) :", mes)
    for nom, f in (("modele corrige du momentum", lambda t: taux_corrige(taux73(t), ka)), ("modele du tour 73", taux73)):
        t = float(mes[1e-6])
        print(f"   {nom}")
        print("      delta    t predit  t mesure  ecart")
        for d in (1e-7, 1e-8, 1e-9, 1e-10):
            acc, dt = 0.0, 0.1
            while acc < np.log(10.0):
                acc += f(t) * dt
                t += dt
            print(f"      {d:.0e}   {t:7.0f}   {mes[d]:7d}   {100 * (t / mes[d] - 1):+.1f} %")
