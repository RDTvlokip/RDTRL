"""Tour 73 : modele du gain pour le temps de depart du col.

taux(t) = lr* lambda_s [1/(s(t) + eps) + 1/(s(t) + 26 eps)], lambda_s = 7,30e-5 sqrt(x) (jacobien du tour 66), eps = q lr*,
s(t) = s(t0) exp(-(t - t0)/2000) (v decroit d'un facteur 0,999 par pas, sqrt(v) d'un facteur 0,9995 : constante de temps 2000).
Depart d'une decade : integrale du taux = ln 10. On part du temps mesure t(1e-6) (tol 0,02, au-dessus de lr*) et on predit les decades suivantes,
a comparer aux mesures 600, 719, 833, 944 (verifier_tour73_decades.py). s(t0) = mediane de sqrt(v_e3) mesuree sur le trajet a lr* + 1e-7 autour de t0 = 200.
Usage : python verifier_tour73_modele_gain.py [<q> <xw>]
"""

import sys

import numpy as np
from mpmath import mpf

sys.path.insert(0, ".")
import verifier_tour72_retour_et_jacobien as v72

MESURES = {1e-6: 476, 1e-7: 600, 1e-8: 719, 1e-9: 833, 1e-10: 944}

if __name__ == "__main__":
    q = float(sys.argv[1]) if len(sys.argv) > 1 else 5.95e-7
    xw = sys.argv[2] if len(sys.argv) > 2 else "1e-6"
    ls = 0.314002321343052
    r = v72.trajet(ls + 1e-7, q, xw)
    t0 = 200
    s0 = float(np.median(r[t0 - 20:t0 + 20, 3]))
    eps = q * ls
    lam = 7.30e-5 * float(mpf(xw)) ** 0.5
    tau = 2000.0

    def taux(t):
        s = s0 * np.exp(-(t - t0) / tau)
        return ls * lam * (1.0 / (s + eps) + 1.0 / (s + 26.0 * eps))

    print(f"s(t0 = {t0}) = {s0:.4e}, eps = {eps:.3e}, lambda_s = {lam:.3e}, taux(200) = {taux(200):.5f}, taux(700) = {taux(700):.5f}, taux(900) = {taux(900):.5f}")
    t = float(MESURES[1e-6])
    print("   delta      t predit  t mesure  ecart")
    print(f"   1e-06      {t:7.0f}   {MESURES[1e-6]:7d}   (point de depart)")
    for d in (1e-7, 1e-8, 1e-9, 1e-10):
        # integrale du taux = ln 10, pas de 0,1 pas
        acc, dt = 0.0, 0.1
        while acc < np.log(10.0):
            acc += taux(t) * dt
            t += dt
        print(f"   {d:.0e}   {t:7.0f}   {MESURES[d]:7d}   {100 * (t / MESURES[d] - 1):+.1f} %")
