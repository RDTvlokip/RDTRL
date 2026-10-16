"""Tour 73 (vraie critique de dipankarsarkar, 19/10/2026) : le cout d'une decade de delta au-dessus et au-dessous de lr*, jusqu'a delta = 1e-10.

Il lit mes tableaux du tour 72 : sous lr*, tol 0,005 : decades 67, 133, 132, 127 (de 1e-3 vers le bas) ; au-dessus, tol 0,02 : 133, 129, 124 ; le
log10 / ln(valeur propre) du jacobien : 132 (t = 350), 133 (t = 500) sous lr*, 137, 131, 118 (t = 200, 350, 500) au-dessus ; ajustement 1e-7 a 1e-4 : -130,8,
taux 0,01760 contre ln 1,01757 = 0,01742. Il predit : si le gain monte, delta = 1e-8 coute moins de 124 pas pour sa decade au-dessus.

Usage : python verifier_tour73_decades.py controle
        python verifier_tour73_decades.py decades [<q> <xw>]       (defaut 5,95e-7 1e-6 ; lr* par bissection a 1e-15)
"""

import sys

import numpy as np
from mpmath import mpf

sys.path.insert(0, ".")
import verifier_tour72_retour_et_jacobien as v72
from verifier_tour61_decomposition_d3 import PLI, d3_noeud

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "controle":
        bas = {1e-7: 539, 1e-6: 412, 1e-5: 280, 1e-4: 147, 1e-3: 80, 3e-3: 57}
        hau = {1e-7: 600, 1e-6: 476, 1e-5: 347, 1e-4: 214, 1e-3: 100}
        d = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
        print("decades sous lr* (tol 0,005) :", [bas[d[i + 1]] - bas[d[i]] for i in range(4)])
        print("decades au-dessus (tol 0,02), de 1e-4 vers le bas :", [hau[d[i + 1]] - hau[d[i]] for i in range(1, 4)])
        for nom, lead in (("sous lr* t = 350", 1.01757), ("sous lr* t = 500", 1.01745), ("t = 100", 1.03235), ("au-dessus t = 200", 1.01692), ("au-dessus t = 350", 1.01772), ("au-dessus t = 500", 1.01970)):
            print(f"   {nom} : ln = {np.log(lead):.6f}, ln 10 / ln = {np.log(10) / np.log(lead):.1f}")
        x = np.log10([1e-7, 1e-6, 1e-5, 1e-4]); y = [539, 412, 280, 147]
        sl = np.polyfit(x, y, 1)[0]
        print(f"   pente 1e-7 a 1e-4 : {sl:.1f} ; taux {np.log(10) / abs(sl):.5f} ; ln 1,01757 = {np.log(1.01757):.5f} ; ecart {100 * (np.log(10) / abs(sl) / np.log(1.01757) - 1):.2f} %")
    else:
        q = float(sys.argv[2]) if len(sys.argv) > 2 else 5.95e-7
        xw = sys.argv[3] if len(sys.argv) > 3 else "1e-6"
        v72.TOL_BISSECTION = 1e-15
        ls = v72.bissection(q, xw)
        d3j = float(d3_noeud(PLI - mpf(xw), -1)); z3_col = float(np.log(26 * (1 - d3j) / d3j))
        print(f"q = {q:.3e} xw = {xw} : lr* = {ls:.15f} (bissection a 1e-15)")
        res = {}
        for sens, nom in ((+1, "au-dessus"), (-1, "au-dessous")):
            for tol in (0.02, 0.005):
                t = []
                for e in range(-3, -11, -1):
                    d = 10.0 ** e
                    r = v72.trajet(ls + sens * d, q, xw)
                    ok = np.abs(r[:, 1] - z3_col) < tol
                    t.append(int(np.nonzero(ok)[0][-1]) + 1 if np.any(ok) else 0)
                res[(nom, tol)] = t
        print("   delta        " + "  ".join(f"{10.0 ** e:8.0e}" for e in range(-3, -11, -1)))
        for (nom, tol), t in res.items():
            dec = [t[i + 1] - t[i] for i in range(len(t) - 1)]
            print(f"   {nom:10s} tol {tol:5.3f} t = {t}")
            print(f"   {'':10s}           cout de la decade (1e-3->1e-4 ... 1e-9->1e-10) : {dec}")
