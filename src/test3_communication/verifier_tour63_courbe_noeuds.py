"""Tour 63 (08/10/2026) : le point (D_d, D_r) = (mediane d3 - d3 au pli, mediane r4 - r4 au pli)
est-il sur la COURBE DES NOEUDS fermee (d3_noeud(delta), r4_noeud(delta)) d'un delta decale ?

Le critere "rapport delta'_r4/delta'_d3 = 1" (tour 61) n'est valide qu'en reponse lineaire
(|biais| << ecart noeud-jumeau). Ici le biais est de 6e-6 a 9e-6 pour un ecart de 2e-7 : on est tres
hors du regime lineaire. Test correct : trouver x_eff tel que d3_noeud(PLI - x_eff) - d3_pli = D_d, puis
predire D_r = r4_noeud(PLI - x_eff) - r4_pli et comparer a la valeur mesuree.

Entrees (mesurees dans la reduction numba, voir verifier_tour63_deplacement_burst.py, x <= 1e-11) :
  eps 1e-10 lr 0,05 : D_d = -6,0814e-6 , D_r = -4,3914e-5   (tour 62, x = 1e-14)
  eps 1e-8  lr 0,05 : D_d = -8,6313e-6 , D_r = -6,1159e-5   (x = 1e-13)
Usage : python verifier_tour63_courbe_noeuds.py [etiquette D_d D_r]...  (sinon les deux ci-dessus)
"""

import sys
import numpy as np
from mpmath import mp, mpf
from scipy.optimize import brentq

sys.path.insert(0, ".")
from verifier_tour61_decomposition_d3 import PLI, d3_noeud, r4_noeud

mp.dps = 60


def courbe(x):
    return (float(d3_noeud(PLI - mpf(x)) - d3_noeud(PLI - mpf("1e-30"))),
            float(r4_noeud(PLI - mpf(x)) - r4_noeud(PLI - mpf("1e-30"))))


def x_eff_pour(Dd):
    f = lambda lx: courbe(10.0 ** lx)[0] - Dd
    return 10.0 ** brentq(f, -16.0, -4.0, xtol=1e-12)


if __name__ == "__main__":
    args = sys.argv[1:]
    cas = [(args[i], float(args[i + 1]), float(args[i + 2])) for i in range(0, len(args), 3)] if args else [
        ("eps 1e-10 lr 0,05", -6.0814e-6, -4.3914e-5), ("eps 1e-8 lr 0,05", -8.6313e-6, -6.1159e-5)]
    print("cas                   D_d mesure   x_eff (d3 sur la courbe)   D_r predit (courbe)   D_r mesure    ecart      secante pred.   secante mes.")
    for nom, Dd, Dr in cas:
        x = x_eff_pour(Dd)
        dd, dr = courbe(x)
        print(f"{nom:20s}  {Dd:+.4e}   {x:.4e}               {dr:+.4e}          {Dr:+.4e}   {100 * (Dr / dr - 1):+6.2f} %   {dr / dd:.4f}       {Dr / Dd:.4f}")
