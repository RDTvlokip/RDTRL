"""Tour 62 (07/10/2026) : pente rho = lim_{x->0} (d r4_noeud/d delta)/(d d3_noeud/d delta) du
mode mou au pli, en forme fermee (mpmath). La limite du rapport des medianes est
L = (D_r/D_d)/rho, avec D_r/D_d = 7,2210 (deplacement de la mediane depuis le point de pli,
mesure dans la reduction, constant a 4 chiffres pour x <= 1e-11). On compare L a la limite 0,96799
de l'ajustement c0 + c1 sqrt(x) + c2 x.
"""

import sys
from mpmath import mp, mpf

sys.path.insert(0, ".")
from verifier_tour61_decomposition_d3 import PLI, d3_noeud, r4_noeud, pente_fn

mp.dps = 60

if __name__ == "__main__":
    DR_DD = 7.2210  # moyenne des lignes x <= 1e-11 du test de deplacement sature : 7,22126 / 7,22088 / 7,22076 / 7,22097
    print("x             rho(x) = pente_r/pente_d      L = (D_r/D_d)/rho")
    for off in ("-1e-9", "-1e-10", "-1e-12", "-1e-14", "-1e-16", "-1e-20"):
        rho = pente_fn(r4_noeud, PLI + mpf(off)) / pente_fn(d3_noeud, PLI + mpf(off))
        print(f"{off:>8s}      {rho:.6f}                    {DR_DD / rho:.5f}")
