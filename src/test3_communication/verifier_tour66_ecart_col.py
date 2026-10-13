"""Tour 66 : ecart noeud-col en d3, d3(col) - d3(noeud), en fonction de x = pli - delta (forme fermee a trois
lignes, d3_noeud(signe=-1) donne le jumeau instable). Hypothese H66-7 : l'effondrement a petit x est la
mediane qui passe du cote du col ; l'effondrement doit commencer quand D_d (> 0) atteint cet ecart.
"""

import sys
from mpmath import mp, mpf

sys.path.insert(0, ".")
from verifier_tour60_exposant_reduction import PLI, d3_noeud

mp.dps = 50

if __name__ == "__main__":
    print("x          d3 noeud          d3 col            ecart col-noeud   ecart / sqrt(x)")
    for x in ("1e-13", "1e-11", "1e-10", "3e-10", "4e-10", "1e-9", "4e-9", "1e-8"):
        d = PLI - mpf(x)
        n, s = d3_noeud(d, +1), d3_noeud(d, -1)
        print(f"{x:8s}   {float(n):.10e}  {float(s):.10e}  {float(s - n):+.4e}       {float((s - n) / mp.sqrt(mpf(x))):.4f}")
