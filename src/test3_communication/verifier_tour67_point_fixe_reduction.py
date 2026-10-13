"""Tour 67 : point fixe propre de la reduction (avec la queue Se, Sed de l'etat de base) contre le noeud ferme
d3_noeud (qui ignore la queue). Le gradient de la reduction (verifier_tour66_forme_normale.grad) s'annule en
z3 = X3(u), z4 = X4(u), g_l3(u) = 0 ; une racine en u (une inconnue). Compare d3 et r4 des deux, pour x = 1e-8, 1e-9,
1e-10, 1e-11, 1e-13, et le produit (difference) * sqrt(x) (constante si la difference vient d'un residu de gradient
divise par une valeur propre molle en sqrt(x)).
Si la difference est K/sqrt(x), D_d contre le noeud ferme contient un terme parasite D0 = K/sqrt(x) : D_d' = D_d - D0.
"""

import json
import sys

from mpmath import mp, mpf, exp, findroot, log, sqrt

sys.path.insert(0, ".")
from verifier_tour61_decomposition_d3 import PLI, d3_noeud, r4_noeud, u_noeud
from verifier_tour66_forme_normale import grad, B

mp.dps = 60

if __name__ == "__main__":
    d = json.load(open("etat_base_reduction_mur23.json", encoding="utf-8"))
    Se, Sed = mpf(d["Se"]), mpf(d["Sed"])
    print(f"Se = {float(Se):.4e}   Sed = {float(Sed):.4e}")
    print("x         d3 noeud ferme     d3 point fixe      diff              diff*sqrt(x)   r4 ferme - r4 point fixe")
    for xs in ("1e-8", "1e-9", "1e-10", "1e-11", "1e-13"):
        x = mpf(xs)
        delta = PLI - x

        def comp(u):
            w3 = exp(u); Z = w3 + 1 + Se; r3 = w3 / Z; r4 = 1 / Z
            return (1 - delta) * r3 / B, (1 + delta) * r4 / B, r4

        def f(u):
            z3, z4, _ = comp(u)
            g, _ = grad(z3, z4, u, delta, Se, Sed)
            return g[2]

        u0 = u_noeud(delta)
        uf = findroot(f, u0, tol=mpf(10) ** -40, maxsteps=200)
        z3, z4, r4f = comp(uf)
        ex3 = 26 * exp(-z3); d3f = ex3 / (1 + ex3) * (1 / (1 + ex3)) * (1 + ex3) / (1 + ex3) if False else ex3 / (1 + ex3)
        dn = d3_noeud(delta); rn = r4_noeud(delta)
        print(f"{xs:7s}   {float(dn):.10e}   {float(d3f):.10e}   {float(d3f - dn):+.4e}   {float((d3f - dn) * sqrt(x)):+.4e}   {float(rn - r4f):+.4e}")
