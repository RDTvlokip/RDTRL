"""Tour 68 : sa racine g_l3 = g_l4 (au lieu de g_l3 = 0) pour le point fixe de la reduction, 60 chiffres.

Condition : u = l3 - l4 est immobile quand lr m4/(sqrt(v4)+eps) = lr m5/(sqrt(v5)+eps) ; a un point fixe m -> g, sqrt(v) -> |g|,
et g/(|g|+eps) est strictement monotone : g_l3 = g_l4 (pour tout lr, tout eps). z3 : ge3 = 0 donc z3 = X3(u). z4 : ge4 porte d4 (~1e-12 a 1e-17) et
z4 ne bouge que de ~1e-15 par pas : le simulateur le garde a sa valeur de base (30,02), pas a X4(u) = 41.
Sorties pour x = 1e-8, 1e-9, 1e-10 : d3 du noeud ferme, de la racine g_l3 = 0 (z4 = X4), g_l3 = g_l4 (z4 = X4), g_l3 = g_l4 (z4 base),
et les differences multipliees par sqrt(x) ; puis la valeur commune g_l3 = g_l4 au point fixe (+3,495e-13 attendu) ;
puis la tangence (disparition du noeud) : PLI de la forme fermee, de g_l3 = 0 et de g_l3 = g_l4 avec z4 a la base.
"""

import json
import sys

from mpmath import mp, mpf, exp, findroot, log, sqrt

sys.path.insert(0, ".")
from verifier_tour61_decomposition_d3 import PLI, d3_noeud, u_noeud
from verifier_tour66_forme_normale import B, N, C

mp.dps = 60


def g_l(u, z3, z4, delta, Se, Sed):
    """g[4], g[5] de run() (convention de la reduction : gradient de J)."""
    ex4 = C * exp(-z4); d4 = ex4 / (1 + ex4); s4 = 1 - d4
    ex3 = C * exp(-z3); s3 = 1 / (1 + ex3)
    w3 = exp(u); Z = w3 + 1 + Se; r3 = w3 / Z; r4 = 1 / Z; logZ = log(Z)
    Hr = -(r3 * (u - logZ) + r4 * (-logZ) + (Sed - logZ * Se) / Z)
    a3 = (1 - delta) * s3; a4 = (1 + delta) * s4; abar = r3 * a3 + r4 * a4
    g4 = r3 * ((a3 - abar) - B * ((u - logZ) + Hr)) / N
    g5 = r4 * ((a4 - abar) - B * ((-logZ) + Hr)) / N
    return g4, g5


def d3_de(u, delta, Se):
    w3 = exp(u); Z = w3 + 1 + Se
    z3 = (1 - delta) * (w3 / Z) / B
    ex3 = C * exp(-z3)
    return ex3 / (1 + ex3), z3


def racine(delta, Se, Sed, mode, z4fixe):
    """mode 'zero' : g4 = 0 ; 'egal' : g4 - g5 = 0. z4fixe=None -> z4 = X4(u)."""
    def f(u):
        _, z3 = d3_de(u, delta, Se)
        z4 = z4fixe if z4fixe is not None else (1 + delta) * (1 / (1 + exp(u) + Se)) / B
        g4, g5 = g_l(u, z3, z4, delta, Se, Sed)
        return g4 if mode == "zero" else g4 - g5
    u0 = u_noeud(delta)
    return findroot(f, u0, tol=mpf(10) ** -45, maxsteps=300)


if __name__ == "__main__":
    d = json.load(open("etat_base_reduction_mur23.json", encoding="utf-8"))
    Se, Sed = mpf(d["Se"]), mpf(d["Sed"])
    z4b = mpf(d["x"][2]) - mpf(d["x"][3])
    print(f"z4 de base = {float(z4b):.6f}   Se = {float(Se):.4e}   Sed = {float(Sed):.4e}")
    print("x       d3 [g_l3 = 0, z4=X4]    d3 [g_l3 = g_l4, z4=X4]   diff        *sqrt(x)   | d3 [g_l3=g_l4, z4 base]   moins noeud ferme   *sqrt(x)   | g_l3 = g_l4 au point fixe")
    for xs in ("1e-8", "1e-9", "1e-10"):
        x = mpf(xs); delta = PLI - x
        dn = d3_noeud(delta)
        u0 = racine(delta, Se, Sed, "zero", None); u1 = racine(delta, Se, Sed, "egal", None); u2 = racine(delta, Se, Sed, "egal", z4b)
        a, _ = d3_de(u0, delta, Se); b, _ = d3_de(u1, delta, Se); c, z3c = d3_de(u2, delta, Se)
        g4, g5 = g_l(u2, z3c, z4b, delta, Se, Sed)
        print(f"{xs:6s}  {float(a):.10e}   {float(b):.10e}   {float(b - a):+.3e}  {float((b - a) * sqrt(x)):+.3e} |  {float(c):.10e}   {float(c - dn):+.3e}   {float((c - dn) * sqrt(x)):+.3e} |  {float(g4):+.4e} {float(g5):+.4e}")
    print("\ntangence : (u, delta) avec F = 0 et dF/du = 0 (z4 a la base), F = g_l3 ou g_l3 - g_l4 ; le noeud n'existe que pour delta < delta_t")
    from mpmath import diff
    for nom, mode in (("g_l3 = 0", "zero"), ("g_l3 = g_l4", "egal")):
        def F(u, delta, mode=mode):
            _, z3 = d3_de(u, delta, Se)
            g4, g5 = g_l(u, z3, z4b, delta, Se, Sed)
            return g4 if mode == "zero" else g4 - g5
        sol = findroot(lambda u, dl: (F(u, dl), diff(lambda uu: F(uu, dl), u)), (mpf("-1.478146"), PLI), tol=mpf(10) ** -40, maxsteps=100)
        print(f"   {nom} : tangence en u = {float(sol[0]):.9f}, delta = PLI {float(sol[1] - PLI):+.4e}  (x_c = {float(PLI - sol[1]):+.4e})")
