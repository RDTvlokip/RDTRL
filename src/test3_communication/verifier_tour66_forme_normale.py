"""Tour 66 (vraie critique de dipankarsarkar, 13/10/2026) : verification de ses affirmations.

(1) Forme normale 1-D du pli, g = d^2 - x (descente), noeud a +sqrt(x), col a -sqrt(x) ; Adam a la PyTorch
    (correction de biais, eps ajoute apres sqrt(v_hat)), 3000 pas depuis un coup de 1e-5 au-dessus du noeud.
    Trois transformations, c = 0,4 : (c lr, c^2 eps, c^2 x, c d) [la sienne, exacte d'apres lui, 7e-15] ;
    (c lr, c eps, x fixe) [ma paire, 54 % d'apres lui] ; (c lr, c eps, c^2 x) [58 % d'apres lui]. Il ne donne
    pas lr, eps, x de base : plusieurs jeux de base, l'ecart depend du jeu.
(2) Son arithmetique : d / (2 sqrt(x)) pour d = 1e-5, x = 1e-13 et 1e-9.
(3) La reduction a deux lignes : le gradient ne depend que de (z3 = e3-o3, z4 = e4-o4, u = l3-l4). Jacobien
    3x3 de (g_e3, g_e4, g_l3) par rapport a (z3, z4, u) au noeud pres du pli (forme fermee, Se/Sed de l'etat
    de base) : valeurs propres, vecteur nul (la direction du pli), et la raideur lineaire de z3 seule,
    k3 = -d g_e3 / d z3, qui ne depend pas de x. Plus la meme chose en unites d'Adam (divisee par
    sqrt(v) mesure par coordonnee, tour 64/65) pour voir quelle coordonnee porte le mode mou.
"""

import sys
import numpy as np
from mpmath import mp, mpf, exp, log, matrix, eig

sys.path.insert(0, ".")
from verifier_tour61_decomposition_d3 import PLI, u_noeud
from hasard_reduction_tour59 import base_state

mp.dps = 50
B, C, N = mpf("0.02"), mpf(26), mpf(27)


def adam_1d(lr, eps, x, d0, n, b1=0.9, b2=0.999):
    d = d0; m = 0.0; v = 0.0
    out = np.empty(n)
    for t in range(1, n + 1):
        g = d * d - x
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        d = d - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        out[t - 1] = d
    return out


def ecart(ref, autre, c):
    return np.max(np.abs(autre - c * ref)) / np.max(np.abs(c * ref))


def grad(z3, z4, u, delta, Se, Sed):
    ex3 = C * exp(-z3); s3 = 1 / (1 + ex3); d3 = ex3 * s3
    ex4 = C * exp(-z4); d4 = ex4 / (1 + ex4); s4 = 1 - d4
    w3 = exp(u); Z = w3 + 1 + Se; r3 = w3 / Z; r4 = 1 / Z; logZ = log(Z)
    Hr = -(r3 * (u - logZ) + r4 * (-logZ) + (Sed - logZ * Se) / Z)
    a3 = (1 - delta) * s3; a4 = (1 + delta) * s4; abar = r3 * a3 + r4 * a4
    g4 = r3 * ((a3 - abar) - B * ((u - logZ) + Hr)) / N
    X3 = (1 - delta) * r3 / B; X4 = (1 + delta) * r4 / B
    ge3 = s3 * d3 * B * (X3 - z3) / N
    ge4 = s4 * d4 * B * (X4 - z4) / N
    return [ge3, ge4, g4], (d3, r4)


if __name__ == "__main__":
    print("(1) forme normale g = d^2 - x, 3000 pas, coup de 1e-5 au-dessus du noeud, c = 0,4")
    c = 0.4
    for lr, eps, x in ((1e-6, 1e-8, 1e-8), (1e-5, 1e-8, 1e-8), (1e-6, 1e-10, 1e-8), (1e-6, 1e-9, 1e-9), (1e-5, 1e-7, 1e-6)):
        d0 = np.sqrt(x) + 1e-5
        ref = adam_1d(lr, eps, x, d0, 3000)
        a = adam_1d(c * lr, c * c * eps, c * c * x, c * d0, 3000)
        b = adam_1d(c * lr, c * eps, x, np.sqrt(x) + c * 1e-5, 3000)
        b2 = adam_1d(c * lr, c * eps, c * c * x, c * d0, 3000)
        print(f"   lr {lr:.0e} eps {eps:.0e} x {x:.0e} : (c lr, c^2 eps, c^2 x, c d) {ecart(ref, a, c):.1e} | (c lr, c eps, x fixe) {ecart(ref - np.sqrt(x), b - np.sqrt(x), c):.2f}"
              f" (ecart a c*(d - noeud)) ; meme paire, d brut contre c*d : {ecart(ref, b, c):.2f} | (c lr, c eps, c^2 x) {ecart(ref, b2, c):.2f}")
    print("\n(2) d / (2 sqrt(x)) pour d = 1e-5 :", {x: round(1e-5 / (2 * np.sqrt(x)), 3) for x in (1e-13, 1e-11, 1e-9, 1e-8)})

    print("\n(3) jacobien de la reduction au noeud (z3, z4, u)")
    s = base_state()
    Se, Sed = mpf(s["Se"]), mpf(s["Sed"])
    print(f"   Se = {float(Se):.3e}  Sed = {float(Sed):.3e}")
    for xo in ("1e-13", "1e-9", "1e-8"):
        delta = PLI - mpf(xo)
        u = u_noeud(delta)
        w3 = exp(u); Z = w3 + 1 + Se; r3 = w3 / Z; r4 = 1 / Z
        z3 = (1 - delta) * r3 / B; z4 = (1 + delta) * r4 / B
        p = [z3, z4, u]
        g0, (d3, r4v) = grad(z3, z4, u, delta, Se, Sed)
        h = mpf("1e-15")
        J = matrix(3, 3)
        for j in range(3):
            pp = list(p); pm = list(p); pp[j] += h; pm[j] -= h
            gp, _ = grad(*pp, delta, Se, Sed); gm, _ = grad(*pm, delta, Se, Sed)
            for i in range(3):
                J[i, j] = (gp[i] - gm[i]) / (2 * h)
        ev, vec = eig(J)
        order = sorted(range(3), key=lambda k: abs(ev[k]))
        print(f"   x = {xo} : residu |g| au noeud {float(max(abs(t) for t in g0)):.1e} (Se != 0 : le noeud ferme ignore la queue) ; d3 = {float(d3):.6e}, r4 = {float(r4v):.6f}")
        for k in order:
            vv = [complex(vec[i, k]) for i in range(3)]
            nrm = np.sqrt(sum(abs(t) ** 2 for t in vv))
            print(f"      valeur propre {complex(ev[k]).real:+.4e}   vecteur (z3, z4, u) = ({vv[0].real / nrm:+.4f}, {vv[1].real / nrm:+.4f}, {vv[2].real / nrm:+.4f})")
        k3 = -J[0, 0]
        print(f"      raideur de z3 seule k3 = -dg_e3/dz3 = {float(k3):.4e} (s3 d3 beta / N = {float((1 - d3) * d3 * B / N):.4e}) ; dg_e3/du = {float(J[0, 2]):+.3e} ; dg_l3/dz3 = {float(J[2, 0]):+.3e} ; dg_l3/du = {float(J[2, 2]):+.3e}")
        # en unites d'Adam : z3 bouge de 2*lr*m/sqrt(v_e3) (o3 suit a l'oppose), u bouge de lr*(m4/sqrt(v4) - m5/sqrt(v5))
        # bord de stabilite d'Adam (b1 = 0,9) : lr * k_eff / sqrt(v) < 2 (1 + b1) / (1 - b1) = 38
        for sv_lr in (8.2e-7, 9.5e-7):
            print(f"      lr * 2 k3 / sqrt(v_e3) avec sqrt(v_e3)/lr = {sv_lr:.1e} : {float(2 * k3) / sv_lr:.2f}  (bord de stabilite d'Adam : 38)")
