"""Tour 65 (vraie critique de dipankarsarkar) : (1) ses chiffres recalcules a partir de la table T du tour 64 ;
(2) son invariance d'echelle d'Adam, (lr, eps, deplacement) -> (c lr, c eps, c deplacement), verifiee sur un
gradient lineaire puis sur un gradient avec terme quadratique (la symetrie doit casser) ;
(3) a quel point ses "paires" sont a eps/lr egal (la symetrie ne s'applique qu'a eps/lr egal).
"""

import math
import numpy as np
from verifier_tour64_analyse_grille import T


def adam_traj(grad, x0, lr, eps, n, b1=0.9, b2=0.999):
    x = np.array(x0, float); m = np.zeros_like(x); v = np.zeros_like(x)
    out = np.empty((n, x.size))
    for t in range(1, n + 1):
        g = grad(x)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        x = x - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        out[t - 1] = x
    return out


if __name__ == "__main__":
    by = {(lr, eps): (sd, D, L, sv) for lr, eps, sd, D, L, sv in T}
    print("(1) ses nombres, recalcules de ma table")
    print("   fixe lr 0,05, eps 1e-10 -> 1e-8 : sd /", round(by[(0.05, 1e-10)][0] / by[(0.05, 1e-8)][0], 3), " |D_d| x", round(by[(0.05, 1e-8)][1] / by[(0.05, 1e-10)][1], 3))
    pairs = [((0.05, 1e-10), (0.02, 1e-10)), ((0.05, 1e-9), (0.02, 3e-10)), ((0.05, 3e-9), (0.02, 1e-9)), ((0.05, 1e-8), (0.02, 3e-9)), ((0.05, 2.6e-8), (0.02, 1e-8))]
    print("   paire (lr 0,05 | 0,02)        eps/lr (0,05 | 0,02)   ecart eps/lr   sd    sqrt(v)   |D_d|   1-L (0,05 | 0,02)")
    for a, b in pairs:
        sa, sb = by[a], by[b]
        qa, qb = a[1] / a[0], b[1] / b[0]
        print(f"   {a[1]:7.1e} | {b[1]:7.1e}      {qa:8.1e} | {qb:8.1e}      x{max(qa, qb) / min(qa, qb):5.2f}    {sa[0] / sb[0]:5.2f}   {sa[3] / sb[3]:5.3f}   {sa[1] / sb[1]:5.2f}   {1 - sa[2]:.4f} | {1 - sb[2]:.4f}")
    sa, sb = by[(0.05, 2.6e-8)], by[(0.02, 1e-8)]
    print(f"   paire du haut : sqrt(v)/sd = {sa[3] / sa[0]:.5f} | {sb[3] / sb[0]:.5f} ; sd contre 2,5 : {sa[0] / sb[0] / 2.5 - 1:+.4f} ; |D_d| contre 2,5 : {sa[1] / sb[1] / 2.5 - 1:+.3f} ; 1-L : x{(1 - sa[2]) / (1 - sb[2]):.2f}")
    s_eps = math.log(by[(0.05, 1e-8)][0] / by[(0.05, 2.6e-8)][0]) / math.log(2.6)
    print(f"   sensibilite locale de sd a eps (lr 0,05, de 1e-8 a 2,6e-8) : d ln sd / d ln eps = {-s_eps:.2f} ; un ecart de 4 % en eps/lr (5,2e-7 contre 5,0e-7) pese donc {abs(s_eps) * 4:.1f} % sur sd, plus que les 0,3 % qu'il cite")
    print(f"   r pour eps 1e-7 (lr 0,05) : {1e-7 / by[(0.05, 2.6e-8)][3]:.2f} avec sqrt(v) du dernier etat ; extrapolation lineaire en eps de sqrt(v) : ", end="")
    s1, s2 = by[(0.05, 1e-8)][3], by[(0.05, 2.6e-8)][3]
    sv = s2 + (s2 - s1) / (2.6e-8 - 1e-8) * (1e-7 - 2.6e-8)
    print(f"sqrt(v) = {sv:.3e}, r = {1e-7 / sv:.2f}")

    print("\n(2) invariance d'echelle d'Adam, systeme 2D a gradient g = H x (+ q x^2 : non lineaire), 20 000 pas")
    H = np.array([[2.0, 0.7], [0.7, 0.5]])
    x0 = np.array([1.0, -0.6])
    lr, eps, n = 0.05, 1e-3, 20000
    for nom, q in (("lineaire", 0.0), ("quadratique q = 1", 1.0)):
        grad = lambda x, q=q: H @ x + q * x * x
        t0 = adam_traj(grad, x0, lr, eps, n)
        for c in (0.4, 2.5):
            t1 = adam_traj(grad, c * x0, c * lr, c * eps, n)
            e = [np.max(np.abs(t1[:k] - c * t0[:k])) / np.max(np.abs(c * t0[:k])) for k in (200, 2000, 20000)]
            print(f"   {nom:18s} c = {c}: ecart relatif max (pas 200 | 2000 | 20000) = {e[0]:.2e} | {e[1]:.2e} | {e[2]:.2e}")
    print("   (gradient lineaire : ecart ~ arrondi puis amplifie par la dynamique chaotique des salves ; quadratique : il est macroscopique des les premiers pas)")
