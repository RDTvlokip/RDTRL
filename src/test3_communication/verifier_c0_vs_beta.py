"""Hypothese formee en attendant d'autres calculs (17/09/2026) : le
coefficient C0=0,2212603871 (le prefacteur du terme sqrt(delta_c-delta)
au pli, verifie_puiseux_gap.py) suit-il une loi de puissance simple en
beta (ex. C0 ~ sqrt(beta), ou C0 ~ beta) ? Ce serait une reponse
POURQUOI/COMBIEN a "d'ou vient ce chiffre precis" allant au-dela de
"c'est la solution numerique du systeme a beta=0,02".

Balayage par CONTINUATION (chaque beta reutilise la solution du beta
precedent comme point de depart) -- une premiere tentative sans
continuation (meme estimation initiale pour tous les beta) avait donne
un delta_c(beta) NON MONOTONE et meme negatif a grand beta, signe d'un
saut de branche du solveur de Newton plutot qu'un vrai phenomene ; la
continuation confirme que c'est bien reel (courbe lisse, continue).
"""

import sys
sys.path.insert(0, '.')
import mpmath as mp

mp.mp.dps = 50


def h(d3, delta, beta):
    R = 1 / (1 + mp.e ** (-(2 * delta + (1 - delta) * d3) / beta))
    return d3 / (26 * (1 - d3)) - mp.e ** (-(1 - delta) * (1 - R) / beta)


def localiser_pli(beta, d3_guess, delta_guess):
    d3, delta = mp.mpf(d3_guess), mp.mpf(delta_guess)
    for _ in range(100):
        F = mp.matrix([h(d3, delta, beta), mp.diff(lambda x: h(x, delta, beta), d3)])
        Jd3 = mp.matrix([
            mp.diff(lambda x: h(x, delta, beta), d3),
            mp.diff(lambda x: mp.diff(lambda y: h(y, delta, beta), x), d3),
        ])
        Jde = mp.matrix([
            mp.diff(lambda de: h(d3, de, beta), delta),
            mp.diff(lambda de: mp.diff(lambda y: h(y, de, beta), d3), delta),
        ])
        J = mp.matrix([[Jd3[0], Jde[0]], [Jd3[1], Jde[1]]])
        dv = mp.lu_solve(J, -F)
        d3 += dv[0]
        delta += dv[1]
        if max(abs(dv[0]), abs(dv[1])) < mp.mpf('1e-35'):
            break
    return d3, delta


def C0_de(d3c, dc, beta):
    F_xx = mp.diff(lambda x: h(x, dc, beta), d3c, 2)
    F_eps = -mp.diff(lambda de: h(d3c, de, beta), dc)
    A = -2 * F_eps / F_xx
    return 2 * mp.sqrt(A)


if __name__ == "__main__":
    d3_ref = mp.mpf('0.0027251196899644737')
    delta_ref = mp.mpf('0.013437210066097283')

    print("=== montee depuis beta=0.02 (continuation) ===")
    d3g, deg = d3_ref, delta_ref
    for beta_s in ('0.021', '0.022', '0.024', '0.026', '0.03', '0.035',
                   '0.04', '0.05', '0.06', '0.08', '0.10'):
        beta = mp.mpf(beta_s)
        d3c, dc = localiser_pli(beta, d3g, deg)
        C0 = C0_de(d3c, dc, beta)
        print(f"  beta={float(beta):.4f}  delta_c={float(dc):.8f}  d3_c={float(d3c):.6e}  "
              f"C0={float(C0):.8f}  C0/sqrt(beta)={float(C0/mp.sqrt(beta)):.6f}  "
              f"C0/beta={float(C0/beta):.6f}")
        d3g, deg = d3c, dc

    print("\n=== descente depuis beta=0.02 (continuation) ===")
    d3g, deg = d3_ref, delta_ref
    for beta_s in ('0.019', '0.018', '0.016', '0.014', '0.012', '0.01',
                   '0.008', '0.006', '0.005'):
        beta = mp.mpf(beta_s)
        d3c, dc = localiser_pli(beta, d3g, deg)
        C0 = C0_de(d3c, dc, beta)
        print(f"  beta={float(beta):.4f}  delta_c={float(dc):.8f}  d3_c={float(d3c):.6e}  "
              f"C0={float(C0):.8f}  C0/sqrt(beta)={float(C0/mp.sqrt(beta)):.6f}  "
              f"C0/beta={float(C0/beta):.6f}")
        d3g, deg = d3c, dc
