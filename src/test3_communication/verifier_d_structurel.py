"""Verification independante de la critique de l'agent-dipankar
(17/09/2026, verifier_puiseux_ordre_suivant.py) : D~8,0021 est-il une
coincidence numerique a (beta=0,02, N=27) ou une constante structurelle
du pli ? L'agent affirme l'avoir teste par balayage beta ET N ("26"
generalise a N_autres), trouvant une derive continue (4,86 a 12,4 en
beta ; 7,5 a 8,6 en N) -- PAS pris pour acquis, refait ici de zero.

Generalise aussi le "26" de l'equation H7 en N_autres (parametre libre),
pour le balayage en N -- verifie AU PASSAGE que N_autres=26 (le cas
publie) reproduit bien D=8,0021212.
"""

import sys
sys.path.insert(0, '.')
import mpmath as mp

mp.mp.dps = 50


def h(d3, delta, beta, n_autres):
    R = 1 / (1 + mp.e ** (-(2 * delta + (1 - delta) * d3) / beta))
    return d3 / (n_autres * (1 - d3)) - mp.e ** (-(1 - delta) * (1 - R) / beta)


def localiser_pli(beta, n_autres, d3_guess, delta_guess):
    d3, delta = mp.mpf(d3_guess), mp.mpf(delta_guess)
    for _ in range(100):
        F = mp.matrix([h(d3, delta, beta, n_autres),
                        mp.diff(lambda x: h(x, delta, beta, n_autres), d3)])
        Jd3 = mp.matrix([
            mp.diff(lambda x: h(x, delta, beta, n_autres), d3),
            mp.diff(lambda x: mp.diff(lambda y: h(y, delta, beta, n_autres), x), d3),
        ])
        Jde = mp.matrix([
            mp.diff(lambda de: h(d3, de, beta, n_autres), delta),
            mp.diff(lambda de: mp.diff(lambda y: h(y, de, beta, n_autres), d3), delta),
        ])
        J = mp.matrix([[Jd3[0], Jde[0]], [Jd3[1], Jde[1]]])
        dv = mp.lu_solve(J, -F)
        d3 += dv[0]
        delta += dv[1]
        if max(abs(dv[0]), abs(dv[1])) < mp.mpf('1e-35'):
            break
    return d3, delta


def coefficients(d3c, dc, beta, n_autres):
    Fh = lambda x, de: h(x, de, beta, n_autres)
    F_xx = mp.diff(lambda x: Fh(x, dc), d3c, 2)
    F_xxx = mp.diff(lambda x: Fh(x, dc), d3c, 3)
    F_xxxx = mp.diff(lambda x: Fh(x, dc), d3c, 4)
    F_eps = -mp.diff(lambda de: Fh(d3c, de), dc)
    F_xeps = -mp.diff(lambda x: mp.diff(lambda de: Fh(x, de), dc), d3c)
    F_xxeps = -mp.diff(lambda x: mp.diff(lambda de: Fh(x, de), dc), d3c, 2)
    F_epseps = mp.diff(lambda de: Fh(d3c, de), dc, 2)

    A = -2 * F_eps / F_xx
    a1 = mp.sqrt(A)
    C0 = 2 * a1
    a2 = -(F_xeps + (F_xxx / 6) * a1**2) / F_xx
    reste = ((F_xxx / 2) * a1**2 * a2 + (F_xxxx / 24) * a1**4
             + F_xeps * a2 + (F_xxeps / 2) * a1**2 + F_epseps / 2)
    a3 = -(reste + F_xx * (a2**2 / 2)) / (F_xx * a1)
    D = 2 * a3
    return C0, D


if __name__ == "__main__":
    d3_ref = mp.mpf('0.0027251196899644737')
    delta_ref = mp.mpf('0.013437210066097283')
    beta0 = mp.mpf('0.02')
    n0 = mp.mpf('26')

    print("=== D(beta), N_autres=26 fixe, continuation ===")
    d3g, deg = d3_ref, delta_ref
    for beta_s in ('0.018', '0.02', '0.022', '0.025', '0.03', '0.04'):
        beta = mp.mpf(beta_s)
        try:
            d3c, dc = localiser_pli(beta, n0, d3g, deg)
            C0, D = coefficients(d3c, dc, beta, n0)
            print(f"  beta={float(beta):.4f}  delta_c={float(dc):.8f}  C0={float(C0):.6f}  D={float(D):.6f}")
            d3g, deg = d3c, dc
        except Exception as e:
            print(f"  beta={float(beta):.4f}  ECHEC: {e}")

    print("\n=== D(N_autres), beta=0,02 fixe, continuation ===")
    d3g, deg = d3_ref, delta_ref
    for n_s in ('26', '30', '40', '60', '100'):
        n_autres = mp.mpf(n_s)
        try:
            d3c, dc = localiser_pli(beta0, n_autres, d3g, deg)
            C0, D = coefficients(d3c, dc, beta0, n_autres)
            print(f"  N_autres={float(n_autres):.0f}  delta_c={float(dc):.8f}  C0={float(C0):.6f}  D={float(D):.6f}")
            d3g, deg = d3c, dc
        except Exception as e:
            print(f"  N_autres={float(n_autres):.0f}  ECHEC: {e}")

    print("\n=== Question de l'agent : Jacobien singulier pres de beta=0,015 ? ===")
    d3g, deg = d3_ref, delta_ref
    for beta_s in ('0.019', '0.018', '0.017', '0.016', '0.0155', '0.015', '0.0145', '0.014'):
        beta = mp.mpf(beta_s)
        try:
            d3c, dc = localiser_pli(beta, n0, d3g, deg)
            F_xx = mp.diff(lambda x: h(x, dc, beta, n0), d3c, 2)
            print(f"  beta={float(beta):.5f}  delta_c={float(dc):.8f}  d3_c={float(d3c):.6e}  F_xx={float(F_xx):.6f}")
            d3g, deg = d3c, dc
        except Exception as e:
            print(f"  beta={float(beta):.5f}  ECHEC: {e}")
