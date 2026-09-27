"""Tour 59 (27/09/2026) : verification INDEPENDANTE (regle 5bis) du pli de
dipankarsarkar, a partir de ses trois lignes seulement :
  X3 = (1-d) r3/beta,  X4 = (1+d) r4/beta,  s_i = 1/(1 + C e^{-X_i}),
  log(r4/r3) = ((1+d) s4 - (1-d) s3)/beta,  r3 + r4 = 1,
avec C le nombre de concurrents sans recompense dans la ligne de chaque
collisionneur (26 dans le code a 27 messages).

Points fixes = racines en r3 de F(r3) = log((1-r3)/r3) - ((1+d)s4 - (1-d)s3)/beta.
Balayage fin de r3 (echelle log pres de 0 et de 1), comptage des
changements de signe, puis bissection de delta_c (nombre de racines
qui change) pour C = 25..29.
"""

import numpy as np
from mpmath import mp, mpf, exp, log, findroot

BETA = 0.02


def F(r3, d, C):
    r4 = 1 - r3
    s3 = 1 / (1 + C * np.exp(-(1 - d) * r3 / BETA))
    s4 = 1 / (1 + C * np.exp(-(1 + d) * r4 / BETA))
    return np.log(r4 / r3) - ((1 + d) * s4 - (1 - d) * s3) / BETA


GRILLE = np.unique(np.concatenate([np.logspace(-14, np.log10(0.5), 200000),
                                   1 - np.logspace(-14, np.log10(0.5), 200000)]))


def racines(d, C):
    f = F(GRILLE, d, C)
    idx = np.where(np.sign(f[:-1]) != np.sign(f[1:]))[0]
    out = []
    for i in idx:
        a, b = GRILLE[i], GRILLE[i + 1]
        for _ in range(80):
            m = 0.5 * (a + b)
            if np.sign(F(m, d, C)) == np.sign(F(a, d, C)):
                a = m
            else:
                b = m
        out.append(0.5 * (a + b))
    return out


def delta_c(C, lo=0.005, hi=0.03):
    n_lo = len(racines(lo, C))
    for _ in range(40):
        m = 0.5 * (lo + hi)
        if len(racines(m, C)) == n_lo:
            lo = m
        else:
            hi = m
    return lo, hi, n_lo, len(racines(hi, C))


if __name__ == "__main__":
    print("=== sa table : branche de collision (r3 le plus grand < 1/2 ... ) ===")
    for d in (0.002, 0.006, 0.010, 0.012, 0.0134, 0.01344):
        rs = racines(d, 26)
        X3 = [(1 - d) * r / BETA for r in rs]
        print(f"delta={d:<8} {len(rs)} racines, X3 = " + "  ".join(f"{x:.5f}" for x in X3))
    print("\n=== delta_c(C) ===")
    for C in (25, 26, 27, 28, 29):
        lo, hi, a, b = delta_c(C)
        print(f"C={C}  delta_c in ({lo:.7f}, {hi:.7f})   racines {a} -> {b}")
