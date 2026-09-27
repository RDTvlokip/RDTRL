"""Tour 59 : branche stable (collision) et jumeau instable du pli, en
variables mesurables (d3 = 1-s3, R_b = s4 r4), pour comparer aux traces
Adam juste sous le pli. Meme systeme que verifier_tour59_pli_concurrents.py,
en coordonnee u = log(r3/r4) (aucune racine perdue pres de 0 ou 1)."""

import sys
import numpy as np

B, C = 0.02, 26
F_PLI = 0.013437210041901


def mesures(u, d):
    r3 = 1 / (1 + np.exp(-u)); r4 = 1 - r3
    s3 = 1 / (1 + C * np.exp(-(1 - d) * r3 / B)); s4 = 1 / (1 + C * np.exp(-(1 + d) * r4 / B))
    return 1 - s3, s4 * r4, r4


def G(u, d):
    r3 = 1 / (1 + np.exp(-u)); r4 = 1 / (1 + np.exp(u))
    s3 = 1 / (1 + C * np.exp(-(1 - d) * r3 / B)); s4 = 1 / (1 + C * np.exp(-(1 + d) * r4 / B))
    return -u - ((1 + d) * s4 - (1 - d) * s3) / B


def branche_et_jumeau(d):
    """Les deux racines qui fusionnent au pli (u entre -3 et 1)."""
    U = np.linspace(-3, 1, 400001)
    g = G(U, d)
    out = []
    for i in np.where(np.sign(g[:-1]) != np.sign(g[1:]))[0]:
        a, b = U[i], U[i + 1]
        for _ in range(100):
            m = 0.5 * (a + b)
            a, b = (m, b) if np.sign(G(m, d)) == np.sign(G(a, d)) else (a, m)
        out.append(0.5 * (a + b))
    return sorted(out, reverse=True)  # [collision (u plus grand), jumeau]


if __name__ == "__main__":
    offs = [float(x) for x in sys.argv[1:]] or [-1e-5, -1e-6, -1e-7, -3e-8, -1e-8]
    for o in offs:
        d = F_PLI + o
        rs = branche_et_jumeau(d)
        (dA, RA, rA), (dJ, RJ, rJ) = mesures(rs[0], d), mesures(rs[1], d)
        print(f"delta=pli{o:+.0e}  stable d3={dA:.6e} R_b={RA:.6f} r4={rA:.6f}   "
              f"jumeau d3={dJ:.6e} R_b={RJ:.6f} r4={rJ:.6f}   ecart d3={dJ - dA:.3e}  ecart r4={rJ - rA:.3e}")
