"""Tour 59 : recomptage COMPLET des points fixes du systeme de dipankar
(C=26), en coordonnee u = log(r3/r4) sur [-200, 200], parce que la grille
en r3 de verifier_tour59_pli_concurrents.py s'arretait a r3 = 1e-14 et
ratait la racine effondree (r3 ~ 1e-21). Resultat attendu : 5 racines
sous le pli, 3 au-dessus. Imprime aussi le pli en pleine precision."""

import numpy as np
from verifier_tour59_pli_concurrents import delta_c

B = 0.02


def G(u, d, C=26):
    r3 = 1 / (1 + np.exp(-u)); r4 = 1 / (1 + np.exp(u))
    s3 = 1 / (1 + C * np.exp(-(1 - d) * r3 / B)); s4 = 1 / (1 + C * np.exp(-(1 + d) * r4 / B))
    return -u - ((1 + d) * s4 - (1 - d) * s3) / B


if __name__ == "__main__":
    U = np.linspace(-200, 200, 4000001)
    for d in (0.0, 0.002, 0.0134, 0.01344, 0.02):
        g = G(U, d)
        i = np.where(np.sign(g[:-1]) != np.sign(g[1:]))[0]
        print(f"delta={d:<8} {len(i)} racines (u, X3) :",
              [(round(U[k], 3), round((1 - d) / (1 + np.exp(-U[k])) / B, 5)) for k in i])
    lo, hi, _, _ = delta_c(26, 0.0134, 0.0135)
    print(f"pli C=26 en pleine precision : ({lo!r}, {hi!r})")
