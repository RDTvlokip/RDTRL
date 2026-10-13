"""Tour 67 : zero de D_d en eps/lr selon la reference choisie. D_d(noeud ferme) mesure (verifier_tour67_zero_dd.py, x = 1e-8) ;
D_d' = D_d - Delta, avec Delta = d3(point fixe de la reduction) - d3(noeud ferme) = -6,0557e-9 (verifier_tour67_point_fixe_reduction.py)
ou Delta = +1,0e-8 (l'hypothese de D0 comme erreur de reference de signe positif, refutee). Interpolation lineaire de D_d en q.
"""

import numpy as np

# lr -> (q en 1e-7, D_d en 1e-9) ; lignes recopiees des sorties du 14/10/2026
DATA = {
    0.005: ([4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 5.9, 6.0, 6.2, 6.5], [0.29683, 1.7370, 3.1213, 4.4482, 5.7201, 6.9378, 8.0996, 8.6544, 9.1962, 10.214, 11.621]),
    0.01: ([5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.85, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5], [-11.416, -8.8044, -6.2794, -3.7789, -1.3455, 1.0260, 2.1966, 3.3451, 5.5795, 7.7489, 9.8140, 11.525, 13.568, 15.640]),
    0.02: ([5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5], [-18.565, -9.3236, -0.34979, 8.2468, 15.310, 23.824, 32.402]),
    0.05: ([5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5], [-170.35, -112.10, -54.633, 0.53655, 47.788, 104.87, 163.20]),
    0.1: ([5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5], [-627.78, -408.06, -181.63, 28.626, 259.97, 512.14, 788.63]),
}


def zero(q, d):
    q = np.array(q); d = np.array(d)
    for i in range(len(q) - 1):
        if d[i] < 0 <= d[i + 1]:
            return q[i] + (q[i + 1] - q[i]) * (0 - d[i]) / (d[i + 1] - d[i])
    return None


if __name__ == "__main__":
    refs = (("noeud ferme (Delta = 0)", 0.0), ("point fixe propre (Delta = -6,06e-9)", -6.0557), ("Delta = +10e-9 (hypothese refutee)", 10.0))
    print("zero de D_d - Delta, en eps/lr (1e-7) ; 'hors grille' = pas de changement de signe sur la grille mesuree")
    print(f"{'lr':>6s}  " + "  ".join(f"{n:>38s}" for n, _ in refs))
    for lr, (q, d) in DATA.items():
        cols = []
        for n, delta in refs:
            z = zero(q, np.array(d) - delta)
            cols.append("hors grille" if z is None else f"{z:.3f}")
        print(f"{lr:6.3f}  " + "  ".join(f"{c:>38s}" for c in cols))
