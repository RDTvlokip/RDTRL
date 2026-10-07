"""Tour 62 (07/10/2026, vraie critique de dipankarsarkar) : le rapport
delta'_med(r4)/delta'_med(d3) des medianes de l'etat calme a eps 1e-10 se stabilise
vers ~0,968. Il propose 1 - rapport = c0 + c1 x^q (x = delta_c - delta), avec q libre
(limite 0,970) ou q = 0,47 (limite 0,967), et demande : a pli-1e-13, le rapport est-il
plus pres de 0,9687 ou de 0,9668 ?

Usage : python verifier_tour62_ratio_limite.py mesure <offsets...>   -> rapports (moy, ES)
        python verifier_tour62_ratio_limite.py ajuste                -> ses deux ajustements sur
                                                                      les lignes de ratios_tour62.txt
Instrument : reduction numba (reduction_numba_tour59), 24 phases x 2 000 000 pas apres 10 000
de mise en regime. Les medianes sont exactes par coordonnee (d3 monotone en z3, r4 monotone en u).
Affiche aussi d4 = 1 - s4 au debut de la trace (l'etat de la ligne 4 dans la reduction).
"""

import sys
import numpy as np
from mpmath import mp, mpf

sys.path.insert(0, ".")
import reduction_numba_tour59 as redlib
from hasard_reduction_tour59 import base_state, warmed
from verifier_tour61_decomposition_d3 import PLI, d3_noeud, r4_noeud, pente_fn

mp.dps = 50
FICHIER = "D:/tmp/rdtrl_tour62_ratios.txt"
NPH = 24


def mesure(off):
    ks = [int(a) for a in np.random.default_rng(7).integers(0, 4444, NPH)]
    s1 = base_state()
    rng = np.random.default_rng(11)
    delta = float(PLI + mpf(off))
    dn, rn = float(d3_noeud(PLI + mpf(off))), float(r4_noeud(PLI + mpf(off)))
    sl3, sl4 = pente_fn(d3_noeud, PLI + mpf(off)), pente_fn(r4_noeud, PLI + mpf(off))
    rap, esc, d4s = [], 0, []
    for k in ks:
        w = warmed(s1, k, rng, 1e-9)
        redlib.avancer_obs(w, delta, 10000, np.zeros((1, 7)))
        z4 = float(w["x"][2] - w["x"][3])
        d4s.append(26.0 * np.exp(-z4) / (1.0 + 26.0 * np.exp(-z4)))
        rec = np.zeros((2_000_000, 7))
        first, maxrb, rb, done = redlib.avancer_obs(w, delta, 2_000_000, rec)
        if maxrb > 0.9:
            esc += 1
            continue
        ex = 26.0 * np.exp(-rec[:, 1]); d3 = ex / (1.0 + ex)
        r4 = 1.0 / (1.0 + np.exp(rec[:, 2]) + w["Se"])
        rap.append(((np.median(r4) - rn) / sl4) / ((np.median(d3) - dn) / sl3))
    rap = np.array(rap)
    return rap.mean(), rap.std(ddof=1) / np.sqrt(len(rap)), len(rap), esc, float(np.median(d4s))


def ajuste():
    L = [l.split() for l in open(FICHIER) if l.strip() and not l.startswith("#")]
    # argument optionnel : x maximal retenu (ex. 3e-10 pour ses six lignes -3e-10 ... -1e-12) et
    # plancher d'erreur standard (les ES de 1e-5 donnent des chi2 enormes : il les ajuste a 3 chiffres)
    xmax = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    plancher = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    L = [a for a in L if abs(float(a[0])) <= xmax]
    x = np.array([abs(float(a[0])) for a in L]); r = np.array([float(a[1]) for a in L])
    se = np.maximum(np.array([float(a[2]) for a in L]), plancher)
    print("lignes utilisees :")
    for xi, ri, si in zip(x, r, se):
        print(f"   x = {xi:.0e}   rapport {ri:.5f} +- {si:.5f}")
    y = 1.0 - r
    from scipy.optimize import least_squares

    def modele(p, xx, q=None):
        # x en unites de 1e-9 : c1 reste de l'ordre de 1e-2 a 1 au lieu de 1e2 a 1e3
        c0, c1 = p[0], p[1]
        qq = p[2] if q is None else q
        return c0 + c1 * (xx / 1e-9) ** qq

    # q libre : 3 parametres, plusieurs departs, bornes larges
    best = None
    for q0 in (0.05, 0.15, 0.25, 0.4, 0.6):
        for c10 in (0.005, 0.02, 0.1):
            f = lambda p: (modele(p, x) - y) / se
            s = least_squares(f, [0.03, c10, q0], bounds=([-1, -100, 0.01], [1, 100, 2.0]), x_scale=[0.01, 0.01, 0.1])
            if best is None or s.cost < best.cost:
                best = s
    c0, c1, q = best.x
    print(f"\nq libre : 1 - rapport = {c0:.5f} + {c1:.5f} (x/1e-9)^{q:.3f}   limite rapport = {1 - c0:.5f}   chi2 = {2 * best.cost:.2f} ({len(x) - 3} ddl)")
    f2 = lambda p: (modele(p, x, 0.47) - y) / se
    s2 = least_squares(f2, [0.03, 0.02], x_scale=[0.01, 0.01])
    c0b, c1b = s2.x
    print(f"q = 0,47 : 1 - rapport = {c0b:.5f} + {c1b:.5f} (x/1e-9)^0.47   limite rapport = {1 - c0b:.5f}   chi2 = {2 * s2.cost:.2f} ({len(x) - 2} ddl)")
    for xi, ri in zip(x, r):
        print(f"   x={xi:.0e}  mesure {ri:.5f}  q libre {1 - modele(best.x, xi):.5f}  q=0,47 {1 - modele([c0b, c1b], xi, 0.47):.5f}")
    # Developpement en sqrt(x) : le biais des salves SATURE (distance fixe au point de pli) pendant que le noeud de
    # reference bouge comme sqrt(x) ; 1 - rapport = c0 + c1 sqrt(x) + c2 x (q = 1/2 exactement, plus la correction suivante)
    X = np.sqrt(x / 1e-9)
    A = np.column_stack([np.ones_like(X), X, X ** 2]) / se[:, None]
    for nom, cols in (("c0 + c1 sqrt(x)", [0, 1]), ("c0 + c1 sqrt(x) + c2 x", [0, 1, 2])):
        coef, *_ = np.linalg.lstsq(A[:, cols], y / se, rcond=None)
        chi2 = float(np.sum(((A[:, cols] @ coef) - y / se) ** 2))
        pred = {xp: 1 - sum(coef[i] * (np.sqrt(xp / 1e-9) ** cols[i]) for i in range(len(cols))) for xp in (1e-13, 3e-14, 1e-14)}
        print(f"developpement {nom:24s}: coef (c0, c1, c2) = {', '.join(f'{c:.5f}' for c in coef)} ; limite rapport = {1 - coef[0]:.5f}  chi2 = {chi2:.2f} ({len(x) - len(cols)} ddl)  "
              + "  ".join(f"x={xp:.0e}: {v:.5f}" for xp, v in pred.items()))
    for xp in (1e-13, 1e-14):
        print(f"prediction rapport a pli-{xp:.0e} : q libre {1 - modele(best.x, xp):.5f}   q=0,47 {1 - modele([c0b, c1b], xp, 0.47):.5f}")


if __name__ == "__main__":
    if sys.argv[1] == "mesure":
        for off in sys.argv[2:]:
            m, se, n, esc, d4 = mesure(off)
            print(f"{off:8s}  rapport {m:.5f} +- {se:.5f}   ({n} phases, {esc} echappees)   d4 median au debut de la trace = {d4:.3e}", flush=True)
            with open(FICHIER, "a") as fh:
                fh.write(f"{off} {m:.6f} {se:.6f}\n")
    else:
        ajuste()
