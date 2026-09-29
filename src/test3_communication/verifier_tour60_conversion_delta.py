"""Tour 60 (29/09/2026, vraie critique de dipankarsarkar) : verification
INDEPENDANTE (regle 5bis) de ses chiffres.

(1) racines du systeme a trois lignes a pli-1e-5 et pli+1e-5, en coordonnee
    u = log(r3/r4) sur [-200, 200] : il annonce 5 racines (X3 = 0, 9,035,
    9,292, 43,585, 49,329) puis 3 (X3 = 0, 43,585, 49,328).
(2) pente du noeud dd3/ddelta a pli-1e-5, -1e-6, -1e-7, -1e-8, -1e-9 : il
    annonce 16, 54, 552 pour -1e-5, -1e-6, -1e-8.
(3) ses conversions "mean - median" en unites de delta : pour eps 1e-7 /
    1e-8 / 1e-10, les biais mesures a pli-1e-6 (+3,0e-7 / -5,8e-7 / -2,3e-6)
    donnent selon lui pli-5,6e-9 / pli+1,1e-8 / pli+4,3e-8 ; et pour eps
    1e-10 le decalage converti serait -5,0e-8 (biais -8e-7 a pli-1e-5),
    -4,3e-8 (-2,3e-6 a pli-1e-6), -2,0e-8 (-8e-6 a pli-1e-8).
"""

import numpy as np

from verifier_tour59_branches_fermees import branche_et_jumeau, mesures, F_PLI
from verifier_tour59_recomptage_racines import G

B = 0.02


def racines(d):
    U = np.linspace(-200, 200, 4000001)
    g = G(U, d)
    i = np.where(np.sign(g[:-1]) != np.sign(g[1:]))[0]
    out = []
    for k in i:
        a, b = U[k], U[k + 1]
        for _ in range(100):
            m = 0.5 * (a + b)
            a, b = (m, b) if np.sign(G(m, d)) == np.sign(G(a, d)) else (a, m)
        u = 0.5 * (a + b)
        out.append((u, (1 - d) / (1 + np.exp(-u)) / B, 1 / (1 + np.exp(-u)), 1 / (1 + np.exp(u))))
    return out


def pente_noeud(off):
    h = min(abs(off) * 1e-3, 1e-12) if abs(off) < 1e-9 else abs(off) * 1e-3
    d = F_PLI + off
    v = []
    for dd in (d - h, d + h):
        rs = branche_et_jumeau(dd)
        v.append(mesures(rs[0], dd)[0])
    return (v[1] - v[0]) / (2 * h)


if __name__ == "__main__":
    for off in (-1e-5, +1e-5):
        r = racines(F_PLI + off)
        print(f"pli{off:+.0e} : {len(r)} racines, X3 = " + ", ".join(f"{x:.3f}" for _, x, _, _ in r)
              + f"   (r3 min = {min(a for _, _, a, _ in r):.2e}, r4 min = {min(b for _, _, _, b in r):.2e})")
    print("\npente du noeud dd3/ddelta (ma fermeture) ; lui : 16 / 54 / 552 a -1e-5 / -1e-6 / -1e-8")
    pentes = {}
    for off in (-1e-5, -1e-6, -1e-7, -1e-8, -1e-9):
        pentes[off] = pente_noeud(off)
        print(f"   pli{off:+.0e} : {pentes[off]:9.2f}")
    print("\nconversions du biais de d3 en unites de delta (biais / pente) :")
    cas = [
        ("eps 1e-7  mean-median a pli-1e-6", +3.0e-7, -1e-6, "lui : pli - 5,6e-9"),
        ("eps 1e-8  mean-median a pli-1e-6", -5.8e-7, -1e-6, "lui : pli + 1,1e-8"),
        ("eps 1e-10 mean-median a pli-1e-6", -2.3e-6, -1e-6, "lui : pli + 4,3e-8"),
        ("eps 1e-10 biais moyen   a pli-1e-5", -7.95e-7, -1e-5, "lui : delta - 5,0e-8"),
        ("eps 1e-10 biais moyen   a pli-1e-6", -2.263e-6, -1e-6, "lui : delta - 4,3e-8"),
        ("eps 1e-10 biais moyen   a pli-1e-8", -8.034e-6, -1e-8, "lui : delta - 2,0e-8"),
    ]
    for nom, biais, off, lui in cas:
        s = biais / pentes[off]
        print(f"   {nom:38s} biais {biais:+.3e} / pente {pentes[off]:8.2f} = {s:+.3e}   ({lui})")
