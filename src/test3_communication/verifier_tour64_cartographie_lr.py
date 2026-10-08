"""Tour 64 (vraie critique de dipankarsarkar) : verification de ses chiffres a partir de MES lignes.

Colonne lr 0,05 (reduction, x = 1e-13) : eps -> L, D_d/sd, |D_d|, sd ; ligne lr 0,02 (eps 1e-10).
Cartographie "lineaire en log eps" de la ligne lr 0,02 sur la colonne lr 0,05 : eps equivalent
pour L, pour D_d/sd, pour |D_d| ; puis le facteur sur sqrt(v3) qu'exigerait L = F(r), r = eps/sqrt(v3),
et l'exposant q tel que sqrt(v3) ~ sd^q reproduit ce facteur.
Lignes (tour 63) : eps 1e-10 / 1e-9 / 3e-9 / 1e-8 a lr 0,05 ; eps 1e-10 a lr 0,02.
"""

import math
import numpy as np

COL = {  # eps: (L, D_d/sd, |D_d|, sd)
    1e-10: (0.96799, 0.277, 6.0837e-6, 2.1995e-5),
    1e-9: (0.96620, 0.411, 7.1539e-6, 1.7417e-5),
    3e-9: (0.95593, 0.672, 8.3244e-6, 1.2380e-5),
    1e-8: (0.94988, 1.356, 8.6313e-6, 6.3665e-6),
}
LR02 = (0.96540, 0.963, 5.8490e-6, 6.0720e-6)


def eps_equiv(valeur, idx):
    """eps de la colonne lr 0,05 qui reproduit `valeur`, interpolation lineaire en log10(eps) ; None hors plage."""
    es = sorted(COL)
    for a, b in zip(es, es[1:]):
        va, vb = COL[a][idx], COL[b][idx]
        if min(va, vb) <= valeur <= max(va, vb):
            f = (valeur - va) / (vb - va)
            return 10 ** (math.log10(a) + f * (math.log10(b) - math.log10(a)))
    return None


if __name__ == "__main__":
    print("monotonie de la colonne lr 0,05 en eps : L baisse ?", all(COL[a][0] > COL[b][0] for a, b in zip(sorted(COL), sorted(COL)[1:])),
          " ; |D_d| monte ?", all(COL[a][2] < COL[b][2] for a, b in zip(sorted(COL), sorted(COL)[1:])))
    print(f"lr 0,02 : L {LR02[0]} (colonne : {COL[1e-10][0]} a eps 1e-10) -> {'plus bas = r plus grand' if LR02[0] < COL[1e-10][0] else 'plus haut'} ;"
          f" |D_d| {LR02[2]:.3e} (colonne : {COL[1e-10][2]:.3e}) -> {'plus bas = r plus petit' if LR02[2] < COL[1e-10][2] else 'plus haut'}")
    for nom, idx, val in (("L", 0, LR02[0]), ("D_d/sd", 1, LR02[1]), ("|D_d|", 2, LR02[2])):
        e = eps_equiv(val, idx)
        print(f"  {nom:7s} lr 0,02 = {val:.5g} -> eps equivalent a lr 0,05 : " + (f"{e:.3g}" if e else "aucun (hors de la colonne)"))
    e = eps_equiv(LR02[0], 0)
    facteur = e / 1e-10
    sd_eq = 10 ** np.interp(math.log10(e), [math.log10(k) for k in sorted(COL)], [math.log10(COL[k][3]) for k in sorted(COL)])
    ratio_sd = sd_eq / LR02[3]
    print(f"\nL = F(r) exige sqrt(v3)(lr 0,05, eps {e:.3g}) / sqrt(v3)(lr 0,02, eps 1e-10) = eps_eq/1e-10 = {facteur:.2f}")
    print(f"sd sature a (lr 0,05, eps {e:.3g}) (log-interpole) = {sd_eq:.3e} ; sd(lr 0,02) = {LR02[3]:.3e} ; rapport {ratio_sd:.2f}")
    print(f"exposant q (sqrt(v3) ~ sd^q) qui donne {facteur:.2f} avec un rapport de sd {ratio_sd:.2f} : q = {math.log(facteur) / math.log(ratio_sd):.2f}")
    # prediction de L(lr 0,02) pour q = 1 et q = 2 : eps equivalent = 1e-10 * ratio_sd^q
    for q in (1, 2):
        eq = 1e-10 * ratio_sd ** q
        es = sorted(COL)
        L = np.interp(math.log10(eq), [math.log10(k) for k in es], [COL[k][0] for k in es])
        print(f"  q = {q} : eps equivalent {eq:.3g} -> L predit {L:.5f} (mesure {LR02[0]})")
