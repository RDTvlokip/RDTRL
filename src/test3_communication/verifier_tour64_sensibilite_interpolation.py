"""Tour 64 : le facteur 10,9 que "L = F(r)" exigerait sur sqrt(v3) depend-il de l'interpolation de L
entre mes lignes eps 1e-9 (0,96620) et 3e-9 (0,95593) ? Il dit lui-meme : "L falls 0.010 between 1e-9 and 3e-9,
so 1.09e-9 is coarse". La borne qu'il declare independante de l'interpolation : L(lr 0,02) = 0,96540 est
SOUS L(lr 0,05, eps 1e-9) = 0,96620, donc l'eps equivalent est > 1e-9 quelle que soit la forme de la courbe,
et le facteur exige est > 1e-9/1e-10 = 10.
Ici : (1) la borne, (2) eps equivalent pour une interpolation lineaire en log eps, lineaire en eps, en
racine carree, en 1/eps, (3) l'incertitude de mesure sur L (ES entre phases) propagee sur l'eps equivalent.
"""

import math
import numpy as np

E = np.array([1e-9, 3e-9]); L = np.array([0.96620, 0.95593]); CIBLE = 0.96540
SE_L = 2e-5  # ordre de grandeur de l'ES entre phases sur L (rapport de deux medianes, 16 phases)


def eps_eq(forme):
    f = (L[0] - CIBLE) / (L[0] - L[1])
    if forme == "log":
        return 10 ** (math.log10(E[0]) + f * (math.log10(E[1]) - math.log10(E[0])))
    if forme == "lineaire":
        return E[0] + f * (E[1] - E[0])
    if forme == "racine":
        return (math.sqrt(E[0]) + f * (math.sqrt(E[1]) - math.sqrt(E[0]))) ** 2
    if forme == "inverse":
        return 1.0 / (1.0 / E[0] + f * (1.0 / E[1] - 1.0 / E[0]))


if __name__ == "__main__":
    print(f"borne : L(lr 0,02) = {CIBLE} < L(lr 0,05, eps 1e-9) = {L[0]} -> eps equivalent > 1e-9 -> facteur > 10 (quelle que soit la forme)")
    for forme in ("log", "lineaire", "racine", "inverse"):
        e = eps_eq(forme)
        print(f"  interpolation {forme:9s}: eps equivalent = {e:.3e}  facteur sur sqrt(v3) = {e / 1e-10:.2f}")
    for d in (-SE_L, +SE_L):
        f = (L[0] - (CIBLE + d)) / (L[0] - L[1])
        print(f"  L(lr 0,02) {'-' if d < 0 else '+'}{SE_L:g} : facteur (log) = {10 ** (math.log10(E[0]) + f * (math.log10(E[1]) - math.log10(E[0]))) / 1e-10:.2f}")
    print("\nrapport MESURE de la mediane de sqrt(v_e3) : 4,6509e-8 / 1,9090e-8 =", round(4.6509e-8 / 1.9090e-8, 3))
