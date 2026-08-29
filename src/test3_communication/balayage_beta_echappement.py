"""RDTRL — sa question decisive du tour vingt-neuf : le taux d'echappement de
REINFORCE contre la montee exacte depend-il de beta ?

Si le decrochage 92 % contre 5 % (§7.36) vient du terme d'entropie qui laisse
une rangee ouverte plutot que d'un bassin plat que seul le bruit peut fuir,
alors l'ecart doit se refermer a beta = 0 : sans bonus d'entropie, rien ne
retient un referent perdant indecis, donc le bruit de REINFORCE ne devrait
plus avoir d'avantage particulier sur la montee exacte.

Beta in {0, 0,005, 0,02}. Meme protocole que echappement_du_piege.py mais
depuis l'ALEATOIRE (pas depuis un piege deja atteint), pour mesurer le taux de
bijection complet de chaque dynamique, comme au tour 19/20 (§7.35quinquies,
§7.36).
"""

import numpy as np
import torch

from grammaire3 import N
from iso_echantillons import reinforce_variante
from representable_atteignable_stable import EmetteurTabulaire, Recepteur, monter, lire_code

PAS = 20000
GRAINES = 10
BETAS = (0.0, 0.005, 0.02)


def cellule_exacte(beta, graine_base):
    g = np.random.default_rng(graine_base)
    bijections = 0
    collisions = []
    for k in range(GRAINES):
        e, r = EmetteurTabulaire(g), Recepteur(g)
        monter(e, r, beta, PAS, lr=0.05)
        code = lire_code(e)
        d = len(set(code.tolist()))
        bijections += (d == N)
        collisions.append(N - d)
    return bijections, np.mean(collisions)


def cellule_reinforce(beta, graine_base):
    g = np.random.default_rng(graine_base)
    bijections = 0
    collisions = []
    for k in range(GRAINES):
        e, r = EmetteurTabulaire(g), Recepteur(g)
        reinforce_variante(e, r, beta, PAS, lot=64, lr=0.01, graine=k, ligne="ema")
        code = lire_code(e)
        d = len(set(code.tolist()))
        bijections += (d == N)
        collisions.append(N - d)
    return bijections, np.mean(collisions)


if __name__ == "__main__":
    print("=== BALAYAGE BETA : L'ECART 92%/5% SE REFERME-T-IL A BETA=0 ? ===")
    print(f"  {GRAINES} graines par cellule, {PAS} pas, lr 0,05 (exact) / 0,01 (REINFORCE lot 64)\n")
    print(f"  {'beta':>8}{'exact bij.':>12}{'exact colls':>13}"
          f"{'REINFORCE bij.':>16}{'REINFORCE colls':>17}")
    for beta in BETAS:
        be, ce = cellule_exacte(beta, 606)
        br, cr = cellule_reinforce(beta, 606)
        print(f"  {beta:>8.4f}{be:>9} /{GRAINES}{ce:>13.2f}"
              f"{br:>13} /{GRAINES}{cr:>17.2f}")
