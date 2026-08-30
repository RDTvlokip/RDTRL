"""RDTRL — extension du recensement : la confiance du CODEUR (S), pas
seulement celle du recepteur, pour verifier son croisement egalite/mur.

Il pretend : les 42 egalites ont S proche de 1 des deux cotes ; les 8 murs
ont un cote a l'entropie maximale exacte (S proche de 1/27, ecart max-min
de l'ordre de 1e-4). Si vrai, un mur n'est pas une collision du tout : c'est
un referent qui a cesse d'emettre, classe sous le message que le bruit
flottant lui assigne.
"""

import numpy as np
import torch

from grammaire3 import N
from representable_atteignable_stable import EmetteurTabulaire, Recepteur, monter, lire_code

BETA = 0.02
PAS = 20000
GRAINES = 30


def classer(e, r, code):
    with torch.no_grad():
        rr = r.loi()
        s = e.loi()
    uniques, comptes = np.unique(code, return_counts=True)
    doublons = uniques[comptes > 1]
    lignes = []
    for m in doublons:
        refs = np.where(code == m)[0]
        if len(refs) != 2:
            continue
        masses_r = sorted(float(rr[m, ref]) for ref in refs)
        a, b = masses_r
        classe = "egalite" if (0.1 < a < 0.9 and 0.1 < b < 0.9) else \
                 ("mur" if (a < 0.01 and b > 0.99) else "autre")
        s_max = [float(s[ref].max()) for ref in refs]
        lignes.append((classe, sorted(s_max)))
    return lignes


if __name__ == "__main__":
    print(f"=== CROISEMENT CONFIANCE DU CODEUR CONTRE CLASSE RECEPTEUR ===")
    print(f"  {GRAINES} graines, beta={BETA}, {PAS} pas\n")
    g = np.random.default_rng(31415)
    toutes = []
    for k in range(GRAINES):
        e, r = EmetteurTabulaire(g), Recepteur(g)
        monter(e, r, BETA, PAS, lr=0.05)
        code = lire_code(e)
        toutes.extend(classer(e, r, code))

    print(f"  {len(toutes)} collisions binaires classees\n")
    print(f"  {'classe':>10}{'n':>5}{'S max min':>14}{'S max max':>14}")
    for classe in ("egalite", "mur", "autre"):
        lot = [l for c, l in toutes if c == classe]
        if not lot:
            print(f"  {classe:>10}{0:>5}")
            continue
        mins = [min(l) for l in lot]
        print(f"  {classe:>10}{len(lot):>5}{min(mins):>14.6f}{max(mins):>14.6f}")

    print("\n  detail : pour chaque collision, la confiance S la plus BASSE des deux membres")
    print(f"  {'classe':>10}{'S le plus bas':>16}")
    for classe, l in toutes:
        print(f"  {classe:>10}{min(l):>16.6f}")
