"""RDTRL — la variance a MISES A JOUR FIXES, l'axe que ni lui ni moi n'avons balaye.

Il tient les tirages fixes et fait varier les mises a jour. J'ai tenu les mises a
jour fixes et fait varier les tirages, mais sur DEUX points seulement (lot 8 et
lot 64), ce qui ne distingue pas une monotonie d'un maximum interieur.

Les trois points deja acquis, tous a 20 000 mises a jour, disent que ce n'est pas
monotone :

    montee exacte   variance 0          0 / 12 bijections
    REINFORCE 64    variance 1,1e-03   11 / 12
    REINFORCE 8     variance 9,2e-03    3 / 12

Zero variance piege, trop de variance empeche de converger. Si c'est vrai, la
courbe a un maximum interieur, et nos deux recits sont chacun la bonne moitie.
Ce fichier la remplit : lot 8, 16, 32, 64, 128, 256 a 20 000 pas, memes graines,
meme lr, une seule chose qui bouge.

Variance mesuree separement, a theta fixe, par `variance_du_gradient.py`, et non
inferee du lot : elle decroit comme 1/lot mais le rapport se lit sur des nombres
mesures.
"""

import numpy as np

from grammaire3 import N
from iso_echantillons import GRAINES, cellule, entete, rendre
from representable_atteignable_stable import EmetteurTabulaire, Recepteur, monter
from variance_du_gradient import lois, mesurer

BETA = 0.02
LOTS = (8, 16, 32, 64, 128, 256)
PAS = 20000


def variance_au_piege(lot, generateur, replicats=20000):
    """Variance totale de l'estimateur au point piege, pour l'axe des abscisses."""
    g = np.random.default_rng(606)
    e, r = EmetteurTabulaire(g), Recepteur(g)
    monter(e, r, BETA, PAS, lr=0.05)
    s, rr = lois(e, r)
    p = float((s * rr.T).sum() / N)
    _, _, var, _ = mesurer(s, rr, lot, "ema", generateur, replicats, p, 0.0)
    return var


if __name__ == "__main__":
    print("=== LA VARIANCE A MISES A JOUR FIXES : 20 000 PAS PARTOUT ===")
    print("  memes graines, meme lr, meme objectif. Seul le lot change.")

    generateur = np.random.default_rng(777)
    print(f"\n  {'lot':>6}{'variance au piege':>20}{'x lot 64':>11}")
    variances = {}
    for lot in LOTS:
        variances[lot] = variance_au_piege(lot, generateur)
    for lot in LOTS:
        print(f"  {lot:>6}{variances[lot]:>20.4e}"
              f"{variances[lot] / variances[64]:>11.3f}")

    for lr in (0.01,):
        print(f"\n  lr = {lr}")
        entete()
        for lot in LOTS:
            rs, bj, cols = cellule(lot, PAS, lr)
            rendre(f"var {variances[lot]:.2e}", lot, PAS, lr, rs, bj, cols)
