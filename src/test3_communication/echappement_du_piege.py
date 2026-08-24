"""RDTRL — test 3 : qui sort du piege, depuis LE MEME piege.

§7.36 comparait des runs partis de l'aleatoire. C'est un contraste sur toute la
trajectoire, donc il melange « ou l'on tombe » et « si l'on en sort ». L'evenement
qui porte la conclusion est l'echappement, et il se mesure la ou il a lieu : au
point critique lui-meme, atteint une fois, puis quitte par plusieurs dynamiques
depuis un etat identique au bit pres.

`variance_du_gradient.py` a etabli le fait qui rend cette mesure necessaire : au
point piege, ||grad E[R]|| vaut 3,1e-09 tandis que l'ecart-type du gradient
echantillonne vaut 9,6e-02 au lot 8. Le rapport variance/||grad||^2 y est de
9,3e+14, contre 2,4e+04 en milieu de montee. L'estimateur echantillonne n'est donc
pas « un peu bruite » au piege : il n'y porte presque aucun signal.

D'ou un quatrieme recit que ni lui ni moi n'avions nomme, a cote de bruit,
tirages et mises a jour : **le preconditionneur**. Adam normalise coordonnee par
coordonnee, donc il transforme un gradient sans signal en un pas de taille lr dans
une direction quelconque. Ce qui sort du piege pourrait etre Adam-sur-du-bruit et
non le bruit. Le contraste est direct : meme estimateur, meme point de depart,
Adam contre SGD, avec le deplacement moyen par pas imprime pour que les taux
d'apprentissage soient comparables.
"""

import numpy as np
import torch

from grammaire3 import N
from iso_echantillons import reinforce_variante
from representable_atteignable_stable import (EmetteurTabulaire, Recepteur,
                                              activer, cloner, lire_code,
                                              objectif, parametres)

BETA = 0.02
GRAINES = 12
PAS = 20000


def etat(emetteur, recepteur):
    with torch.no_grad():
        _, recompense = objectif(emetteur, recepteur, BETA)
    code = lire_code(emetteur)
    return float(recompense), N - len(set(code.tolist()))


def deplacement(agents_avant, emetteur, recepteur):
    with torch.no_grad():
        return float(sum(((t - a) ** 2).sum()
                         for t, a in zip(parametres(emetteur, recepteur),
                                         agents_avant)) ** 0.5)


def monter_suivi(emetteur, recepteur, pas, lr):
    """Montee exacte, avec le deplacement total en parametres."""
    activer(emetteur, recepteur)
    depart = [t.detach().clone() for t in parametres(emetteur, recepteur)]
    optimiseur = torch.optim.Adam(parametres(emetteur, recepteur), lr=lr)
    for _ in range(pas):
        j, _ = objectif(emetteur, recepteur, BETA)
        optimiseur.zero_grad()
        (-j).backward()
        optimiseur.step()
    return deplacement(depart, emetteur, recepteur)


def bras_reinforce(emetteur, recepteur, lot, lr, ligne, algo, graine):
    depart = [t.detach().clone() for t in parametres(emetteur, recepteur)]
    reinforce_variante(emetteur, recepteur, BETA, PAS, lot=lot, lr=lr,
                       graine=graine, ligne=ligne, algo=algo)
    return deplacement(depart, emetteur, recepteur)


BRAS = [
    ("montee exacte", None, 0.05, None, None),
    ("R lot 8  ema   adam", 8, 0.01, "ema", "adam"),
    ("R lot 8  loo   adam", 8, 0.01, "loo", "adam"),
    ("R lot 8  aucune adam", 8, 0.01, "aucune", "adam"),
    ("R lot 64 ema   adam", 64, 0.01, "ema", "adam"),
    ("R lot 64 ema   sgd", 64, 0.01, "ema", "sgd"),
    ("R lot 64 ema   sgd", 64, 0.10, "ema", "sgd"),
    ("R lot 64 ema   sgd", 64, 1.00, "ema", "sgd"),
]


if __name__ == "__main__":
    print("=== LE PIEGE, ATTEINT UNE FOIS, QUITTE PAR HUIT DYNAMIQUES ===")
    print(f"  {GRAINES} graines. Chaque piege = 20 000 pas de montee exacte a "
          "lr 0,05.")
    print("  Tous les bras repartent du MEME etat, clone parametre par parametre.")

    pieges = []
    generateur = np.random.default_rng(606)
    for k in range(GRAINES):
        e, r = EmetteurTabulaire(generateur), Recepteur(generateur)
        monter_suivi(e, r, PAS, 0.05)
        rec, colls = etat(e, r)
        pieges.append((e, r, rec, colls, generateur))

    recompenses = np.array([p[2] for p in pieges])
    collisions = np.array([p[3] for p in pieges])
    print(f"\n  les 12 pieges : E[R] {recompenses.mean():.5f}"
          f"  [{recompenses.min():.5f} ; {recompenses.max():.5f}]"
          f"   collisions {collisions.mean():.2f}"
          f"  [{collisions.min()} ; {collisions.max()}]")
    print(f"  bijectifs au depart : {int((collisions == 0).sum())} / {GRAINES}")

    print(f"\n  {'bras':>22}{'lr':>7}{'E[R] apres':>12}{'delta E[R]':>12}"
          f"{'biject':>9}{'colls':>8}{'sortis':>9}{'|dtheta|':>11}")
    for nom, lot, lr, ligne, algo in BRAS:
        finales, colls_fin, sorties, deplacements = [], [], 0, []
        for k, (e0, r0, rec0, colls0, gen) in enumerate(pieges):
            e = cloner(e0, np.random.default_rng(1000 + k))
            r = cloner(r0, np.random.default_rng(2000 + k))
            if lot is None:
                d = monter_suivi(e, r, PAS, lr)
            else:
                d = bras_reinforce(e, r, lot, lr, ligne, algo, graine=k)
            rec, colls = etat(e, r)
            finales.append(rec)
            colls_fin.append(colls)
            deplacements.append(d)
            sorties += (colls < colls0)
        finales = np.array(finales)
        colls_fin = np.array(colls_fin)
        print(f"  {nom:>22}{lr:>7.2f}{finales.mean():>12.5f}"
              f"{(finales - recompenses).mean():>+12.5f}"
              f"{int((colls_fin == 0).sum()):>6} /{GRAINES:<2}"
              f"{colls_fin.mean():>8.2f}{sorties:>6} /{GRAINES:<2}"
              f"{np.mean(deplacements):>11.4f}")
