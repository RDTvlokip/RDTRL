"""RDTRL — test 3 : sa ligne iso-echantillons, et la colonne LOO qu'il demande.

Sa critique du balayage precedent est juste : lot x pas fait varier la variance et
le volume ENSEMBLE, donc chaque cellule est compatible avec les deux recits. Deux
controles ici, dans le meme fichier :

  1. la ligne iso-echantillons a 160 000 tirages — lot 8 x 20 000, lot 16 x 10 000,
     lot 64 x 2 500 — plus lot 64 x 20 000 hors ligne comme ancre ;
  2. sa colonne : lot 8 x 20 000, memes graines, avec et sans ligne de reference
     leave-one-out dans le lot.

Ce que `variance_du_gradient.py` a mesure AVANT de lancer ceci, et qui contraint
la lecture de la colonne 2 : LOO n'abaisse pas la variance ici, il la MONTE
(x 1,02 a 1,20 selon le point), et il annule entierement 39 a 73 % des lots. Sa
colonne tient donc les tirages fixes, monte legerement la variance, et coupe les
mises a jour effectives de moitie. Elle est lancee quand meme parce qu'elle a ete
demandee et que le resultat vaut d'etre au dossier, pas parce qu'elle isole ce
qu'elle devait isoler.

`reinforce_variante` est verifie identique a `reinforce` canonique sur la meme
graine avant tout usage : la seule difference autorisee est la ligne de reference.
"""

import numpy as np
import torch

from grammaire3 import N
from representable_atteignable_stable import (EmetteurTabulaire, Recepteur,
                                              activer, lire_code, objectif,
                                              parametres, reinforce)

BETA = 0.02
GRAINES = 12


def reinforce_variante(emetteur, recepteur, beta, pas, lot=64, lr=0.01, graine=0,
                       ligne="ema", algo="adam"):
    """Copie mot pour mot de `reinforce`, ligne de reference et optimiseur en plus."""
    activer(emetteur, recepteur)
    if algo == "adam":
        optimiseur = torch.optim.Adam(parametres(emetteur, recepteur), lr=lr)
    else:
        optimiseur = torch.optim.SGD(parametres(emetteur, recepteur), lr=lr)
    g = torch.Generator().manual_seed(graine)
    baseline = 0.0
    for _ in range(pas):
        s, r = emetteur.loi(), recepteur.loi()
        referents = torch.randint(0, N, (lot,), generator=g)
        messages = torch.multinomial(s[referents], 1, generator=g).squeeze(1)
        reconstruits = torch.multinomial(r[messages], 1, generator=g).squeeze(1)
        recompenses = (reconstruits == referents).double()
        moyenne = float(recompenses.mean())
        if ligne == "ema":
            av = [float(x) - baseline for x in recompenses]
        elif ligne == "aucune":
            av = [float(x) for x in recompenses]
        else:
            somme = float(recompenses.sum())
            av = [(lot * float(x) - somme) / (lot - 1) for x in recompenses]
        avantages = torch.tensor(av, dtype=torch.float64)
        baseline = 0.9 * baseline + 0.1 * moyenne
        log_s = torch.log(s[referents, messages].clamp_min(1e-300))
        log_r = torch.log(r[messages, reconstruits].clamp_min(1e-300))
        entropie = -(s * torch.log(s.clamp_min(1e-300))).sum() / N \
                   - (r * torch.log(r.clamp_min(1e-300))).sum() / N
        perte = -((log_s + log_r) * avantages).mean() - beta * entropie
        optimiseur.zero_grad()
        perte.backward()
        optimiseur.step()
    with torch.no_grad():
        _, recompense = objectif(emetteur, recepteur, beta)
    return float(recompense)


def controle_identite():
    """La variante doit rendre le canonique bit pour bit a `ligne='ema'`."""
    g = np.random.default_rng(11)
    e1, r1 = EmetteurTabulaire(g), Recepteur(g)
    g = np.random.default_rng(11)
    e2, r2 = EmetteurTabulaire(g), Recepteur(g)
    a = reinforce(e1, r1, BETA, 400, lot=8, lr=0.01, graine=3)
    b = reinforce_variante(e2, r2, BETA, 400, lot=8, lr=0.01, graine=3, ligne="ema")
    print(f"  controle d'identite variante/canonique : "
          f"{a:.15f} vs {b:.15f}   ecart {abs(a - b):.2e}")
    assert abs(a - b) == 0.0, "la variante n'est pas le canonique"


def cellule(lot, pas, lr, ligne="ema", algo="adam"):
    g = np.random.default_rng(606)
    rs, bj, cols = [], 0, []
    for k in range(GRAINES):
        e, r = EmetteurTabulaire(g), Recepteur(g)
        rs.append(reinforce_variante(e, r, BETA, pas, lot=lot, lr=lr, graine=k,
                                     ligne=ligne, algo=algo))
        d = len(set(lire_code(e).tolist()))
        bj += (d == N)
        cols.append(N - d)
    return np.array(rs), bj, np.array(cols)


def rendre(etiquette, lot, pas, lr, rs, bj, cols, marque=""):
    print(f"  {etiquette:>22}{lot:>5}{pas:>8}{lot * pas:>10}{lr:>7.2f}"
          f"{rs.mean():>10.5f}{bj:>7}/{GRAINES}{cols.mean():>8.2f}"
          f"{rs.min():>10.5f}{rs.max():>10.5f}{marque}")


def entete():
    print(f"  {'ligne de reference':>22}{'lot':>5}{'pas':>8}{'tirages':>10}"
          f"{'lr':>7}{'E[R]':>10}{'biject':>9}{'colls':>8}"
          f"{'min E[R]':>10}{'max E[R]':>10}")


if __name__ == "__main__":
    print("=== CONTROLE DU CODE AVANT MESURE ===")
    controle_identite()

    print("\n=== 1. SA LIGNE ISO-ECHANTILLONS : 160 000 TIRAGES ===")
    print("  tirages fixes, mises a jour variables d'un facteur 8.")
    entete()
    for lr in (0.01, 0.05):
        for lot, pas in ((8, 20000), (16, 10000), (64, 2500), (64, 20000)):
            rs, bj, cols = cellule(lot, pas, lr)
            marque = "   <- hors ligne" if lot * pas != 160000 else ""
            rendre("ema (canonique)", lot, pas, lr, rs, bj, cols, marque)
        print()

    print("=== 2. SA COLONNE : LOT 8 x 20 000, AVEC ET SANS LOO ===")
    print("  tirages fixes, mises a jour nominales fixes, memes graines.")
    entete()
    for lr in (0.01, 0.05):
        for ligne in ("ema", "loo", "aucune"):
            rs, bj, cols = cellule(8, 20000, lr, ligne=ligne)
            rendre(ligne, 8, 20000, lr, rs, bj, cols)
        print()
