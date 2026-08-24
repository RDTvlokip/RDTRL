"""RDTRL — test 3 : est-ce le bruit, ou est-ce CE bruit-la.

Le mot « bruit » a servi de mecanisme pendant trois tours sans que personne
demande de quelle loi. Le bruit de REINFORCE n'est pas isotrope : a chaque pas il
vit sur les lignes de P indexees par les referents tires et sur les lignes de Q
indexees par les messages tires, soit au plus `lot` lignes sur 27, et sa direction
dans chaque ligne est celle du score. C'est un bruit epars et structure.

Le contraste qui separe « du bruit » de « ce bruit-la » : montee EXACTE plus un
bruit gaussien isotrope de MEME variance totale, mesuree au meme point et
reactualisee pendant la descente. Meme optimiseur, meme lr, meme depart, meme
budget.

  - s'il sort comme REINFORCE, le mecanisme est la magnitude, et le mot « bruit »
    suffisait ;
  - s'il ne sort pas, ce qui sort est la STRUCTURE de l'estimateur, et ni lui ni
    moi n'avions nomme la bonne chose.

Troisieme bras, pour trancher entre les deux moities de la structure : meme
motif epars que REINFORCE, mais recompenses re-tirees independamment de l'action.
La parcimonie est conservee, la correlation recompense/action est detruite.
"""

import numpy as np
import torch

from grammaire3 import N
from representable_atteignable_stable import (EmetteurTabulaire, Recepteur,
                                              activer, cloner, lire_code,
                                              objectif, parametres)
from variance_du_gradient import (estimateurs, lois, mesurer, tirer_lots)

BETA = 0.02
GRAINES = 12
PAS = 20000
RAFRAICHIR = 250
REPLICATS = 2000


def etat(emetteur, recepteur):
    with torch.no_grad():
        _, recompense = objectif(emetteur, recepteur, BETA)
    return float(recompense), N - len(set(lire_code(emetteur).tolist()))


def monter_bruite(emetteur, recepteur, pas, lr, lot, mode, generateur):
    """Montee exacte + bruit ajoute au gradient, variance reappariee en continu."""
    activer(emetteur, recepteur)
    optimiseur = torch.optim.Adam(parametres(emetteur, recepteur), lr=lr)
    tenseurs = parametres(emetteur, recepteur)
    n_coord = sum(t.numel() for t in tenseurs)
    sigma = 0.0
    variances = []
    for etape in range(pas):
        if etape % RAFRAICHIR == 0:
            s, r = lois(emetteur, recepteur)
            p = float((s * r.T).sum() / N)
            _, _, var, _ = mesurer(s, r, lot, "ema", generateur, REPLICATS, p, 0.0)
            variances.append(var)
            sigma = float(np.sqrt(var / n_coord))
        j, _ = objectif(emetteur, recepteur, BETA)
        optimiseur.zero_grad()
        (-j).backward()
        if mode == "isotrope":
            with torch.no_grad():
                for t in tenseurs:
                    t.grad.add_(torch.randn(t.shape, dtype=torch.float64) * sigma)
        elif mode == "epars":
            with torch.no_grad():
                bruit_p, bruit_q = bruit_structure(emetteur, recepteur, lot,
                                                   generateur)
                emetteur.p[0].grad.add_(torch.from_numpy(-bruit_p))
                recepteur.p[0].grad.add_(torch.from_numpy(-bruit_q))
        optimiseur.step()
    return float(np.mean(variances))


def bruit_structure(emetteur, recepteur, lot, generateur):
    """Motif epars de REINFORCE, recompenses re-tirees independamment de l'action."""
    s, r = lois(emetteur, recepteur)
    p = float((s * r.T).sum() / N)
    refs, msg, rec, _ = tirer_lots(s, r, lot, 1, generateur)
    faux = (generateur.random((1, lot)) < p).astype(np.float64)
    g_p, g_q = estimateurs(s, r, refs, msg, rec, faux - p)
    gp_moy = s * (r.T - (s * r.T).sum(axis=1)[:, None]) / N
    gq_moy = r * (s.T - (r * s.T).sum(axis=1)[:, None]) / N
    return g_p[0] - gp_moy, g_q[0] - gq_moy


BRAS = [
    ("exact + isotrope, var(lot 8)", 8, "isotrope"),
    ("exact + isotrope, var(lot 64)", 64, "isotrope"),
    ("exact + epars decorrele, lot 8", 8, "epars"),
    ("exact + epars decorrele, lot 64", 64, "epars"),
    ("exact seul (temoin)", 8, "aucun"),
]


if __name__ == "__main__":
    print("=== DU BRUIT, OU CE BRUIT-LA ===")
    print(f"  {GRAINES} pieges, memes graines que echappement_du_piege.py.")
    print(f"  Variance reapparee tous les {RAFRAICHIR} pas sur {REPLICATS} lots.")

    pieges = []
    generateur = np.random.default_rng(606)
    for k in range(GRAINES):
        e, r = EmetteurTabulaire(generateur), Recepteur(generateur)
        activer(e, r)
        optimiseur = torch.optim.Adam(parametres(e, r), lr=0.05)
        for _ in range(PAS):
            j, _ = objectif(e, r, BETA)
            optimiseur.zero_grad()
            (-j).backward()
            optimiseur.step()
        rec, colls = etat(e, r)
        pieges.append((e, r, rec, colls))

    recompenses = np.array([p[2] for p in pieges])
    collisions = np.array([p[3] for p in pieges])
    print(f"\n  les 12 pieges : E[R] {recompenses.mean():.5f}"
          f"   collisions {collisions.mean():.2f}"
          f"   bijectifs {int((collisions == 0).sum())} / {GRAINES}")

    print(f"\n  {'bras':>34}{'E[R] apres':>12}{'delta':>10}{'biject':>9}"
          f"{'colls':>8}{'sortis':>9}{'var moyenne':>13}")
    tirage = np.random.default_rng(31337)
    for nom, lot, mode in BRAS:
        finales, colls_fin, sorties, vars_ = [], [], 0, []
        for k, (e0, r0, rec0, colls0) in enumerate(pieges):
            e = cloner(e0, np.random.default_rng(1000 + k))
            r = cloner(r0, np.random.default_rng(2000 + k))
            torch.manual_seed(k)
            if mode == "aucun":
                v = monter_bruite(e, r, PAS, 0.01, lot, "rien", tirage)
            else:
                v = monter_bruite(e, r, PAS, 0.01, lot, mode, tirage)
            rec, colls = etat(e, r)
            finales.append(rec)
            colls_fin.append(colls)
            vars_.append(v)
            sorties += (colls < colls0)
        finales = np.array(finales)
        colls_fin = np.array(colls_fin)
        print(f"  {nom:>34}{finales.mean():>12.5f}"
              f"{(finales - recompenses).mean():>+10.5f}"
              f"{int((colls_fin == 0).sum()):>6} /{GRAINES:<2}"
              f"{colls_fin.mean():>8.2f}{sorties:>6} /{GRAINES:<2}"
              f"{np.mean(vars_):>13.4e}")
