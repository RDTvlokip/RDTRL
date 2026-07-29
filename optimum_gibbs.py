"""Optimum theorique de l'objectif reellement optimise, calcule exactement.

La perte de REINFORCE avec bonus d'entropie est

    -E[R] - beta * somme_t H(a_t | a_<t)

et la somme des entropies conditionnelles par pas EST l'entropie de trajectoire.
On optimise donc E[R] + beta*H(s), dont l'optimum unique est la loi de Gibbs

    pi*(s) = exp(R(s)/beta) / Z

Sur la grammaire courte, les 8 000 sequences sont enumerables : pi* est donc
calculable exactement, sans aucun entrainement. Cela permet de decomposer un
mauvais resultat en deux causes qu'on confond d'habitude :

  - la TAXE DE MISE EN FORME : pi* lui-meme n'est pas parfait. La recompense
    graduee paye 0.833 un quasi-raton, donc son optimum contient des phrases
    invalides en proportion exp(-Delta/beta). Aucun optimiseur ne peut faire
    mieux que pi*.
  - l'ECART D'OPTIMISATION : ce qui separe la politique apprise de pi*.

Note importante : les 48 phrases valides ont toutes R = 1 exactement. La loi de
Gibbs leur assigne donc a toutes la MEME probabilite, quel que soit beta. A
l'optimum on a donc toujours 48 modes effectifs, 100 % d'uniformite et un
partage singulier/pluriel a 50/50. Tout ecart mesure est un echec
d'optimisation, pas une propriete de la tache.
"""

import json
import os
from itertools import product

import numpy as np

from grammaire import Grammaire

DOSSIER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultats_test2")
COEFS = [0.01, 0.02, 0.05, 0.08, 0.12, 0.2, 0.35, 0.5]


def profil_gibbs(grammaire, recompenses, valide, familles, beta):
    """Caracteristiques de pi* pour un beta donne."""
    if beta <= 0:
        # Limite beta -> 0 : uniforme sur les sequences de recompense maximale
        maxi = recompenses.max()
        p = (recompenses >= maxi - 1e-12).astype(float)
        p /= p.sum()
    else:
        logits = recompenses / beta
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()

    masse_valide = p[valide].sum()
    pv = p[valide] / masse_valide
    entropie = float(-(pv * np.log2(np.clip(pv, 1e-300, None))).sum())
    return {
        "masse_valide_pct": round(100 * float(masse_valide), 3),
        "modes_effectifs": round(float(2 ** entropie), 1),
        "uniformite_pct": round(100 * entropie / np.log2(valide.sum()), 1),
        "sg_pct": round(100 * float(pv[familles[valide] == "sg"].sum()), 1),
        "pl_pct": round(100 * float(pv[familles[valide] == "pl"].sum()), 1),
    }


if __name__ == "__main__":
    g = Grammaire(longue=False)
    sequences = list(product(range(g.taille), repeat=g.longueur))
    i_nom = g.positions["nom"]

    r_graduee = np.array([g.recompense_graduee(s) for s in sequences])
    r_sparse = np.array([g.recompense_tout_ou_rien(s) for s in sequences])
    valide = np.array([g.analyser(s)["valide"] for s in sequences])
    familles = np.array([g.traits(g.tokens[s[i_nom]])["nombre"] or "?" for s in sequences])

    print("=" * 92)
    print("OPTIMUM DE GIBBS — ce que l'objectif vise reellement, calcule exactement")
    print("=" * 92)
    print(f"Sequences enumerees : {len(sequences)} | phrases valides : {int(valide.sum())}")
    niveaux = sorted(set(np.round(r_graduee, 4)), reverse=True)[:4]
    print(f"Niveaux de recompense graduee les plus hauts : {niveaux}")
    print(f"Ecart entre valide (1.0) et meilleur quasi-raton : "
          f"{niveaux[0] - niveaux[1]:.4f}")
    print(f"Nombre de sequences au niveau {niveaux[1]:.4f} : "
          f"{int((np.round(r_graduee,4) == niveaux[1]).sum())}")
    print()

    # Valeurs atteintes par l'entrainement, si le rapport est disponible
    atteint = {}
    chemin = os.path.join(DOSSIER, "rapport.json")
    if os.path.exists(chemin):
        with open(chemin, encoding="utf-8") as f:
            rapport = json.load(f)
        for b in rapport.get("balayage_entropie", []):
            atteint[b["coef_entropie"]] = b

    for nom, recompenses in (("GRADUEE", r_graduee), ("TOUT-OU-RIEN", r_sparse)):
        print(f"--- recompense {nom} ---")
        print(f"{'beta':>6} | {'OPTIMUM pi*':^34} | {'ATTEINT':^34} | {'ecart':>8}")
        print(f"{'':>6} | {'valide%':>9} {'modes':>7} {'sg/pl':>15} | "
              f"{'valide%':>9} {'modes':>7} {'sg/pl':>15} | {'modes':>8}")
        for beta in COEFS:
            opt = profil_gibbs(g, recompenses, valide, familles, beta)
            ligne = (f"{beta:>6} | {opt['masse_valide_pct']:>9.2f} "
                     f"{opt['modes_effectifs']:>7.1f} "
                     f"{opt['sg_pct']:>7.1f}/{opt['pl_pct']:<7.1f} | ")
            a = atteint.get(beta) if nom == "GRADUEE" else None
            if a:
                fam = a["repartition_familles"]
                ligne += (f"{a['masse_valide_exacte_pct']:>9.2f} "
                          f"{a['modes_effectifs']:>7.1f} "
                          f"{fam['sg']:>7.1f}/{fam['pl']:<7.1f} | "
                          f"{opt['modes_effectifs'] - a['modes_effectifs']:>8.1f}")
            else:
                ligne += f"{'-':>9} {'-':>7} {'-':>15} | {'-':>8}"
            print(ligne)
        print()

    print("Lecture :")
    print("  - la colonne OPTIMUM donne toujours 48 modes et 50/50 : les 48 phrases")
    print("    valides sont a egalite exacte de recompense, donc equiprobables sous")
    print("    max-ent, a n'importe quel beta.")
    print("  - la taxe de mise en forme est 100 - valide% de l'optimum : elle est nulle")
    print("    pour le tout-ou-rien et non nulle pour la recompense graduee, qui paye")
    print("    les quasi-ratons par construction.")
    print("  - l'ecart de modes est integralement imputable a l'optimisation.")
