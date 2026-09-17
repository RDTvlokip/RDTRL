"""Recette exacte pour reproduire la deuxieme egalite documentee du
projet (apres referents 3/4 de replay_mur23_referent3.py) : referents
23/25, message 13, tour 30-31 (§7.47-48 de CARNET.md).

Retrouvee le 17/09/2026 en grepant le transcript JSONL de la session
courante pour "23/25" -- meme methode que replay_idx5.py. Recette :
`default_rng(50000)`, UN SEUL couple EmetteurTabulaire/Recepteur (pas
de boucle a sauter, contrairement a idx5 et mur23/24), monter(beta=0.02,
20000 pas, lr=0.05).

Verifie : masses [0.5001132951215442, 0.4998866825562411] -- identiques
au chiffre pres a celles publiees dans REPONSE_ORDRE31.md /
CARNET.md §7.47.

A la difference du mur referents 3/4 (perturbation artificielle +30),
cette egalite est NATURELLE -- atteinte par la dynamique seule, sans
aucune poussee. Bon candidat pour reproduire le mecanisme k(R)/delta_c
des tours 48-52 sur une DEUXIEME collision independante.
"""

import numpy as np
import torch

from representable_atteignable_stable import EmetteurTabulaire, Recepteur, monter, lire_code

BETA = 0.02
GRAINE = 50000
PAS = 20000


def construire():
    g = np.random.default_rng(GRAINE)
    e, r = EmetteurTabulaire(g), Recepteur(g)
    monter(e, r, BETA, PAS, lr=0.05)
    return e, r


if __name__ == "__main__":
    torch.set_printoptions(precision=15)
    e, r = construire()
    code = lire_code(e)
    uniques, comptes = np.unique(code, return_counts=True)
    doublons = uniques[comptes > 1]
    with torch.no_grad():
        rr = r.loi()
    print("=== collisions a graine 50000, 20000 pas ===")
    for m in doublons:
        refs = np.where(code == m)[0]
        if len(refs) == 2:
            masses = [float(rr[m, ref]) for ref in refs]
            print(f"  message={m}  refs={refs.tolist()}  masses={masses}")
    print("\n  attendu : message=13  refs=[23, 25]  "
          "masses=[0.5001132951215442, 0.4998866825562411]")
