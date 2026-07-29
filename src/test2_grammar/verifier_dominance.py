"""Verification numerique de l'argument de dominance.

J'ai affirme que l'uniforme sur les 48 phrases valides domine strictement les
politiques apprises sur les DEUX termes de l'objectif E[R] + beta*H. Sur E[R]
c'est evident. Sur l'entropie je n'avais rien calcule : la politique apprise
place une part de sa masse sur des sequences invalides, eparpillee sur un espace
de 7 952 possibilites, ce qui CONTRIBUE a son entropie.

Ce script calcule exactement, pour chaque politique sauvegardee :
    J = E[R] + beta * H(trajectoire)
et le compare a J(uniforme sur les 48) = 1 + beta * ln(48).

Si J(uniforme48) > J(apprise), la politique apprise est prouvablement
sous-optimale, sans aucun formalisme max-ent. Sinon, l'argument ne tient pas a
ce beta et il faut le dire.

Entropie en nats, comme torch.distributions.Categorical.entropy().
"""

import os
from itertools import product

import numpy as np
import torch

from grammaire import Grammaire
from rl_grammaire import PolitiqueGRU, distribution_exacte, DOSSIER_SORTIE

COEFS = [0.0, 0.01, 0.02, 0.05, 0.08, 0.12, 0.2, 0.35, 0.5]


def profil_politique(politique, grammaire, recompenses, valide):
    """E[R], entropie de trajectoire exacte, masse valide."""
    _, probas = distribution_exacte(politique, grammaire)
    p = probas.double().numpy()
    p = p / p.sum()                       # garde-fou numerique
    esperance_r = float((p * recompenses).sum())
    non_nuls = p[p > 1e-300]
    entropie = float(-(non_nuls * np.log(non_nuls)).sum())
    return esperance_r, entropie, float(p[valide].sum())


if __name__ == "__main__":
    g = Grammaire(longue=False)
    sequences = list(product(range(g.taille), repeat=g.longueur))
    recompenses = np.array([g.recompense_graduee(s) for s in sequences])
    valide = np.array([g.analyser(s)["valide"] for s in sequences])
    n_valides = int(valide.sum())

    # Reference : uniforme sur les 48 phrases valides
    r_ref = 1.0
    h_ref = float(np.log(n_valides))
    print("=" * 96)
    print("VERIFICATION DE L'ARGUMENT DE DOMINANCE")
    print("=" * 96)
    print(f"Reference = uniforme sur les {n_valides} phrases valides : "
          f"E[R] = {r_ref:.4f}, H = ln({n_valides}) = {h_ref:.4f} nats\n")

    print(f"{'beta':>6} | {'POLITIQUE APPRISE':^34} | {'OBJECTIF J = E[R] + beta*H':^32} |")
    print(f"{'':>6} | {'E[R]':>8} {'H nats':>8} {'valide%':>9} | "
          f"{'J apprise':>11} {'J uniforme48':>13} | {'verdict':>22}")

    lignes = []
    for beta in COEFS:
        chemin = os.path.join(DOSSIER_SORTIE, f"politique_ent{beta}.pt")
        if not os.path.exists(chemin):
            print(f"{beta:>6} | {'(modele absent)':^34} |")
            continue
        politique = PolitiqueGRU(g.taille)
        politique.load_state_dict(torch.load(chemin))
        politique.eval()

        r, h, masse = profil_politique(politique, g, recompenses, valide)
        j_apprise = r + beta * h
        j_ref = r_ref + beta * h_ref
        domine = j_ref > j_apprise
        verdict = "sous-optimale (prouve)" if domine else "argument NE TIENT PAS"
        print(f"{beta:>6} | {r:>8.4f} {h:>8.4f} {100*masse:>9.2f} | "
              f"{j_apprise:>11.4f} {j_ref:>13.4f} | {verdict:>22}")
        lignes.append({"beta": beta, "E_R": r, "H": h, "masse_valide": masse,
                       "J_apprise": j_apprise, "J_ref": j_ref, "domine": domine})

    print()
    ok = [l for l in lignes if l["domine"]]
    ko = [l for l in lignes if not l["domine"]]
    print(f"L'argument de dominance prouve la sous-optimalite pour beta = "
          f"{[l['beta'] for l in ok]}")
    if ko:
        print(f"Il NE prouve rien pour beta = {[l['beta'] for l in ko]} : a ces valeurs")
        print("la politique apprise disperse assez de masse sur l'invalide pour avoir")
        print("une entropie superieure a celle de l'uniforme sur les 48. La conclusion")
        print("doit donc etre explicitement restreinte au plateau.")
    if ok:
        marges = [l["J_ref"] - l["J_apprise"] for l in ok]
        print(f"Marge minimale la ou il tient : {min(marges):.4f} "
              f"(sur beta = {ok[int(np.argmin(marges))]['beta']})")
