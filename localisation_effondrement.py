"""Q-C corrige — quelle position porte l'effondrement ?

Version precedente fausse : l'action de la position figee etait remplacee APRES
generation, donc la suite de la sequence restait conditionnee sur le token que le
modele avait tire, pas sur celui qu'on imposait. La sequence evaluee ne
correspondait a aucun deroule coherent.

Version correcte : le token est impose PENDANT la generation (parametre `forcer`
de generer), donc l'etat cache et toutes les positions suivantes en tiennent
compte. Et la position figee est exclue du terme REINFORCE, puisque la politique
n'a aucun controle dessus.

Lecture : si figer une position suffit a restaurer diversite ET validite, tout
l'effondrement est localise la. Sinon il est distribue.
"""

import json
import os
from collections import deque
from itertools import product

import numpy as np
import torch

from grammaire import Grammaire
from rl_grammaire import (PolitiqueGRU, analyse_exacte, distribution_exacte,
                          fixer_graine, DOSSIER_SORTIE)
from sonde_capacite import phrases_valides_en_ids, ajuster

EPISODES = 20000
BETA = 0.02


def entrainer_position_figee(grammaire, position_figee, marginale, graine=0,
                             episodes=EPISODES, beta=BETA, lr=1e-3):
    """REINFORCE avec une position imposee pendant la generation."""
    fixer_graine(graine)
    politique = PolitiqueGRU(grammaire.taille)
    optimiseur = torch.optim.Adam(politique.parameters(), lr=lr)
    historique = deque(maxlen=100)
    generateur = np.random.default_rng(graine)

    for _ in range(episodes):
        if position_figee is None:
            forcer = None
        else:
            token = int(generateur.choice(len(marginale), p=marginale))
            forcer = {position_figee: token}
        actions, log_probs, entropies, _ = politique.generer(
            grammaire.longueur, taille_lot=1, forcer=forcer)

        r = grammaire.recompense_graduee(actions[0].tolist())
        baseline = sum(historique) / len(historique) if historique else 0.0
        historique.append(r)
        avantage = torch.tensor(r - baseline, dtype=torch.float32)

        # La politique n'a pas choisi la position figee : on l'exclut du gradient.
        libres = [t for t in range(grammaire.longueur) if t != position_figee]
        perte = -(log_probs[0, libres].sum() * avantage) - beta * entropies[0, libres].sum()

        optimiseur.zero_grad()
        perte.backward()
        torch.nn.utils.clip_grad_norm_(politique.parameters(), 5.0)
        optimiseur.step()
    return politique


if __name__ == "__main__":
    g = Grammaire(longue=False)
    sequences = list(product(range(g.taille), repeat=g.longueur))
    seq_arr = np.array(sequences)
    cibles = phrases_valides_en_ids(g)

    # Marginales de la politique ideale, obtenues par ajustement supervise
    fixer_graine(0)
    ideale = PolitiqueGRU(g.taille)
    ajuster(ideale, cibles, etapes=3000, lr=5e-3)
    _, probas = distribution_exacte(ideale, g)
    p_ideal = probas.double().numpy()
    p_ideal /= p_ideal.sum()
    marginales = []
    for position in range(g.longueur):
        m = np.array([p_ideal[seq_arr[:, position] == i].sum() for i in range(g.taille)])
        marginales.append(m / m.sum())

    print("=" * 88)
    print("Q-C (corrige) — quelle position porte l'effondrement ?")
    print("=" * 88)
    print("Le token impose l'est PENDANT la generation, donc la suite en tient compte.")
    print("La position figee est exclue du terme REINFORCE.\n")
    print("Reference sans rien figer (beta=0.02) : ~99.9 % valide, 12-19 modes, 1 branche\n")
    print(f"{'figee':>16} {'valide%':>9} {'modes':>7} {'unifor%':>9} {'sg%':>6} {'pl%':>6} "
          f"{'P(nom|det)':>11} {'P(vb|nom)':>10}")

    lignes = []
    for position in [None] + list(range(g.longueur)):
        etiquette = "aucune" if position is None else f"pos{position} ({g.structure[position]})"
        politique = entrainer_position_figee(
            g, position, marginales[position] if position is not None else None)
        ex = analyse_exacte(politique, g)
        print(f"{etiquette:>16} {ex['masse_valide_pct']:>9.2f} {ex['modes_effectifs']:>7.1f} "
              f"{ex['uniformite_pct']:>9.1f} {ex['repartition_familles']['sg']:>6.1f} "
              f"{ex['repartition_familles']['pl']:>6.1f} {ex['moyenne_cond_det']:>11.3f} "
              f"{ex['moyenne_cond_nom']:>10.3f}")
        lignes.append({"figee": etiquette, "valide_pct": ex["masse_valide_pct"],
                       "modes": ex["modes_effectifs"],
                       "familles": ex["repartition_familles"],
                       "cond_det": ex["moyenne_cond_det"],
                       "cond_nom": ex["moyenne_cond_nom"]})

    print()
    print("Lecture : une ligne qui retrouve a la fois ~99 % de validite ET ~48 modes")
    print("localiserait tout l'effondrement dans la position correspondante.")

    with open(os.path.join(DOSSIER_SORTIE, "localisation_effondrement.json"), "w",
              encoding="utf-8") as f:
        json.dump(lignes, f, indent=2, ensure_ascii=False)
