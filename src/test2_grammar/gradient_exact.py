"""Deux questions que seul un espace enumerable permet de poser.

PARTIE 1 — De quoi le gradient est-il fait au depart ?

A politique uniforme, le gradient de REINFORCE pour la position p ne voit que
E[R | token en p], c'est-a-dire la projection d'ORDRE 1 de la recompense : toutes
les interactions entre positions sont moyennees et disparaissent. L'agent est
donc initialement AVEUGLE aux contraintes d'accord, qui sont d'ordre 2 par
nature. Il ne peut suivre que le signal marginal.

Ce script calcule ce signal marginal exactement. S'il pointe vers une famille de
solutions plutot qu'une autre, le "choix de branche" n'est pas une loterie : il
est predit par la structure de la recompense.

PARTIE 2 — L'effondrement vient-il du BRUIT ou de la GEOMETRIE ?

Impossible a trancher sur un vrai modele : on ne sait pas calculer le gradient
exact. Ici si. On optimise directement

    J = somme_s p(s) R(s) + beta * H(p)

ou p(s) est la probabilite exacte de chaque sequence sous la politique, obtenue
par enumeration. Gradient exact, zero echantillonnage.

  - si la politique converge vers l'uniforme sur les 48 : l'effondrement observe
    en RL est entierement un artefact du bruit d'echantillonnage ;
  - si elle s'effondre quand meme : c'est la geometrie de l'objectif qui est en
    cause, et aucune reduction de variance ne sauvera REINFORCE.
"""

from itertools import product

import numpy as np
import torch

from grammaire import Grammaire
from rl_grammaire import PolitiqueGRU, analyse_exacte, fixer_graine

ETAPES = 4000
LR = 5e-3


def probabilites_exactes(politique, sequences_t):
    """p(s) pour toutes les sequences, differentiable."""
    debut = torch.full((len(sequences_t), 1), politique.token_debut, dtype=torch.long)
    entrees = torch.cat([debut, sequences_t[:, :-1]], dim=1)
    sorties, _ = politique.gru(politique.embedding(entrees))
    log_probas = torch.log_softmax(politique.tete(sorties), dim=-1)
    log_p = log_probas.gather(2, sequences_t.unsqueeze(-1)).squeeze(-1).sum(1)
    return log_p


def partie1_signal_ordre1(g, sequences, recompenses):
    """E[R | token t en position p] sous politique uniforme = projection d'ordre 1."""
    print("=" * 84)
    print("PARTIE 1 — Ce que le gradient voit au depart : la recompense d'ordre 1")
    print("=" * 84)
    print("A politique uniforme, toutes les interactions entre positions sont")
    print("moyennees : l'agent ne percoit que la recompense marginale par token.\n")

    seq = np.array(sequences)
    for position, categorie in enumerate(g.structure):
        print(f"  Position {position} ({categorie}) — E[R | token ici], politique uniforme :")
        marginales = {}
        for token in g.tokens:
            masque = seq[:, position] == g.index[token]
            marginales[token] = float(recompenses[masque].mean())
        for token, valeur in sorted(marginales.items(), key=lambda kv: -kv[1]):
            traits = g.traits(token)
            marque = "  <-" if traits["categorie"] == categorie else ""
            print(f"    {token:9s} {valeur:.4f}   ({traits['categorie']}, "
                  f"{traits['genre']}, {traits['nombre']}){marque}")
        print()

    # Le point decisif : comparer les familles sur la position du nom
    i_nom = g.positions["nom"]
    familles = {"sg": [], "pl": []}
    for nom in g.tokens_par_categorie["nom"]:
        masque = seq[:, i_nom] == g.index[nom]
        familles[g.traits(nom)["nombre"]].append(float(recompenses[masque].mean()))
    m_sg, m_pl = np.mean(familles["sg"]), np.mean(familles["pl"])
    print(f"  >> Signal marginal moyen des noms SINGULIERS : {m_sg:.4f}")
    print(f"  >> Signal marginal moyen des noms PLURIELS   : {m_pl:.4f}")
    print(f"  >> Ecart : {m_sg - m_pl:+.4f} en faveur du "
          f"{'SINGULIER' if m_sg > m_pl else 'PLURIEL'}")
    print()
    print("  Cause : les determinants pluriels 'les' et 'des' sont neutres en genre.")
    print("  Ils valent 1.0 avec un nom pluriel mais 0.5 avec un nom singulier (le")
    print("  genre passe, le nombre non), alors que 'le/la/un/une' donnent 0.5 a un")
    print("  nom pluriel. Les noms singuliers recoltent donc plus de credit PARTIEL")
    print("  en moyenne. Le signal d'ordre 1 favorise le singulier.\n")
    return m_sg, m_pl


def partie2_gradient_exact(g, sequences_t, recompenses_t, beta, graine):
    """Optimisation de J = E[R] + beta*H avec gradient exact, sans echantillonnage."""
    fixer_graine(graine)
    politique = PolitiqueGRU(g.taille)
    optimiseur = torch.optim.Adam(politique.parameters(), lr=LR)
    for etape in range(ETAPES):
        log_p = probabilites_exactes(politique, sequences_t)
        p = log_p.exp()
        p = p / p.sum()                       # garde-fou numerique
        esperance = (p * recompenses_t).sum()
        entropie = -(p * torch.log(p.clamp_min(1e-30))).sum()
        perte = -(esperance + beta * entropie)
        optimiseur.zero_grad()
        perte.backward()
        optimiseur.step()
    return politique


if __name__ == "__main__":
    g = Grammaire(longue=False)
    sequences = list(product(range(g.taille), repeat=g.longueur))
    recompenses = np.array([g.recompense_graduee(s) for s in sequences])
    sequences_t = torch.tensor(sequences, dtype=torch.long)
    recompenses_t = torch.tensor(recompenses, dtype=torch.float32)

    partie1_signal_ordre1(g, sequences, recompenses)

    print("=" * 84)
    print("PARTIE 2 — Gradient EXACT : l'effondrement est-il du bruit ou de la geometrie ?")
    print("=" * 84)
    print(f"{'beta':>6} {'graine':>7} {'valide%':>9} {'modes':>7} {'unifor%':>8} "
          f"{'sg%':>6} {'pl%':>6} {'P(nom|det)':>11}")
    for beta in (0.01, 0.02, 0.05, 0.08):
        for graine in (0, 1, 2):
            politique = partie2_gradient_exact(g, sequences_t, recompenses_t, beta, graine)
            ex = analyse_exacte(politique, g)
            print(f"{beta:>6} {graine:>7} {ex['masse_valide_pct']:>9.2f} "
                  f"{ex['modes_effectifs']:>7.1f} {ex['uniformite_pct']:>8.1f} "
                  f"{ex['repartition_familles']['sg']:>6.1f} "
                  f"{ex['repartition_familles']['pl']:>6.1f} "
                  f"{ex['moyenne_cond_det']:>11.3f}")
    print()
    print("Lecture : si les modes effectifs approchent 48 avec un partage 50/50, alors")
    print("l'effondrement observe en RL echantillonne est un artefact du BRUIT. S'ils")
    print("restent bas, c'est la GEOMETRIE de l'objectif, et aucune reduction de")
    print("variance ne sauvera la methode.")
