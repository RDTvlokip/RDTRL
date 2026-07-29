"""Deux questions de plus, dont une qui pourrait etre un correctif.

Q-D — LA CAUSE EST-ELLE L'OBJECTIF OU LA PARAMETRISATION ?

On a deja elimine le bruit d'echantillonnage : avec gradient exact, la politique
s'effondre quand meme. Restent deux suspects, qu'on n'a jamais separes :
  (a) l'objectif E[R] + beta*H lui-meme ;
  (b) la factorisation AUTOREGRESSIVE (un meme reseau doit servir les 6
      conditionnelles apres chaque determinant, avec un etat cache partage).

Test : optimiser le MEME objectif avec le MEME gradient exact, mais sur une
parametrisation tabulaire — un logit libre par sequence, aucune factorisation,
aucun partage de parametres. Si le tabulaire atteint l'uniforme sur les 48 et
pas le GRU, alors la coupable est la parametrisation autoregressive, pas
l'objectif. Ce serait une interference representationnelle entre conditionnelles,
un mecanisme qui n'a rien a voir avec le RL.

Q-E — LE RECUIT DE BETA PEUT-IL SAUVER LES DEUX BRANCHES ?

Le balayage montre deux regimes incompatibles : beta eleve garde les deux
branches vivantes mais detruit la grammaticalite, beta faible est grammatical
mais mono-branche. Personne ne pense a les enchainer, parce qu'il faut d'abord
savoir que la structure en branches existe.

On demarre a beta eleve (les deux branches vivantes) puis on descend
progressivement. Si les branches survivent a la descente, on obtient a la fois la
validite et la couverture — donc un correctif, pas seulement un diagnostic. Si
elles ne survivent pas, le systeme a de l'hysteresis et le recuit est inutile.
"""

import json
import os
from collections import deque
from itertools import product

import numpy as np
import torch

from grammar import Grammaire
from rl_grammar import PolitiqueGRU, analyse_exacte, fixer_graine, DOSSIER_SORTIE

ETAPES_TABULAIRE = 6000
EPISODES_RECUIT = 30000


def profil_depuis_probas(p, sequences, grammaire):
    """Memes mesures que analyse_exacte, mais depuis un vecteur de probabilites."""
    valide = np.array([grammaire.analyser(s)["valide"] for s in sequences])
    i_nom = grammaire.positions["nom"]
    masse = float(p[valide].sum())
    pv = p[valide] / max(masse, 1e-300)
    h = float(-(pv * np.log2(np.clip(pv, 1e-300, None))).sum())
    familles = {"sg": 0.0, "pl": 0.0}
    for j, s in enumerate(np.array(sequences)[valide]):
        familles[grammaire.traits(grammaire.tokens[s[i_nom]])["nombre"]] += float(pv[j])
    return {"valide_pct": round(100 * masse, 2), "modes": round(2 ** h, 1),
            "uniformite_pct": round(100 * h / np.log2(valide.sum()), 1),
            "sg": round(100 * familles["sg"], 1), "pl": round(100 * familles["pl"], 1)}


def q_d_tabulaire(grammaire, sequences, recompenses, beta, graine):
    """Un logit libre par sequence : aucune factorisation, aucun partage."""
    fixer_graine(graine)
    logits = torch.zeros(len(sequences), requires_grad=True)
    optimiseur = torch.optim.Adam([logits], lr=0.05)
    r = torch.tensor(recompenses, dtype=torch.float32)
    for _ in range(ETAPES_TABULAIRE):
        p = torch.softmax(logits, dim=0)
        j = (p * r).sum() - beta * (p * torch.log(p.clamp_min(1e-30))).sum()
        optimiseur.zero_grad()
        (-j).backward()
        optimiseur.step()
    return torch.softmax(logits, dim=0).detach().numpy()


def q_e_recuit(grammaire, beta_debut, beta_fin, episodes, graine, lr=1e-3):
    """REINFORCE avec beta decroissant geometriquement."""
    fixer_graine(graine)
    politique = PolitiqueGRU(grammaire.taille)
    optimiseur = torch.optim.Adam(politique.parameters(), lr=lr)
    historique = deque(maxlen=100)
    trace = []
    for episode in range(1, episodes + 1):
        fraction = (episode - 1) / max(episodes - 1, 1)
        beta = beta_debut * (beta_fin / beta_debut) ** fraction
        actions, log_probs, entropies, _ = politique.generer(grammaire.longueur, taille_lot=1)
        r = grammaire.recompense_graduee(actions[0].tolist())
        baseline = sum(historique) / len(historique) if historique else 0.0
        historique.append(r)
        avantage = torch.tensor(r - baseline, dtype=torch.float32)
        perte = -(log_probs.sum() * avantage) - beta * entropies.sum()
        optimiseur.zero_grad()
        perte.backward()
        torch.nn.utils.clip_grad_norm_(politique.parameters(), 5.0)
        optimiseur.step()
        if episode % 3000 == 0:
            ex = analyse_exacte(politique, grammaire)
            trace.append({"episode": episode, "beta": round(beta, 5),
                          "valide_pct": ex["masse_valide_pct"],
                          "modes": ex["modes_effectifs"],
                          "sg": ex["repartition_familles"]["sg"],
                          "pl": ex["repartition_familles"]["pl"]})
    return politique, trace


if __name__ == "__main__":
    g = Grammaire(longue=False)
    sequences = list(product(range(g.taille), repeat=g.longueur))
    recompenses = np.array([g.recompense_graduee(s) for s in sequences])
    resultats = {}

    print("=" * 84)
    print("Q-D — OBJECTIF ou PARAMETRISATION ? Tabulaire contre GRU, gradient exact")
    print("=" * 84)
    print("Meme objectif, meme gradient exact, mais un logit libre par sequence.")
    print("Rappel GRU + gradient exact a beta=0.01 : 12.0 modes, 100 % singulier.\n")
    print(f"{'beta':>6} {'graine':>7} {'valide%':>9} {'modes':>7} {'unifor%':>9} "
          f"{'sg%':>6} {'pl%':>6}")
    tab = []
    for beta in (0.01, 0.02, 0.05):
        for graine in (0, 1):
            p = q_d_tabulaire(g, sequences, recompenses, beta, graine)
            prof = profil_depuis_probas(p, sequences, g)
            prof.update({"beta": beta, "graine": graine})
            tab.append(prof)
            print(f"{beta:>6} {graine:>7} {prof['valide_pct']:>9.2f} {prof['modes']:>7.1f} "
                  f"{prof['uniformite_pct']:>9.1f} {prof['sg']:>6.1f} {prof['pl']:>6.1f}")
    resultats["tabulaire"] = tab
    print()
    print("Si le tabulaire atteint ~48 modes et 50/50 la ou le GRU plafonne a 12,")
    print("la coupable est la FACTORISATION AUTOREGRESSIVE, pas l'objectif :")
    print("un etat cache partage doit servir les 6 conditionnelles a la fois, et")
    print("il se specialise sur celle qui recoit le plus de gradient.\n")

    print("=" * 84)
    print("Q-E — RECUIT DE BETA : garder les deux branches ET la grammaticalite ?")
    print("=" * 84)
    for beta_debut, beta_fin in ((0.2, 0.01), (0.12, 0.02)):
        print(f"  recuit {beta_debut} -> {beta_fin} :")
        politique, trace = q_e_recuit(g, beta_debut, beta_fin, EPISODES_RECUIT, graine=0)
        for point in trace:
            print(f"    ep {point['episode']:6d} | beta {point['beta']:.4f} | "
                  f"valide {point['valide_pct']:6.2f} % | {point['modes']:5.1f} modes | "
                  f"sg {point['sg']:5.1f} / pl {point['pl']:5.1f}")
        fin = trace[-1]
        deux_branches = min(fin["sg"], fin["pl"]) > 5
        print(f"    -> {'LES DEUX BRANCHES SURVIVENT' if deux_branches else 'une seule branche subsiste (hysteresis)'}"
              f" | validite finale {fin['valide_pct']:.2f} %, {fin['modes']} modes")
        resultats[f"recuit_{beta_debut}_{beta_fin}"] = trace
        print()

    print("Reference a battre : beta constant 0.02 donne 99.99 % et 18.6 modes ;")
    print("beta constant 0.12 donne 57.13 % et 45.9 modes. Un recuit reussi ferait")
    print("mieux que les deux simultanement.")

    with open(os.path.join(DOSSIER_SORTIE, "parametrisation_recuit.json"), "w",
              encoding="utf-8") as f:
        json.dump(resultats, f, indent=2, ensure_ascii=False)
