"""Controle du test 4 : le transfert vient-il d'une structure reutilisable
ou seulement du prefixe litteral partage ?

'le chat dort' -> 'le chien dort' partage les 5 premieres positions ('le ch').
Le facteur x1.74 mesure donc peut-etre juste ca, et rien d'abstrait.

Ce script rejoue le meme transfert vers une cible de meme longueur (13) mais
SANS aucun caractere commun a la meme position. Deux lectures possibles :
  - acceleration proche de x1.74 -> l'agent a appris une structure reutilisable ;
  - acceleration proche de x1.00 -> le transfert n'etait que le prefixe partage,
    donc de la memorisation position par position.
"""

import random

import numpy as np
import torch

from rl_copy import (DOSSIER_SORTIE, CIBLE_PRINCIPALE, CIBLE_PERTURBEE, Vocabulaire,
                      construire_vocabulaire, nouvelle_politique, entrainer,
                      fixer_graine)

GRAINE = 0
MAX_EPISODES = 30000


def cible_sans_recouvrement(longueur, vocabulaire, reference, graine=7):
    """Tire une cible de la longueur voulue dont aucun caractere ne coincide
    avec `reference` a la meme position."""
    fixer_graine(graine)
    caracteres = []
    for i in range(longueur):
        interdits = {reference[i]} if i < len(reference) else set()
        choix = [c for c in vocabulaire.caracteres if c not in interdits]
        caracteres.append(random.choice(choix))
    return "".join(caracteres)


vocabulaire = Vocabulaire(construire_vocabulaire([CIBLE_PRINCIPALE, CIBLE_PERTURBEE]))
cible_neutre = cible_sans_recouvrement(len(CIBLE_PERTURBEE), vocabulaire, CIBLE_PRINCIPALE)

recouvrement = sum(1 for i, c in enumerate(CIBLE_PERTURBEE)
                   if i < len(CIBLE_PRINCIPALE) and CIBLE_PRINCIPALE[i] == c)
print(f"Cible d'origine        : '{CIBLE_PRINCIPALE}'")
print(f"Cible perturbee        : '{CIBLE_PERTURBEE}' "
      f"({recouvrement}/{len(CIBLE_PERTURBEE)} positions identiques)")
print(f"Cible sans recouvrement: '{cible_neutre}' (0/{len(cible_neutre)} positions identiques)")
print()

# 1. On refait l'entrainement d'origine (graine 0, donc trajectoire identique
#    a celle du run principal : 1er reward=1.0 a l'episode 1639).
print("Entrainement d'origine sur 'le chat dort'...")
politique = nouvelle_politique(vocabulaire, GRAINE, "cpu")
run_origine = entrainer(politique, CIBLE_PRINCIPALE, vocabulaire,
                        max_episodes=MAX_EPISODES, verbeux=False, etiquette="origine")
print(f"  1er reward=1.0 : {run_origine['premier_parfait']} "
      f"| greedy '{run_origine['greedy_final']}'")
torch.save(politique.state_dict(),
           os.path.join(DOSSIER_SORTIE, "politique_le_chat_dort.pt"))

# 2. Transfert vers la cible sans recouvrement (poids conserves, pas de reset)
print(f"\nTransfert vers '{cible_neutre}' (poids conserves)...")
run_transfert = entrainer(politique, cible_neutre, vocabulaire,
                          max_episodes=MAX_EPISODES, verbeux=False, etiquette="transfert_neutre")
print(f"  1er reward=1.0 : {run_transfert['premier_parfait']} "
      f"| convergence : {run_transfert['episode_convergence']} "
      f"| greedy '{run_transfert['greedy_final']}'")

# 3. Controle : meme cible, poids reinitialises
print(f"\nDepuis zero sur '{cible_neutre}'...")
politique_zero = nouvelle_politique(vocabulaire, GRAINE + 100, "cpu")
run_zero = entrainer(politique_zero, cible_neutre, vocabulaire,
                     max_episodes=MAX_EPISODES, verbeux=False, etiquette="zero_neutre")
print(f"  1er reward=1.0 : {run_zero['premier_parfait']} "
      f"| convergence : {run_zero['episode_convergence']} "
      f"| greedy '{run_zero['greedy_final']}'")

facteur = run_zero["premier_parfait"] / run_transfert["premier_parfait"]
print()
print("=" * 70)
print(f"Acceleration du transfert SANS recouvrement : x{facteur:.2f}")
print(f"(rappel : x1.74 avec 'le chien dort', qui partage {recouvrement} positions)")
if facteur < 1.25:
    print("-> Le transfert observe au test 4 s'explique par le prefixe litteral partage.")
    print("   Aucune structure abstraite reutilisable : memorisation position par position.")
else:
    print("-> Le transfert survit a la perte du recouvrement : l'agent a appris")
    print("   quelque chose de reutilisable au-dela des caracteres partages.")
print("=" * 70)
