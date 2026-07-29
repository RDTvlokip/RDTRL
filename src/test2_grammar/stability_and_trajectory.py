"""Trois questions que seule une trajectoire exacte permet de poser.

Q-A — STABILITE contre ACCESSIBILITE
On sait que REINFORCE n'ATTEINT pas l'uniforme sur les 48. Mais est-ce seulement
un probleme d'acces, ou l'uniforme n'est-il meme pas un point fixe stable ?
On part de la politique ideale (obtenue par ajustement supervise) et on lance
REINFORCE depuis la. Si elle derive, l'ideal n'est pas stable, ce qui est
beaucoup plus fort que "difficile a atteindre". Deux echecs tres differents que
personne ne separe, faute de pouvoir construire la politique ideale.

Q-B — L'OPTIMUM EST-IL UNE ETAPE PLUTOT QU'UNE DESTINATION ?
En partant de l'aleatoire, on suit KL(pi_t || uniforme sur les 48) tout au long
de l'entrainement. Si cette courbe a un MINIMUM en cours de route, alors la
meilleure politique apparait au milieu de l'entrainement et continuer nuit.
Un arret precoce battrait la convergence.

Q-C — QUELLE POSITION EST RESPONSABLE ?
On fige la marginale d'une seule position a sa valeur ideale et on laisse le
reste s'entrainer. Si figer la position 0 suffit a restaurer la diversite, tout
l'effondrement est localise dans le choix du determinant. Localise l'echec au
lieu de le constater.
"""

import json
import os
from itertools import product

import numpy as np
import torch

from grammar import Grammaire
from rl_grammar import (PolitiqueGRU, analyse_exacte, distribution_exacte,
                          fixer_graine, DOSSIER_SORTIE)
from capacity_probe import phrases_valides_en_ids, ajuster

EPISODES = 20000
PERIODE = 250
BETA = 0.02
LR = 1e-3


def kl_vers_ideal(politique, grammaire, indices_valides):
    """KL(pi || uniforme sur les valides), en bits. Infini si masse hors valides."""
    _, probas = distribution_exacte(politique, grammaire)
    p = probas.double().numpy()
    p = p / p.sum()
    q = np.zeros_like(p)
    q[indices_valides] = 1.0 / len(indices_valides)
    masque = p > 1e-300
    # KL(p||q) est infinie des que p met de la masse hors du support de q :
    # on rapporte donc separement la masse hors support et la KL sur le support.
    hors = float(p[q == 0].sum())
    sur = p[indices_valides]
    sur = sur / max(sur.sum(), 1e-300)
    kl_interne = float((sur * np.log2(np.clip(sur * len(indices_valides), 1e-300, None))).sum())
    return kl_interne, hors


def entrainer_avec_trace(politique, grammaire, indices_valides, episodes=EPISODES,
                         beta=BETA, lr=LR, periode=PERIODE, geler_position=None,
                         marginale_ideale=None, etiquette=""):
    """REINFORCE echantillonne, avec releve exact de l'etat tous les `periode`."""
    optimiseur = torch.optim.Adam(politique.parameters(), lr=lr)
    from collections import deque
    historique = deque(maxlen=100)
    trace = []
    for episode in range(1, episodes + 1):
        actions, log_probs, entropies, _ = politique.generer(
            grammaire.longueur, taille_lot=1)
        if geler_position is not None:
            # On remplace l'action de la position figee par un tirage dans la
            # marginale ideale : la politique n'a plus la main sur ce choix.
            forcee = int(np.random.choice(len(marginale_ideale), p=marginale_ideale))
            actions[0, geler_position] = forcee
        seq = actions[0].tolist()
        r = grammaire.recompense_graduee(seq)
        baseline = sum(historique) / len(historique) if historique else 0.0
        historique.append(r)
        avantage = torch.tensor(r - baseline, dtype=torch.float32)
        perte = -(log_probs.sum() * avantage) - beta * entropies.sum()
        optimiseur.zero_grad()
        perte.backward()
        torch.nn.utils.clip_grad_norm_(politique.parameters(), 5.0)
        optimiseur.step()

        if episode % periode == 0:
            ex = analyse_exacte(politique, grammaire)
            kl, hors = kl_vers_ideal(politique, grammaire, indices_valides)
            trace.append({"episode": episode, "modes": ex["modes_effectifs"],
                          "valide_pct": ex["masse_valide_pct"],
                          "kl_bits": round(kl, 4), "masse_hors_valides": round(hors, 5),
                          "sg": ex["repartition_familles"]["sg"],
                          "pl": ex["repartition_familles"]["pl"]})
    return trace


if __name__ == "__main__":
    g = Grammaire(longue=False)
    sequences = list(product(range(g.taille), repeat=g.longueur))
    cibles = phrases_valides_en_ids(g)
    index_par_sequence = {s: i for i, s in enumerate(sequences)}
    indices_valides = [index_par_sequence[c] for c in cibles]
    resultats = {}

    print("=" * 88)
    print("Q-A — STABILITE : REINFORCE part-il en derive depuis la politique IDEALE ?")
    print("=" * 88)
    fixer_graine(0)
    ideale = PolitiqueGRU(g.taille)
    ajuster(ideale, cibles, etapes=3000, lr=5e-3)
    ex0 = analyse_exacte(ideale, g)
    print(f"  depart (ideal)  : valide {ex0['masse_valide_pct']:6.2f} % | "
          f"{ex0['modes_effectifs']:5.1f} modes | "
          f"sg {ex0['repartition_familles']['sg']:.1f} / pl {ex0['repartition_familles']['pl']:.1f}")
    trace_a = entrainer_avec_trace(ideale, g, indices_valides, etiquette="depuis_ideal")
    for point in trace_a[::8]:
        print(f"    ep {point['episode']:6d} | valide {point['valide_pct']:6.2f} % | "
              f"{point['modes']:5.1f} modes | KL {point['kl_bits']:6.3f} bits | "
              f"sg {point['sg']:5.1f} / pl {point['pl']:5.1f}")
    resultats["depuis_ideal"] = trace_a
    print()

    print("=" * 88)
    print("Q-B — L'optimum est-il une ETAPE ? KL vers l'ideal depuis l'aleatoire")
    print("=" * 88)
    fixer_graine(0)
    depuis_zero = PolitiqueGRU(g.taille)
    trace_b = entrainer_avec_trace(depuis_zero, g, indices_valides, etiquette="depuis_zero")
    meilleur = min(trace_b, key=lambda t: t["kl_bits"])
    plus_de_modes = max(trace_b, key=lambda t: t["modes"])
    for point in trace_b[::8]:
        print(f"    ep {point['episode']:6d} | valide {point['valide_pct']:6.2f} % | "
              f"{point['modes']:5.1f} modes | KL {point['kl_bits']:6.3f} bits")
    print(f"  KL minimale      : {meilleur['kl_bits']:.4f} bits a l'episode "
          f"{meilleur['episode']} ({meilleur['modes']} modes)")
    print(f"  modes maximaux   : {plus_de_modes['modes']} a l'episode "
          f"{plus_de_modes['episode']}")
    print(f"  etat final       : {trace_b[-1]['modes']} modes, "
          f"KL {trace_b[-1]['kl_bits']:.4f} bits")
    gain = plus_de_modes["modes"] - trace_b[-1]["modes"]
    print(f"  -> {'UN ARRET PRECOCE BATTRAIT LA CONVERGENCE' if gain > 1 else 'pas de gain a arreter tot'} "
          f"(ecart {gain:+.1f} modes)")
    resultats["depuis_zero"] = trace_b
    print()

    print("=" * 88)
    print("Q-C — Quelle position porte l'effondrement ?")
    print("=" * 88)
    _, probas_ideales = distribution_exacte(ideale, g)
    p_ideal = probas_ideales.double().numpy()
    p_ideal /= p_ideal.sum()
    seq_arr = np.array(sequences)
    for position, categorie in enumerate(g.structure):
        marginale = np.array([p_ideal[seq_arr[:, position] == i].sum()
                              for i in range(g.taille)])
        marginale = marginale / marginale.sum()
        fixer_graine(0)
        politique = PolitiqueGRU(g.taille)
        trace = entrainer_avec_trace(politique, g, indices_valides, episodes=EPISODES,
                                     geler_position=position, marginale_ideale=marginale,
                                     periode=EPISODES)
        fin = trace[-1]
        print(f"  position {position} ({categorie:5s}) figee a l'ideal -> "
              f"valide {fin['valide_pct']:6.2f} % | {fin['modes']:5.1f} modes | "
              f"sg {fin['sg']:5.1f} / pl {fin['pl']:5.1f}")
        resultats[f"figee_{position}_{categorie}"] = trace

    with open(os.path.join(DOSSIER_SORTIE, "stabilite_trajectoire.json"), "w",
              encoding="utf-8") as f:
        json.dump(resultats, f, indent=2, ensure_ascii=False)
    print(f"\nEcrit dans {DOSSIER_SORTIE}")
