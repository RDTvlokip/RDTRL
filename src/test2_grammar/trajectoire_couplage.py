"""Quand le couplage se decide-t-il, et qu'est-ce qui le decide ?

Question 2 de l'approfondissement du 31/07/2026.

Ce que j'avais mal lu dans ma propre table : les runs a gradient exact qui
atteignent 24 modes a beta = 0,02 ne sont jamais passes par 12. Ils y vont
DEPUIS L'INITIALISATION. Graines 0 et 2 finissent a 24, graine 1 a 12, meme beta,
meme code, meme architecture.

Donc la question n'est pas "pourquoi l'echantillonne ne franchit-il pas le
plafond", c'est "qu'est-ce qui decide, et quand, si I(det;nom) decolle ou reste
a zero".

On suit I(det ; nom) en bits pas a pas, sur les deux procedures. Trois choses a
lire :
  - l'instant de separation : a partir de quel pas les trajectoires a 24 et a 12
    sont-elles distinguables ;
  - si la separation precede ou suit l'effondrement de la validite ;
  - s'il existe un predicteur A L'INITIALISATION, c'est-a-dire avant le premier
    pas de gradient.
"""

import argparse
import json
import os
from collections import deque
from itertools import product as iproduct

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

torch.set_num_threads(1)

from grammaire import Grammaire
from rl_grammaire import PolitiqueGRU, fixer_graine, DOSSIER_SORTIE
from optimum_produit import contexte, mesures


def sonder(politique, sequences, recompenses, valide, g):
    """Mesure exacte de l'etat courant, sans echantillonnage."""
    debut = torch.full((len(sequences), 1), politique.token_debut, dtype=torch.long)
    entrees = torch.cat([debut, sequences[:, :-1]], dim=1)
    with torch.no_grad():
        sorties, _ = politique.gru(politique.embedding(entrees))
        lp = torch.log_softmax(politique.tete(sorties), dim=-1)
        log_p = lp.gather(2, sequences.unsqueeze(-1)).squeeze(-1).sum(1)
    p = log_p.double().exp()
    m = mesures(p, sequences, recompenses, valide, g)
    jointe = (p / p.sum()).reshape([g.taille] * g.longueur)
    i_det = g.positions["det"]
    autres = tuple(k for k in range(g.longueur) if k != i_det)
    masse_det = jointe.sum(dim=autres)
    m["masse_determinants"] = {d: round(float(masse_det[g.index[d]]), 6)
                               for d in sorted(g.tokens_par_categorie["det"])}
    return m


def entrainer_suivi(politique, g, sequences, recompenses, valide, beta,
                    exact, etapes, periode):
    """Une trajectoire, sondee tous les `periode` pas. exact=True : gradient exact."""
    lr = 5e-3 if exact else 1e-3
    opt = torch.optim.Adam(politique.parameters(), lr=lr)
    historique = [{"pas": 0, **sonder(politique, sequences, recompenses, valide, g)}]
    memoire = deque(maxlen=100)

    for pas in range(1, etapes + 1):
        if exact:
            debut = torch.full((len(sequences), 1), politique.token_debut, dtype=torch.long)
            entrees = torch.cat([debut, sequences[:, :-1]], dim=1)
            sorties, _ = politique.gru(politique.embedding(entrees))
            lp = torch.log_softmax(politique.tete(sorties), dim=-1)
            log_p = lp.gather(2, sequences.unsqueeze(-1)).squeeze(-1).sum(1)
            p = log_p.exp()
            p = p / p.sum()
            perte = -((p * recompenses.float()).sum()
                      + beta * (-(p * p.clamp_min(1e-30).log()).sum()))
        else:
            actions, log_probs, entropies, _ = politique.generer(g.longueur, taille_lot=1)
            r = g.recompense_graduee(actions[0].tolist())
            base = sum(memoire) / len(memoire) if memoire else 0.0
            memoire.append(r)
            avantage = torch.tensor(r - base).detach()
            perte = -(log_probs.sum() * avantage) - beta * entropies.sum()

        opt.zero_grad()
        perte.backward()
        torch.nn.utils.clip_grad_norm_(politique.parameters(), 5.0)
        opt.step()

        if pas % periode == 0:
            historique.append({"pas": pas,
                               **sonder(politique, sequences, recompenses, valide, g)})
    return historique


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--beta", type=float, default=0.02)
    p.add_argument("--graines", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--etapes-exact", type=int, default=4000)
    p.add_argument("--etapes-echantillonne", type=int, default=20000)
    p.add_argument("--periode-exact", type=int, default=25)
    p.add_argument("--periode-echantillonne", type=int, default=200)
    p.add_argument("--exact-seulement", action="store_true")
    args = p.parse_args()

    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    g = Grammaire(longue=False)
    sequences, recompenses, valide = contexte(g)
    rapport = {"beta": args.beta, "trajectoires": []}

    modes = [("exact", True, args.etapes_exact, args.periode_exact)]
    if not args.exact_seulement:
        modes.append(("echantillonne", False, args.etapes_echantillonne,
                      args.periode_echantillonne))

    for nom, exact, etapes, periode in modes:
        print("=" * 78)
        print(f"{nom.upper()} — beta = {args.beta}, {etapes} pas, sonde tous les {periode}")
        print("=" * 78)
        for graine in args.graines:
            fixer_graine(graine)
            politique = PolitiqueGRU(g.taille)
            h = entrainer_suivi(politique, g, sequences, recompenses, valide,
                                args.beta, exact, etapes, periode)
            fin = h[-1]
            # Premier pas ou I depasse durablement 0,05 bit
            decollage = next((e["pas"] for e in h
                              if e["information_mutuelle_det_nom_bits"] > 0.05), None)
            rapport["trajectoires"].append({"procedure": nom, "graine": graine,
                                            "decollage_I": decollage, "historique": h})
            print(f"  graine {graine} : fin {fin['modes_effectifs']:5.2f} modes | "
                  f"I = {fin['information_mutuelle_det_nom_bits']:.4f} bits | "
                  f"valide {fin['masse_valide_pct']:.2f} % | "
                  f"decollage de I au pas {decollage}")
            print(f"    I au depart : {h[0]['information_mutuelle_det_nom_bits']:.4f} bits | "
                  f"masse des determinants a l'init : "
                  f"{ {k: round(v, 4) for k, v in h[0]['masse_determinants'].items()} }")

    chemin = os.path.join(DOSSIER_SORTIE, f"trajectoire_couplage_b{args.beta}.json")
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False)
    print(f"\nEcrit dans {chemin}")


if __name__ == "__main__":
    main()
