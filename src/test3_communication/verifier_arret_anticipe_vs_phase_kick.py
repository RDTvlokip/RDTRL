"""Question 13 des 20 (ETAT.md, 21/09/2026) : l'arret anticipe sur
"la perte n'a pas progresse depuis N pas" est-il vulnerable au meme
hasard de timing qu'un cycle plancher-de-v (~456-500 pas mesures ici,
verifier_kicks_adam_grille_fine.py) -- un run arrete trop tot ou trop
tard selon ou dans le cycle la fenetre de patience est tombee ?

Test direct sur le vrai systeme (mur 23) : on trace la trajectoire de
l'objectif j (pas R4 directement -- j est ce qu'un early-stopping
reel surveillerait) sur une longue fenetre en regime stationnaire de
kicks, puis on simule un critere d'arret anticipe classique
("stopper si aucune amelioration de plus de EPS_AMELIORATION pendant
PATIENCE pas") en faisant varier le pas de DEPART du compteur de
patience sur environ un plein cycle de kick. Si le pas d'arret (ou le
fait meme de s'arreter avant PAS_MAX) depend fortement du depart
choisi, la vulnerabilite est confirmee empiriquement, pas seulement
plausible par analogie.

Prediction fermee, posee avant le test : si PATIENCE < periode du
cycle de kicks (~460-500), alors selon que le depart tombe juste avant
ou juste apres un kick, le compteur va soit s'armer sans jamais voir
de kick avant PATIENCE pas (arret anticipe declenche), soit voir un
kick qui reinitialise le compteur (le run continue bien au-dela). Le
pas d'arret devrait donc varier de facon quasi discontinue (par sauts
d'un cycle entier) selon la phase de depart, pas varier en douceur.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere

DELTA = 0.013026615
LR = 0.05
ADAM_EPS = 1e-10
PAS_MAX = 20000
PATIENCES = (200, 400, 600)
EPS_AMELIORATION_VALEURS = (1e-8, 3e-9, 1e-9)
DEBUTS_TESTES = list(range(9700, 10700, 40))


def tracer():
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + DELTA) / N
    poids[3] = (1.0 - DELTA) / N
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=LR, eps=ADAM_EPS)
    vals = []
    for pas in range(PAS_MAX):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        vals.append(j.item())
    return vals


def simuler_arret(vals, debut, patience, eps_amelioration):
    meilleur = vals[debut]
    pas_depuis_amelioration = 0
    for pas in range(debut + 1, len(vals)):
        if vals[pas] > meilleur + eps_amelioration:
            meilleur = vals[pas]
            pas_depuis_amelioration = 0
        else:
            pas_depuis_amelioration += 1
            if pas_depuis_amelioration >= patience:
                return pas
    return None


def main():
    vals = tracer()
    for eps_amelioration in EPS_AMELIORATION_VALEURS:
        for patience in PATIENCES:
            resultats = []
            for debut in DEBUTS_TESTES:
                pas_arret = simuler_arret(vals, debut, patience, eps_amelioration)
                duree = (pas_arret - debut) if pas_arret is not None else None
                resultats.append((debut, pas_arret, duree))
            durees_valides = [d for _, _, d in resultats if d is not None]
            n_arretes = len(durees_valides)
            print(f"eps={eps_amelioration}  patience={patience}  "
                  f"n_departs={len(DEBUTS_TESTES)}  n_arretes_avant_pas_max={n_arretes}")
            if durees_valides:
                print(f"  duree jusqu'a arret : min={min(durees_valides)}  "
                      f"max={max(durees_valides)}  "
                      f"(rapport max/min = {max(durees_valides)/min(durees_valides):.2f})")


if __name__ == "__main__":
    main()
