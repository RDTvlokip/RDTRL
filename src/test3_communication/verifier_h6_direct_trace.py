"""Reference pour la comparaison de piste 3c (le renversement de bassin
sous masse de fond, et si son point selle est le MEME que H6 d'origine
ou un voisin). Rejoue le test H6 d'origine (verifier_sonde_bassin.py,
round 2 : s3 ET R places directement sur le point instable predit,
0,994300 / 0,829390) mais avec un tracage fin des gradients (comme
verifier_trajectoire_renversement.py) au lieu d'un instantane final
seul -- pour pouvoir comparer la FORME de la traversee du point selle
(deux phases lente/rapide, taux local, prefacteur C) entre les deux
contextes.

Resultat deja obtenu (18/09/2026, voir CARNET.md fin de §7.65) : meme
structure qualitative a deux phases, mais taux/duree locale differents
d'un facteur ~6-17x pour un mu comparable -- point selle DE LA MEME
FAMILLE que H6, pas le meme point exact.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import activer, parametres
from verifier_prior_asymetrique import objectif_pondere
from verifier_sonde_bassin import fixer_s3, fixer_r4, poids_delta

DELTA = 0.013
ADAM_EPS = 1e-10
SEUIL_H6 = 0.994300
R_UNSTABLE_H6 = 0.829390


def entrainer_trace_h6(cible_s3, R_init=R_UNSTABLE_H6, pas=400, check_tous=1):
    poids = poids_delta(DELTA)
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    fixer_s3(e, cible_s3)
    fixer_r4(r, R_init)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=0.05, eps=ADAM_EPS)
    trace = []
    for i in range(pas):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        if i % check_tous == 0:
            with torch.no_grad():
                g_e3 = e.p[0].grad[3, 10].item()
                s3 = e.loi()[3, 10].item()
            trace.append((i, s3, g_e3))
        opt.step()
    return trace


if __name__ == "__main__":
    torch.set_printoptions(precision=6)
    for label, cible_s3 in [("SOUS le seuil (0,994295, s'effondre vraiment)", SEUIL_H6 - 5e-6),
                             ("AU-DESSUS du seuil (0,994305, reste gradue)", SEUIL_H6 + 5e-6)]:
        print(f"=== {label}, cible_s3={cible_s3} ===")
        trace = entrainer_trace_h6(cible_s3, pas=60, check_tous=1)
        for (i, s3, g_e3) in trace:
            print(f"  pas={i:3d}  s3={s3:.6f}  g_e3={g_e3:.6e}")
