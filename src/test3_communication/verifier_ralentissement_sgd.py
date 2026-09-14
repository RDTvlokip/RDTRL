"""Pourquoi le test de ralentissement critique (Adam) n'a rien montre :
hypothese H15, formee en derivant H7. Adam normalise son pas par
1/sqrt(second moment), qui suit le gradient de pres -- pres d'un vrai
noeud-col, le gradient s'annule mais le pas normalise d'Adam peut rester
quasi constant (c'est precisement ce pour quoi Adam est concu : rester
insensible a l'echelle du gradient). Si le systeme est un vrai noeud-col
sous le FLOT DE GRADIENT NU, Adam peut avoir efface la signature de
ralentissement critique sans que la bifurcation soit fausse pour autant.

Meme protocole que verifier_ralentissement_critique.py, mais SGD pur
(pas de normalisation adaptative) a la place d'Adam, learning rate
choisi pour donner un pas comparable en ordre de grandeur au debut.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere

DELTA_C = 0.0134295
PAS_MAX = 60000
CHECK_TOUS = 200
LR_SGD = 50.0  # calibre empiriquement pour un pas net comparable a Adam ici


def trace_sgd(delta, pas_max=PAS_MAX, lr=LR_SGD):
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + delta) / N
    poids[3] = (1.0 - delta) / N
    e, r = construire_mur23(adam_eps=1e-10)
    activer(e, r)
    opt = torch.optim.SGD(parametres(e, r), lr=lr)
    trajectoire = []
    for pas in range(pas_max):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        if pas % CHECK_TOUS == 0:
            with torch.no_grad():
                s_, r_ = e.loi(), r.loi()
                R4 = (s_[4, 10] * r_[10, 4]).item()
            trajectoire.append((pas, R4))
    with torch.no_grad():
        s_, r_ = e.loi(), r.loi()
        R4_final = (s_[4, 10] * r_[10, 4]).item()
    return trajectoire, R4_final


if __name__ == "__main__":
    torch.set_printoptions(precision=6)
    distances_relatives = (0.03, 0.003, 0.0003)
    print(f"=== ralentissement critique SOUS SGD PUR (pas Adam), delta_c ~ {DELTA_C} ===")
    print(f"  lr={LR_SGD}")
    for frac in distances_relatives:
        delta = DELTA_C * (1 - frac)
        traj, R_final = trace_sgd(delta)
        cible = 0.99 * R_final if R_final < 1.0 else 0.99
        premier_pas_a_1pct = None
        for pas, R4 in traj:
            if R4 >= cible:
                premier_pas_a_1pct = pas
                break
        print(f"  delta={delta:.7f}  ({frac*100:.3f}% sous delta_c)  "
              f"R_final={R_final:.6f}  premier pas a 1% de la cible = {premier_pas_a_1pct}")
