"""Suite de verifier_prior_asymetrique.py : le premier essai (poids 2/27
contre 1/27) a saute a R[10,4]=1.000000, pas 0.667 — ni l'une ni l'autre
des deux predictions de dipankarsarkar. Avant de conclure quoi que ce
soit, on balaie l'asymetrie en continu pour savoir si c'est un seuil
dur (tout ou rien des le moindre desequilibre) ou un vrai gradient que
mon premier pas de 2:1 a simplement depasse.

Prediction analytique a verifier (cf. lettre) : pour w4=(1+delta)/N,
w3=(1-delta)/N (poids MOYEN inchange, seul le partage entre les deux
change), la condition du premier ordre sur un objectif lineaire en r4
plus un terme d'entropie beta/N donne log(r4/(1-r4)) = 2*delta/beta
(le facteur N s'annule : w4-w3 = 2*delta/N, et la derivee de l'entropie
porte deja un 1/N). Avec beta=0.02, la transition attendue est donc a
l'echelle delta ~ beta/2 = 0.01, pas beta/N.
"""

import sys
sys.path.insert(0, '.')
import math
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import activer, parametres, N
from verifier_prior_asymetrique import objectif_pondere, continuer_sous_prior, etat

ADAM_EPS = 1e-10
PAS_SUITE_ADD = 15000

if __name__ == "__main__":
    torch.set_printoptions(precision=15)
    print("=== balayage du desequilibre de poids (w4 = (1+delta)/N, w3 = (1-delta)/N) ===")
    print(f"  echelle de transition attendue (delta ~ beta/2) : {BETA/2:.6e}")
    for delta in (0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1):
        poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
        poids[4] = (1.0 + delta) / N
        poids[3] = (1.0 - delta) / N
        e, r = construire_mur23(adam_eps=ADAM_EPS)
        continuer_sous_prior(e, r, BETA, PAS_SUITE_ADD, 0.05, ADAM_EPS, poids)
        R, H, m, Hb, S = etat(e, r)
        prediction = 1.0 / (1.0 + math.exp(-2 * delta / BETA)) if delta > 0 else 0.5
        print(f"  delta={delta:<8}  R[10,4]={R[4]:.6f}  R[10,3]={R[3]:.6f}  "
              f"H(27)={H:.6f}  prediction_analytique={prediction:.6f}")
