"""Trace fine (blocs de 20 pas) sous continuer_sous_prior, pour lire la
trajectoire naturelle de s3/s4/R3/R4 pas a pas pres du mur 23 -- utilise
comme sonde rapide avant de lancer un calcul plus long.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from verifier_prior_asymetrique import continuer_sous_prior, etat
from verifier_sonde_bassin import poids_delta
from representable_atteignable_stable import activer, parametres

ADAM_EPS = 1e-10
DELTA = 0.013
PAS_TOTAL = 4000
PAS_BLOC = 20

poids = poids_delta(DELTA)

e, r = construire_mur23(adam_eps=ADAM_EPS)
R, H, m, Hb, S = etat(e, r)
print(f"t=0  s3={S[3]:.6f}  s4={S[4]:.6f}  R4={R[4]:.6f}  R3={R[3]:.6f}")

for bloc in range(PAS_TOTAL // PAS_BLOC):
    continuer_sous_prior(e, r, BETA, PAS_BLOC, 0.05, ADAM_EPS, poids)
    t = (bloc + 1) * PAS_BLOC
    R, H, m, Hb, S = etat(e, r)
    print(f"t={t}  s3={S[3]:.6f}  s4={S[4]:.6f}  R4={R[4]:.6f}  R3={R[3]:.6f}")
