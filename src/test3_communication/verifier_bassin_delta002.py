"""Pourquoi delta=0,02 saute a R[10,4]=1,000000 au lieu de suivre la loi
molle (prediction 0,880797) qui colle exactement jusqu'a delta=0,01 ?

Cinq hypotheses, testees plutot qu'enoncees :

H1 (sous-entrainement) : 40000 pas de plus ne suffisent pas, la
   trajectoire est encore EN ROUTE vers 0,88, pas arrivee a un autre
   point. Teste : prolonger a 200000 pas et voir si ca redescend.

H2 (plancher adam_eps, meme mecanisme que les murs precedents) : le
   gradient restaurateur (entropie) devient trop petit face au plancher
   additif d'Adam une fois pres de la saturation, gelant le systeme
   dans un coin au lieu de le laisser flotter jusqu'a 0,88. Teste :
   adam_eps beaucoup plus petit (1e-14) a delta=0,02.

H3 (pas d'apprentissage trop grand, artefact de la dynamique) : lr=0,05
   est trop grand une fois le paysage devenu plus raide pres du nouvel
   optimum interieur, et pousse la trajectoire au-dela d'un bord de
   bassin. Teste : lr beaucoup plus petit (0,005) a delta=0,02.

H4 (bifurcation reelle : l'optimum interieur cesse d'etre stable sous
   Adam au-dela d'un delta critique, meme s'il reste l'optimum
   analytique statique du J contraint) : si H1, H2 et H3 echouent tous
   a ramener le systeme vers 0,88, c'est la lecture qui reste.

H5 (dependance au chemin / hysteresis) : partir du point symetrique deja
   converge (comme construire_mur23 le fait) plutot que d'imposer le
   poids asymetrique des le debut pourrait biaiser vers le coin. Teste :
   entrainer depuis l'etat au checkpoint 10k (avant la perturbation et
   la convergence a 0,5) directement sous le poids asymetrique.
"""

import sys
sys.path.insert(0, '.')
import math
import torch

from replay_mur23_referent3 import construire_mur23, replay, BETA
from representable_atteignable_stable import activer, parametres, N
from verifier_prior_asymetrique import objectif_pondere, continuer_sous_prior, etat

DELTA = 0.02
PREDICTION_MOLLE = 1.0 / (1.0 + math.exp(-2 * DELTA / BETA))
poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
poids[4] = (1.0 + DELTA) / N
poids[3] = (1.0 - DELTA) / N

print(f"delta={DELTA}  prediction molle={PREDICTION_MOLLE:.6f}")

print("\n=== H1 : prolonger a 200000 pas de plus (au lieu de 40000+15000) ===")
e, r = construire_mur23(adam_eps=1e-10)
continuer_sous_prior(e, r, BETA, 200_000, 0.05, 1e-10, poids)
R, H, m, Hb = etat(e, r)
print(f"  R[10,4]={R[4]:.6f}  H={H:.6f}")

print("\n=== H2 : adam_eps beaucoup plus petit (1e-14) ===")
e, r = construire_mur23(adam_eps=1e-10)
continuer_sous_prior(e, r, BETA, 40_000, 0.05, 1e-14, poids)
R, H, m, Hb = etat(e, r)
print(f"  R[10,4]={R[4]:.6f}  H={H:.6f}")

print("\n=== H3 : lr beaucoup plus petit (0,005) ===")
e, r = construire_mur23(adam_eps=1e-10)
continuer_sous_prior(e, r, BETA, 40_000, 0.005, 1e-10, poids)
R, H, m, Hb = etat(e, r)
print(f"  R[10,4]={R[4]:.6f}  H={H:.6f}")

print("\n=== H5 : poids asymetrique impose DES le checkpoint 10k (pas apres convergence a 0,5) ===")
e, r = replay(77777, 3, 10000)
with torch.no_grad():
    e.p[0][4, 10] += 30.0
continuer_sous_prior(e, r, BETA, 40_000, 0.05, 1e-10, poids)
R, H, m, Hb = etat(e, r)
print(f"  R[10,4]={R[4]:.6f}  H={H:.6f}")
