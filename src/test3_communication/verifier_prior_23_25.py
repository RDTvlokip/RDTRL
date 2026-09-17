"""Generalisation du mecanisme k(R)/delta_c (tours 48-52, referents 3/4)
a une DEUXIEME collision independante : referents 23/25, message 13,
retrouvee dans replay_23_25.py. Meme protocole que
verifier_prior_asymetrique.py : reponderer un referent contre l'autre
et voir si un delta_c similaire apparait.

Ici, poids[25] = (1+delta)/N, poids[23] = (1-delta)/N -- referent 25
favorise, referent 23 defavorise, symetrie inverse de 3/4 par pur choix
arbitraire (les deux referents sont interchangeables au demarrage,
S≈0,5/0,5).
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_23_25 import construire, BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere, continuer_sous_prior, etat

ADAM_EPS = 1e-10
PAS_SUITE = 40000
MESSAGE = 13
REFS = (23, 25)


def etat_23_25(e, r):
    return etat(e, r, msg=MESSAGE, refs=REFS)


if __name__ == "__main__":
    torch.set_printoptions(precision=6)

    print("=== etat de depart (egalite naturelle 23/25, message 13) ===")
    e0, r0 = construire()
    R0, H0, m0, Hb0, S0 = etat_23_25(e0, r0)
    print(f"  R[13,25]={R0[25]:.6f}  R[13,23]={R0[23]:.6f}  "
          f"H(27 voies)={H0:.6f}  masse(23+25)={m0:.9f}  ln2={torch.log(torch.tensor(2.0)).item():.6f}")

    print("\n=== balayage delta (poids[25]=(1+d)/N, poids[23]=(1-d)/N) ===")
    for delta in (0.0, 0.001, 0.005, 0.01, 0.013, 0.02, 0.05):
        poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
        poids[25] = (1.0 + delta) / N
        poids[23] = (1.0 - delta) / N
        e, r = construire()
        continuer_sous_prior(e, r, BETA, PAS_SUITE, 0.05, ADAM_EPS, poids)
        R, H, m, Hb, S = etat_23_25(e, r)
        print(f"  delta={delta:<6}  R[13,25]={R[25]:.6f}  s23={S[23]:.9f}  s25={S[25]:.9f}  H={H:.6f}")
