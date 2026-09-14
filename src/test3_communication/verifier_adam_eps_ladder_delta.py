"""Tour 49 : question 1 de dipankarsarkar, repondue avec une localisation
plutot qu'une fourchette. Teste l'echelle adam_eps (1e-8, 1e-10, 1e-12,
1e-14, la meme que replay_mur23_referent3.py utilise deja) a un delta
FIXE choisi DANS la region graduee (0,015, entre les deux points ou la
loi molle collait encore a quatre decimales), pas a delta=0,02 ou tout
est deja sature.

Si le point trouve bouge avec adam_eps ici, le meme loquet que les murs
(H2) est bien en jeu, meme s'il n'a pas suffi a expliquer la saturation
totale a delta=0,02. S'il ne bouge pas du tout, adam_eps n'est pour rien
dans ce mecanisme precis.
"""

import sys
sys.path.insert(0, '.')
import math
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import N
from verifier_prior_asymetrique import continuer_sous_prior, etat

DELTA = 0.013
PAS_SUITE_ADD = 40000

if __name__ == "__main__":
    torch.set_printoptions(precision=15)
    prediction = 1.0 / (1.0 + math.exp(-2 * DELTA / BETA))
    print(f"=== delta={DELTA} (region graduee), echelle adam_eps ===")
    print(f"  prediction molle (invariante a adam_eps si la loi tient) : {prediction:.6f}")
    for adam_eps in (1e-8, 1e-10, 1e-12, 1e-14):
        poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
        poids[4] = (1.0 + DELTA) / N
        poids[3] = (1.0 - DELTA) / N
        e, r = construire_mur23(adam_eps=adam_eps)
        continuer_sous_prior(e, r, BETA, PAS_SUITE_ADD, 0.05, adam_eps, poids)
        R, H, m, Hb, S = etat(e, r)
        print(f"  adam_eps={adam_eps:.0e}  R[10,4]={R[4]:.6f}  s[3,10]={S[3]:.6e}  H(27)={H:.6f}")
