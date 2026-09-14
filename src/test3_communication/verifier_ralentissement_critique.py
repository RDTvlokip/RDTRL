"""Tour 50, deuxieme experience de dipankar : un noeud-col impose un
ralentissement critique, temps de convergence ~ (delta_c-delta)^(-1/2).
delta_c encadre entre 0,013422 et 0,013437 (verifier_bissection_delta_c.py).
On trace R[10,4] tous les 500 pas a trois distances de l'arete (~3%, ~0,3%,
~0,03% en dessous), et on mesure le premier pas ou R passe sous 1% de sa
valeur d'ecart final au plafond mou (convergence a 1% pres).
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere

ADAM_EPS = 1e-10
DELTA_C = 0.0134295  # milieu de l'encadrement (0.013422, 0.013437)
PAS_MAX = 60000
CHECK_TOUS = 200


def trace(delta, pas_max=PAS_MAX):
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + delta) / N
    poids[3] = (1.0 - delta) / N
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=0.05, eps=ADAM_EPS)
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
    print(f"=== ralentissement critique, delta_c ~ {DELTA_C} ===")
    for frac in distances_relatives:
        delta = DELTA_C * (1 - frac)
        traj, R_final = trace(delta)
        cible = 0.99 * R_final if R_final < 1.0 else 0.99
        premier_pas_a_1pct = None
        for pas, R4 in traj:
            if R4 >= cible:
                premier_pas_a_1pct = pas
                break
        print(f"  delta={delta:.7f}  ({frac*100:.3f}% sous delta_c)  "
              f"R_final={R_final:.6f}  premier pas a 1% de la cible = {premier_pas_a_1pct}")
