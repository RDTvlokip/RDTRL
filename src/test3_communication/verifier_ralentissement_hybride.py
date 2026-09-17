"""Etape 1 de ETAT.md apres la resolution de H15 : refaire le test de
ralentissement critique (§7.63/§7.64) avec l'optimiseur qui entraine
VRAIMENT le referent 3 -- hybride Adam-emetteur / SGD-recepteur -- au
lieu du SGD pur qui le gelait. Meme protocole que
verifier_ralentissement_critique.py (trois distances a delta_c, temps
de convergence a 99% de la valeur finale), mais avec le bon optimiseur.

Si le ralentissement critique apparait maintenant (temps de convergence
qui augmente en approchant delta_c), ca renverse le verdict de §7.63/64
(H6 noeud-col) et relance H11 (crise de bord). Si le temps reste plat
meme avec le referent 3 qui s'entraine vraiment, H6 tient bel et bien.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere

DELTA_C = 0.0134295
PAS_MAX = 60_000
CHECK_TOUS = 200
LR_ADAM_E = 0.05
ADAM_EPS = 1e-10
LR_SGD_R = 50.0


def trace_hybride(delta, pas_max=PAS_MAX):
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + delta) / N
    poids[3] = (1.0 - delta) / N
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    opt_e = torch.optim.Adam(e.p, lr=LR_ADAM_E, eps=ADAM_EPS)
    opt_r = torch.optim.SGD(r.p, lr=LR_SGD_R)
    trajectoire = []
    for pas in range(pas_max):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt_e.zero_grad()
        opt_r.zero_grad()
        (-j).backward()
        opt_e.step()
        opt_r.step()
        if pas % CHECK_TOUS == 0:
            with torch.no_grad():
                s_, r_ = e.loi(), r.loi()
                R4 = (s_[4, 10] * r_[10, 4]).item()
            trajectoire.append((pas, R4))
    with torch.no_grad():
        s_, r_ = e.loi(), r.loi()
        R4f = (s_[4, 10] * r_[10, 4]).item()
    return trajectoire, R4f


if __name__ == "__main__":
    torch.set_printoptions(precision=6)
    distances_relatives = (0.03, 0.003, 0.0003)
    print(f"=== ralentissement critique, optimiseur hybride, delta_c ~ {DELTA_C} ===")
    for frac in distances_relatives:
        delta = DELTA_C * (1 - frac)
        traj, R_final = trace_hybride(delta)
        cible = 0.99 * R_final if R_final < 1.0 else 0.99
        premier_pas_a_1pct = None
        for pas, R4 in traj:
            if R4 >= cible:
                premier_pas_a_1pct = pas
                break
        print(f"  delta={delta:.7f}  ({frac*100:.3f}% sous delta_c)  "
              f"R_final={R_final:.6f}  premier pas a 1% de la cible = {premier_pas_a_1pct}")
