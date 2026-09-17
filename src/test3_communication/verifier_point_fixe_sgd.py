"""Tour 52, creuse seul (dipankarsarkar absent depuis 2 jours au
17/09/2026) : pourquoi verifier_excursions_sgd.py converge a
R[10,4]=0,786283 et pas 0,794756 (valeur de branche predite par la forme
fermee de dipankar) ?

Deux checks : (1) grain fin -- est-ce une approche lente ou un vrai point
fixe ? (2) norme du gradient -- si c'est vraiment un zero de J, quel est
l'etat des deux emetteurs a ce point ?
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere

DELTA = 0.013026615
PAS_LONG = 390_000


def construire_point_sgd(lr=50.0, pas=PAS_LONG):
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + DELTA) / N
    poids[3] = (1.0 - DELTA) / N
    e, r = construire_mur23(adam_eps=1e-10)
    activer(e, r)
    opt = torch.optim.SGD(parametres(e, r), lr=lr)
    for _ in range(pas):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
    return e, r, poids


if __name__ == "__main__":
    torch.set_printoptions(precision=12)
    e, r, poids = construire_point_sgd()

    print("=== grain fin : 200 pas de plus, logges un par un ===")
    opt = torch.optim.SGD(parametres(e, r), lr=50.0)
    for pas in range(200):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        if pas % 10 == 0:
            with torch.no_grad():
                s_, r_ = e.loi(), r.loi()
                R4 = (s_[4, 10] * r_[10, 4]).item()
            print(f"  pas={pas:<4}  R[10,4]={R4:.12f}")

    print("\n=== norme du gradient au point final, et etat des emetteurs ===")
    j, _ = objectif_pondere(e, r, BETA, poids)
    opt.zero_grad()
    (-j).backward()
    gnorm = sum((p.grad ** 2).sum().item() for p in parametres(e, r)) ** 0.5
    with torch.no_grad():
        s_, r_ = e.loi(), r.loi()
        print(f"  norme du gradient : {gnorm:.6e}")
        print(f"  s3={s_[3,10].item():.12f}  s4={s_[4,10].item():.12f}  "
              f"r4={r_[10,4].item():.12f}")
        print(f"  (baseline pre-perturbation, pour comparaison : s3=0.999999999666)")
