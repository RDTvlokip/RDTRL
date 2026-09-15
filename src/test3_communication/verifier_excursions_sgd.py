"""Tour 52 de dipankarsarkar : question directe -- « Does your SGD arm
show the excursions at all? » Si les excursions vues sous Adam (tour
51, §7.64 : pas=160 000, R retombe brievement a ~0,7928) sont un mode
propre du systeme linearise, elles doivent apparaitre sous n'importe
quel optimiseur. Si elles viennent du second moment d'Adam (le rapport
k derive parce que l'echelle adaptative n'est pas la meme pour
l'emetteur et le recepteur), SGD ne doit rien montrer.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere

DELTA = 0.013026615
PAS_MAX = 400_000
CHECK_TOUS = 20_000
LR_SGD = 50.0


if __name__ == "__main__":
    torch.set_printoptions(precision=6)
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + DELTA) / N
    poids[3] = (1.0 - DELTA) / N

    e, r = construire_mur23(adam_eps=1e-10)
    activer(e, r)
    opt = torch.optim.SGD(parametres(e, r), lr=LR_SGD)
    print(f"=== SGD pur, delta={DELTA}, {PAS_MAX} pas, lr={LR_SGD} ===")
    for pas in range(PAS_MAX):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        if pas % CHECK_TOUS == 0:
            with torch.no_grad():
                s_, r_ = e.loi(), r.loi()
                R4 = (s_[4, 10] * r_[10, 4]).item()
            print(f"  pas={pas:<7}  R[10,4]={R4:.10f}")
    with torch.no_grad():
        s_, r_ = e.loi(), r.loi()
        print(f"  final       R[10,4]={(s_[4,10]*r_[10,4]).item():.10f}")
