"""Suite de verifier_sgd_pondere.py : les deux essais de SGD a deux taux
ont echoue (calibration tenseur entier invalide, puis gradient precis
trop minuscule pour un lr fixe sans risque). Piste retenue : optimiseur
HYBRIDE -- Adam sur l'emetteur seul (pour absorber la variation du
gradient sur des ordres de grandeur au fil de l'entrainement), SGD pur
sur le recepteur seul.

Question posee : est-ce l'adaptativite du RECEPTEUR ou celle de
L'EMETTEUR qui porte les excursions vues sous Adam pur (tour 51/52,
pas=160000, R retombe brievement a ~0,7928) ? Si les excursions
persistent avec ce montage (emetteur Adam, recepteur SGD), elles
viennent du cote emetteur. Si elles disparaissent, elles viennent du
cote recepteur -- meme conclusion que "zero excursion sous SGD pur"
serait alors correcte, mais pour une meilleure raison methodologique.
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
LR_ADAM_E = 0.05
ADAM_EPS = 1e-10
LR_SGD_R = 50.0


if __name__ == "__main__":
    torch.set_printoptions(precision=10)
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + DELTA) / N
    poids[3] = (1.0 - DELTA) / N

    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    opt_e = torch.optim.Adam(e.p, lr=LR_ADAM_E, eps=ADAM_EPS)
    opt_r = torch.optim.SGD(r.p, lr=LR_SGD_R)

    print(f"=== hybride : Adam sur emetteur (lr={LR_ADAM_E}), SGD sur recepteur (lr={LR_SGD_R}) ===")
    print(f"    delta={DELTA}, {PAS_MAX} pas")
    for pas in range(PAS_MAX):
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
                s3 = s_[3, 10].item()
            print(f"  pas={pas:<7}  R[10,4]={R4:.10f}  s3={s3:.10f}")
    with torch.no_grad():
        s_, r_ = e.loi(), r.loi()
        print(f"  final       R[10,4]={(s_[4,10]*r_[10,4]).item():.10f}  s3={s_[3,10].item():.10f}")
