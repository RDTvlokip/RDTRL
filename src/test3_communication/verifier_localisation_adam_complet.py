"""Verification independante (18/09-20/09/2026) du test de localisation
precommis par dipankarsarkar : sous Adam COMPLET (emetteur ET recepteur
adaptatifs), le rapport local s3_dip/R_dip mesure sous l'optimiseur
HYBRIDE (Adam-emetteur / SGD-recepteur, verifier_optimiseur_hybride.py)
tient-il encore ? Sa prediction precommise : R_dip~1.956e-3 sous Adam
complet devrait s'accompagner de s3_dip ~ R_dip * ratio_hybride, et donc
d'une chute de s3 vers ~0.957.

Ecrit independamment (pas copie) a partir de replay_mur23_referent3.py
et verifier_prior_asymetrique.py, pour comparer a un calcul deja fait
ce jour sans lui faire confiance sur parole (regle 5bis).
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
LR = 0.05
ADAM_EPS = 1e-10


def main():
    torch.set_printoptions(precision=10)
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + DELTA) / N
    poids[3] = (1.0 - DELTA) / N

    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=LR, eps=ADAM_EPS)

    print(f"=== Adam COMPLET (les deux joueurs), delta={DELTA}, {PAS_MAX} pas ===")
    s3_min, R_max_dip = 1.0, 0.0
    for pas in range(PAS_MAX):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        if pas % CHECK_TOUS == 0:
            with torch.no_grad():
                s_, r_ = e.loi(), r.loi()
                R4 = (s_[4, 10] * r_[10, 4]).item()
                s3 = s_[3, 10].item()
            print(f"  pas={pas:<7}  R[10,4]={R4:.10f}  s3={s3:.10f}")
            s3_min = min(s3_min, s3)
    with torch.no_grad():
        s_, r_ = e.loi(), r.loi()
        R_final = (s_[4, 10] * r_[10, 4]).item()
        s3_final = s_[3, 10].item()
        print(f"  final       R[10,4]={R_final:.10f}  s3={s3_final:.10f}")
        print(f"  s3_min observe sur la fenetre de check = {s3_min:.10f}")


if __name__ == "__main__":
    main()
