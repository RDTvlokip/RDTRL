"""Complement du 20/09/2026 a verifier_localisation_adam_complet.py :
la grille a CHECK_TOUS=20000 pas pourrait rater une excursion plus
profonde de s3 (ou de R) qui se produit ET se resorbe ENTRE deux points
de mesure -- exactement le risque de "single frame of something
oscillating" que dipankarsarkar a deja signale pour une trace proche.

Ici on rejoue le meme systeme (Adam complet, delta=0.013026615) mais
avec un pas de log FIN (tous les 200 pas) sur les fenetres ou le run
grossier a deja vu une excursion de R (autour de pas=60000, 160000,
260000, 320000), pour verifier si s3 descend plus bas entre les points
grossiers que ce que le log grossier montrait.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere

DELTA = 0.013026615
LR = 0.05
ADAM_EPS = 1e-10
CHECK_FIN = 200

FENETRES = [(40000, 80000), (140000, 180000), (240000, 280000), (300000, 340000)]


def main():
    torch.set_printoptions(precision=10)
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + DELTA) / N
    poids[3] = (1.0 - DELTA) / N

    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=LR, eps=ADAM_EPS)

    pas_courant = 0
    s3_global_min = 1.0
    R_global_max_dip = 0.0
    R_baseline = None

    for (debut, fin) in FENETRES:
        # avancer jusqu'au debut de la fenetre au pas grossier
        while pas_courant < debut:
            j, _ = objectif_pondere(e, r, BETA, poids)
            opt.zero_grad()
            (-j).backward()
            opt.step()
            pas_courant += 1
        print(f"\n=== fenetre fine [{debut}, {fin}] ===")
        s3_min_fenetre = 1.0
        R_min_fenetre = 1.0
        while pas_courant < fin:
            j, _ = objectif_pondere(e, r, BETA, poids)
            opt.zero_grad()
            (-j).backward()
            opt.step()
            pas_courant += 1
            if pas_courant % CHECK_FIN == 0:
                with torch.no_grad():
                    s_, r_ = e.loi(), r.loi()
                    R4 = (s_[4, 10] * r_[10, 4]).item()
                    s3 = s_[3, 10].item()
                if R_baseline is None:
                    R_baseline = R4
                s3_min_fenetre = min(s3_min_fenetre, s3)
                R_min_fenetre = min(R_min_fenetre, R4)
                if pas_courant % 2000 == 0 or R4 < R_baseline - 0.001:
                    print(f"  pas={pas_courant:<7}  R[10,4]={R4:.10f}  s3={s3:.10f}")
        print(f"  min sur fenetre : s3_min={s3_min_fenetre:.10f}  R_min={R_min_fenetre:.10f}")
        s3_global_min = min(s3_global_min, s3_min_fenetre)

    print(f"\n=== s3_min global (grille fine) = {s3_global_min:.10f} ===")


if __name__ == "__main__":
    main()
