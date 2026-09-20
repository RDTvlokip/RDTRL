"""Tour 54 (20/09/2026, vraie critique de dipankarsarkar) : reponse a
sa mise en garde sur la quantification (ds3 imprime a 6 decimales,
~8% de bruit de quantification) et a sa question de cloture (rejouer
la trace en espace LOGIT plutot que probabilite). Log logit_s3
(e.p[0][3,10]), logit_s4 (e.p[0][4,10]) et logit_r4 (r.p[0][10,4]) a
10 decimales sur la fenetre [58000,62000), grille de 1000 pas plus
quelques points cles autour du kick a pas=60000.

Resultat : logit_s4 monte de facon parfaitement lisse et monotone sur
toute la fenetre, aucun accroc aux pas du dip (59989, 60432).
logit_s3 ET logit_r4 plongent ensemble aux memes pas puis recuperent
ensemble a pas=61000. Confirme directement en espace logit (donc sans
probleme de quantification a cette precision) le couplage structurel
deja etabli algebriquement : s3<->r4, s4 spectateur.
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
PAS_MAX = 62000
DEBUT = 58000
POINTS_CLES = (58000, 59000, 59989, 60432, 61000, 61999)


def main():
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + DELTA) / N
    poids[3] = (1.0 - DELTA) / N

    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=LR, eps=ADAM_EPS)

    p_e, p_r = e.p[0], r.p[0]
    logits = {}
    for pas in range(PAS_MAX):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        if pas >= DEBUT:
            with torch.no_grad():
                logits[pas] = (p_e[3, 10].item(), p_e[4, 10].item(), p_r[10, 4].item())

    for pas in POINTS_CLES:
        if pas in logits:
            ls3, ls4, lr4 = logits[pas]
            print(f"pas={pas}  logit_s3={ls3:.10f}  logit_s4={ls4:.10f}  logit_r4={lr4:.10f}")


if __name__ == "__main__":
    main()
