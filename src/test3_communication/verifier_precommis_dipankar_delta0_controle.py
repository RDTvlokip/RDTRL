"""Tour 54 (20/09/2026, vraie critique de dipankarsarkar) : test 3 des
trois tests precommis par lui. delta=0 (symetrie complete, aucune
recompense asymetrique) sur la meme fenetre [59000,61000) -- si le
co-dip s3/R persiste, c'est un artefact d'optimiseur independant du
mecanisme de recompense asymetrique.

Resultat, plus fin que sa dichotomie initiale : le KICK de R survit a
delta=0 (un evenement reel a pas~59600, dR4=+3.72e-6, ~2 ordres au-dessus
du bruit de fond ~1e-8) -- confirme que le mecanisme plancher-de-v est
bien intrinseque a l'optimiseur, pas au mecanisme de recompense. MAIS
le co-timing avec s3 ne survit PAS -- s3 reste parfaitement plat
(max|ds3|=1.4e-10) alors que R4 kick au meme moment. Le kick n'a pas
besoin de delta ; sa TRANSMISSION vers s3 en a besoin. Mecanisme :
a delta!=0, un kick de r[10,4] force r[10,3] dans l'autre sens (meme
softmax), et c'est le poids ASYMETRIQUE de la recompense qui transforme
ce mouvement cote recepteur en une poussee differentielle sur le
gradient de s3. A delta=0, le meme kick recepteur se produit mais rien
en aval ne distingue le sens du mouvement de r[10,3] vs r[10,4], donc
s3 ne le voit jamais.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere

DELTA = 0.0  # controle : symetrie complete
LR = 0.05
ADAM_EPS = 1e-10
PAS_MAX = 61000
DEBUT = 59000


def main():
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + DELTA) / N
    poids[3] = (1.0 - DELTA) / N

    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=LR, eps=ADAM_EPS)

    p_e, p_r = e.p[0], r.p[0]
    vals = []
    for pas in range(PAS_MAX):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        if pas >= DEBUT:
            with torch.no_grad():
                s3 = torch.sigmoid(p_e[3, 10]).item()
                R4 = (torch.sigmoid(p_e[4, 10]) * torch.sigmoid(p_r[10, 4])).item()
            vals.append((pas, s3, R4))

    baseline_s3 = sum(v[1] for v in vals[:100]) / 100
    baseline_R4 = sum(v[2] for v in vals[:100]) / 100
    print(f"baseline s3={baseline_s3:.8f}  R4={baseline_R4:.8f}")
    print(f"max |ds3| = {max(abs(v[1]-baseline_s3) for v in vals):.3e}")
    print(f"max |dR4| = {max(abs(v[2]-baseline_R4) for v in vals):.3e}")
    for pas, s3, R4 in vals[::200]:
        print(f"  pas={pas}  ds3={s3-baseline_s3:+.3e}  dR4={R4-baseline_R4:+.3e}")


if __name__ == "__main__":
    main()
