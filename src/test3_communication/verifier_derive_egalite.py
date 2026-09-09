"""Le referent 3/4 (message 10) reste-t-il un point fixe sur la DUREE,
ou n'est-ce qu'une etape transitoire lente qu'on n'a pas assez suivie ?

Le 09/09/2026, dipankarsarkar demande si le discriminateur a un seul
instantane distingue point fixe et etape transitoire ; il ne le fait
pas ici (1-s minuscule des deux cotes). J'ai alors invoque "la
verification longitudinale des tours precedents" SANS la refaire sur
CETTE reconstruction precise (graine 77777, k=3) -- alors que la meme
session vient de montrer que des reconstructions anterieures etaient
fausses (idx5). Ce script comble le trou : il continue l'entrainement
de la config actuelle sur un tres grand nombre de pas et regarde si
R[10,4] derive nettement loin de 0,5 (etape transitoire lente) ou reste
epingle (vrai point fixe).
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, monter_avec_eps_adam, BETA


if __name__ == "__main__":
    torch.set_printoptions(precision=15)
    ADAM_EPS = 1e-10
    print(f"=== derive de R[10,4] sur la duree, adam_eps={ADAM_EPS:.0e} ===")
    e, r = construire_mur23(adam_eps=ADAM_EPS, pas_suite=40000)
    with torch.no_grad():
        s_, r_ = e.loi(), r.loi()
        print(f"  pas_cumules=40000   R[10,4]={(s_[4,10]*r_[10,4]).item():.9f}")
    for tranche in range(9):
        monter_avec_eps_adam(e, r, BETA, 40000, 0.05, ADAM_EPS)
        with torch.no_grad():
            s_, r_ = e.loi(), r.loi()
            R = (s_[4, 10] * r_[10, 4]).item()
            print(f"  pas_cumules={40000*(tranche+2):<7}  R[10,4]={R:.9f}")
