"""Suite du tour 56 (21/09/2026) : un agent de verification a trouve
que le coefficient de variation (CV) des 26 probabilites s[3,j!=10]
decroit de facon dynamique et progressive au fil de l'entrainement
(pas une degenerescence figee des l'initialisation) -- de CV~2 juste
apres la perturbation +30 jusqu'a ~0 (quasi-degenerescence exacte)
apres ~100000 pas cumules. Verifie ici sur UNE SEULE trace continue
depuis la perturbation elle-meme (construire_mur23 fait deja 40000
pas en interne avant de retourner e,r -- source du desaccord initial
entre ma premiere mesure et celle de l'agent, resolu en mesurant tout
depuis pas=0 post-perturbation au lieu de repartir de l'etat deja
construit).
"""

import sys
sys.path.insert(0, '.')
import torch
import statistics

from replay_mur23_referent3 import replay, objectif, BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere

DELTA = 0.013026615
LR = 0.05
ADAM_EPS = 1e-10
PAS_SUITE_INTERNE = 40000  # duree normale du pas_suite de construire_mur23
CHECKPOINTS = (0, 10, 100, 1000, 5000, 40000, 59989, 99989)


def main():
    e, r = replay(77777, 3, 10000)
    with torch.no_grad():
        e.p[0][4, 10] += 30.0

    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=LR, eps=ADAM_EPS)

    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + DELTA) / N
    poids[3] = (1.0 - DELTA) / N

    pas_max = max(CHECKPOINTS) + 1
    for pas in range(pas_max):
        if pas < PAS_SUITE_INTERNE:
            j, _ = objectif(e, r, BETA)
        else:
            j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        if pas in CHECKPOINTS:
            with torch.no_grad():
                s3_ligne = e.loi()[3, :].clone()
            autres = [s3_ligne[jj].item() for jj in range(27) if jj != 10]
            moy = statistics.mean(autres)
            et = statistics.pstdev(autres)
            cv = et / moy if moy else float('nan')
            print(f"pas depuis perturbation={pas}  moyenne_prob_autres={moy:.6e}  CV={cv:.6f}")


if __name__ == "__main__":
    main()
