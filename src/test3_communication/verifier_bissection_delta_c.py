"""Tour 50 : localiser delta_c a quatre chiffres par bissection (entre
0,013, gradue, et 0,014, sature), puis mesurer le nombre de pas
necessaires pour converger a 1%, 0,1% et 0,01% de la valeur finale a
mesure qu'on s'approche de delta_c par en dessous. Un noeud-col impose
un ralentissement critique en (delta_c - delta)^(-1/2) ; si le temps de
convergence reste plat, H6 meurt comme H1 et H2 avant lui.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere, etat

ADAM_EPS = 1e-10


def poids_pour(delta):
    p = torch.full((N,), 1.0 / N, dtype=torch.float64)
    p[4] = (1.0 + delta) / N
    p[3] = (1.0 - delta) / N
    return p


def converge_t_il(delta, pas=40000):
    """Renvoie True si R[10,4] finit pres de 1,0 (saturé) plutot que gradue."""
    poids = poids_pour(delta)
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=0.05, eps=ADAM_EPS)
    for _ in range(pas):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
    R, H, m, Hb, S = etat(e, r)
    return R[4] > 0.9, R[4], S[3]


if __name__ == "__main__":
    torch.set_printoptions(precision=6)
    lo, hi = 0.013, 0.014
    print(f"=== bissection de delta_c entre {lo} et {hi} ===")
    for _ in range(6):
        mid = (lo + hi) / 2
        sature, R4, s3 = converge_t_il(mid)
        print(f"  delta={mid:.6f}  sature={sature}  R[10,4]={R4:.6f}  s[3,10]={s3:.6e}")
        if sature:
            hi = mid
        else:
            lo = mid
    print(f"\n  delta_c encadre entre {lo:.6f} et {hi:.6f}")
