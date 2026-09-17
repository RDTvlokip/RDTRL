"""Derniere piste de l'etape 2 (ETAT.md) : la derive de k(R) vient-elle
des 25 autres lignes du systeme complet (le modele reduit a 2 equations
en est une projection incomplete) ? Teste sur le jouet a 2 referents
(verifier_jouet_2_referents.py, tour 49/H13) qui n'a justement PAS ces
25 autres lignes -- s'il produit un k CONSTANT (pas de derive avec
R_init), les 25 lignes etaient bien la cause de la derive observee sur
le vrai systeme a 27 referents.
"""

import sys
sys.path.insert(0, '.')
import math
import torch

from verifier_jouet_2_referents import BETA, N as N_TOY

DELTA = 0.013


def construire_toy_r_init(delta, R_init, s3_init=0.999999999666, s4_init=0.999999999997):
    p3 = torch.tensor([math.log(s3_init / (1 - s3_init))], dtype=torch.float64, requires_grad=True)
    p4 = torch.tensor([math.log(s4_init / (1 - s4_init))], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([math.log(R_init / (1 - R_init))], dtype=torch.float64, requires_grad=True)
    return p3, p4, q


def objectif_toy(p3, p4, q, poids3, poids4):
    s3 = torch.sigmoid(p3)
    s4 = torch.sigmoid(p4)
    r4 = torch.sigmoid(q)
    r3 = 1 - r4
    recompense = poids3 * s3 * r3 + poids4 * s4 * r4
    ent = lambda p: -(p * torch.log(p.clamp_min(1e-300)) + (1 - p) * torch.log((1 - p).clamp_min(1e-300)))
    entropie = ent(s3) + ent(s4) + ent(r4)
    return (recompense + (BETA / N_TOY) * entropie).sum()


def entrainer_toy(delta, R_init, s3_init, pas=40000, lr=0.05, adam_eps=1e-10):
    poids3 = torch.tensor((1.0 - delta) / N_TOY, dtype=torch.float64)
    poids4 = torch.tensor((1.0 + delta) / N_TOY, dtype=torch.float64)
    p3, p4, q = construire_toy_r_init(delta, R_init, s3_init=s3_init)
    opt = torch.optim.Adam([p3, p4, q], lr=lr, eps=adam_eps)
    for _ in range(pas):
        j = objectif_toy(p3, p4, q, poids3, poids4)
        opt.zero_grad()
        (-j).backward()
        opt.step()
    with torch.no_grad():
        s3 = torch.sigmoid(p3).item()
        r4 = torch.sigmoid(q).item()
        R4 = torch.sigmoid(p4).item() * r4
    return R4, s3


def bissecter_toy(R_init, lo, hi, delta_c_approx, tol=1e-5, pas=40000):
    def grade(cible_s3):
        R4, s3f = entrainer_toy(delta_c_approx, R_init, cible_s3, pas=pas)
        return s3f > 0.5

    g_lo, g_hi = grade(lo), grade(hi)
    assert g_lo != g_hi, f"lo/hi meme issue a R_init={R_init}"
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if grade(mid) == g_lo:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


if __name__ == "__main__":
    torch.set_printoptions(precision=6)
    # delta_c du jouet, deja localise au tour 49 : (0.018688, 0.018711)
    DELTA_C_TOY = 0.0187

    print("=== jouet a 2 referents : point de bascule a trois R_init, delta=delta_c_toy ===")
    for R_init in (0.75, 0.60, 0.50):
        flip = bissecter_toy(R_init, 0.90, 0.999999, DELTA_C_TOY)
        print(f"  R_init={R_init}  flip_s3={flip:.6f}")
