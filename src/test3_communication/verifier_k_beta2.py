"""Etape 2 de ETAT.md : k(R) reste sans mecanisme ferme apres avoir
refute H_chemin (budget) et H_momentum (beta1). La piste non testee :
beta2 (memoire du second moment). Si le k effectif qu'une trajectoire
Adam traverse depend de combien de pas la moyenne mobile de exp_avg_sq
a eu le temps d'integrer, alors changer beta2 (a R_init fixe) devrait
deplacer k -- et par le meme raisonnement que H_momentum, si la DERIVE
de k avec R_init vient de beta2, changer beta2 devrait aussi changer
l'ecart k(0.75)-k(0.50).

Teste ici : k a R_init=0.60 fixe, pour trois valeurs de beta2, PUIS
(si le temps le permet) le spread complet a un beta2 alternatif.
"""

import sys
sys.path.insert(0, '.')
import math
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere, etat
from verifier_sonde_bassin import fixer_s3, fixer_r4, poids_delta

DELTA = 0.013


def issue(cible_s3, R_init, pas=40000, betas=(0.9, 0.999)):
    poids = poids_delta(DELTA)
    e, r = construire_mur23(adam_eps=1e-10)
    fixer_s3(e, cible_s3)
    fixer_r4(r, R_init)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=0.05, eps=1e-10, betas=betas)
    for _ in range(pas):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
    R, H, m, Hb, S = etat(e, r)
    return S[3] > 0.5, S[3]


def bissecter(R_init, lo, hi, tol=1e-5, pas=40000, betas=(0.9, 0.999)):
    grade_lo, _ = issue(lo, R_init, pas, betas)
    grade_hi, _ = issue(hi, R_init, pas, betas)
    assert grade_lo != grade_hi
    while hi - lo > tol:
        mid = (lo + hi) / 2
        g, _ = issue(mid, R_init, pas, betas)
        if g == grade_lo:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def fit_k(target, R_cible, delta=DELTA, beta=BETA, K_pref=26, lo=0.1, hi=6.0):
    def R_br(d3):
        logitR = (2 * delta + (1 - delta) * d3) / beta
        return 1 / (1 + math.exp(-logitR))

    def x_br(R):
        A = math.exp(-(1 - delta) * (1 - R) / beta)
        return K_pref * A / (1 + K_pref * A)

    d3_saddle = 5.70024e-3
    R_saddle = R_br(d3_saddle)

    def sep(k, R_cible, dt=-1e-5, pas_max=3_000_000):
        eps = 1e-7
        x = d3_saddle + eps
        R = R_saddle + eps * 0.1
        for _ in range(pas_max):
            dx = x_br(R) - x
            dR = k * (R_br(x) - R)
            x += dx * dt
            R += dR * dt
            if R <= R_cible:
                return 1 - x
            if x < 0 or x > 1:
                return None
        return None

    for _ in range(40):
        mid = (lo + hi) / 2
        v = sep(mid, R_cible)
        if v is None or v < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


if __name__ == "__main__":
    torch.set_printoptions(precision=6)
    print("=== k a R_init=0.60, trois valeurs de beta2 ===")
    for beta2 in (0.99, 0.999, 0.9999):
        flip = bissecter(0.60, 0.85, 0.999, betas=(0.9, beta2))
        k = fit_k(flip, 0.60)
        print(f"  beta2={beta2}  flip={flip:.6f}  k_fit={k:.4f}")
