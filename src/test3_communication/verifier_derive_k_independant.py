"""Verification independante (20/09/2026) du test croise beta2 propose
par dipankarsarkar : k(0.75)/k(0.50) sous beta2=0.99 reste-t-il proche
de la reference 1.7271 (beta1=0.9, beta2=0.999) ?

Reimplementation propre (pas un import de verifier_derive_k.py) de la
bissection et du fit de k, pour comparer independamment au resultat
deja publie ce jour (flip(0.75)=0.989714, k=2.4357 ; flip(0.50)=0.972195,
k=1.3971, ratio=1.7434, ecart 0.94% vs 1.7271).
"""

import sys
sys.path.insert(0, '.')
import math
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import activer, parametres
from verifier_prior_asymetrique import objectif_pondere, etat
from verifier_sonde_bassin import fixer_s3, fixer_r4, poids_delta

DELTA = 0.013


def train_run(cible_s3, R_init, pas, betas):
    e, r = construire_mur23(adam_eps=1e-10)
    fixer_s3(e, cible_s3)
    fixer_r4(r, R_init)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=0.05, eps=1e-10, betas=betas)
    poids = poids_delta(DELTA)
    for _ in range(pas):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
    _, _, _, _, S = etat(e, r)
    return S[3]


def find_flip(R_init, lo, hi, betas, pas=40000, tol=2e-5):
    """Bissection independante : retourne la valeur de cible_s3 separant
    la branche graduee (S3_final>0.5) de l'effondrement."""
    s3_lo = train_run(lo, R_init, pas, betas)
    s3_hi = train_run(hi, R_init, pas, betas)
    grade_lo = s3_lo > 0.5
    grade_hi = s3_hi > 0.5
    assert grade_lo != grade_hi, f"bornes ne separent pas: lo->{s3_lo}, hi->{s3_hi}"
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        s3_mid = train_run(mid, R_init, pas, betas)
        if (s3_mid > 0.5) == grade_lo:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fit_k_independant(flip_s3, R_cible, delta=DELTA, beta=BETA, K_pref=26,
                       lo=0.05, hi=8.0, iters=50):
    """Reimplementation independante de l'ODE a deux variables (x=1-s3, R)
    pres du col, et du fit par bissection sur k pour reproduire flip_s3."""

    def R_from_x(x):
        d3 = x
        logitR = (2 * delta + (1 - delta) * d3) / beta
        return 1.0 / (1.0 + math.exp(-logitR))

    def x_from_R(R):
        A = math.exp(-(1 - delta) * (1 - R) / beta)
        return K_pref * A / (1 + K_pref * A)

    d3_saddle = 5.70024e-3
    R_saddle = R_from_x(d3_saddle)

    def separatrice_1_moins_x(k):
        eps = 1e-7
        x = d3_saddle + eps
        R = R_saddle + eps * 0.1
        dt = -1e-5
        for _ in range(3_000_000):
            dx = x_from_R(R) - x
            dR = k * (R_from_x(x) - R)
            x += dx * dt
            R += dR * dt
            if R <= R_cible:
                return 1.0 - x
            if x < 0.0 or x > 1.0:
                return None
        return None

    lo_k, hi_k = lo, hi
    v_lo = separatrice_1_moins_x(lo_k)
    v_hi = separatrice_1_moins_x(hi_k)
    for _ in range(iters):
        mid = 0.5 * (lo_k + hi_k)
        v = separatrice_1_moins_x(mid)
        if v is None:
            hi_k = mid
            continue
        if v < flip_s3:
            lo_k = mid
        else:
            hi_k = mid
    return 0.5 * (lo_k + hi_k)


if __name__ == "__main__":
    torch.set_printoptions(precision=6)
    REF_RATIO = 1.7271

    print("=== reimplementation independante, beta2=0.99 ===")
    resultats = {}
    for R_init, bornes in [(0.75, (0.95, 0.999)), (0.50, (0.85, 0.999))]:
        flip = find_flip(R_init, *bornes, betas=(0.9, 0.99), pas=40000)
        k = fit_k_independant(flip, R_init)
        resultats[R_init] = (flip, k)
        print(f"  R_init={R_init}  flip={flip:.6f}  k_fit={k:.4f}")

    ratio = resultats[0.75][1] / resultats[0.50][1]
    ecart_pct = 100.0 * abs(ratio - REF_RATIO) / REF_RATIO
    print(f"\n  ratio k(0.75)/k(0.50) = {ratio:.4f}")
    print(f"  reference (beta1=0.9,beta2=0.999) = {REF_RATIO}")
    print(f"  ecart relatif = {ecart_pct:.3f}%")
