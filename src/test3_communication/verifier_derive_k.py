"""Tour 52 : trois hypotheses posees sur la derive de k (1,42 a 2,45
selon R_init) et jamais testees -- corrige ici.

H_standard : k(R) est une fonction lisse de l'etat seul, independante
  du chemin/budget. Teste : refaire le point de bascule a R_init=0,50
  avec un budget bien plus long (200 000 pas au lieu de 40 000) et
  verifier si k_fit change.

H_chemin : k depend du budget d'entrainement (l'estimation du second
  moment n'a pas fini de se stabiliser a 40 000 pas pour les points
  loin du point selle). Teste par le meme run que H_standard : si k_fit
  change avec le budget, H_chemin est soutenue et H_standard tombe.

H_momentum : la derive vient de beta1 (momentum), pas seulement de
  beta2. Teste : refaire les trois points de bascule (R_init=0,75/0,60/
  0,50) avec beta1=0 (Adam sans momentum, RMSprop de fait) et verifier
  si l'ecart k(0,75) vs k(0,50) se resserre.
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


def continuer(e, r, pas, lr, adam_eps, poids, betas=(0.9, 0.999)):
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=lr, eps=adam_eps, betas=betas)
    for _ in range(pas):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()


def issue(cible_s3, R_init, pas=40000, betas=(0.9, 0.999)):
    poids = poids_delta(DELTA)
    e, r = construire_mur23(adam_eps=1e-10)
    fixer_s3(e, cible_s3)
    fixer_r4(r, R_init)
    continuer(e, r, pas, 0.05, 1e-10, poids, betas=betas)
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


def fit_k(target, R_cible, delta=DELTA, beta=BETA, K_pref=26, lo=0.1, hi=5.0):
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
        if v < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


if __name__ == "__main__":
    torch.set_printoptions(precision=6)

    print("=== H_standard / H_chemin : R_init=0.50, budget 40k vs 200k ===")
    flip_40k = bissecter(0.50, 0.90, 0.999, pas=40000)
    k_40k = fit_k(flip_40k, 0.50)
    print(f"  budget=40000   flip={flip_40k:.6f}  k_fit={k_40k:.4f}")

    flip_200k = bissecter(0.50, 0.90, 0.999, pas=200000)
    k_200k = fit_k(flip_200k, 0.50)
    print(f"  budget=200000  flip={flip_200k:.6f}  k_fit={k_200k:.4f}")

    if abs(k_200k - k_40k) < 0.05:
        print("  -> k STABLE avec le budget : H_chemin refutee, H_standard tient sur cet axe.")
    else:
        print("  -> k CHANGE avec le budget : H_chemin soutenue, H_standard tombe.")

    print("\n=== H_momentum : beta1=0 (Adam sans momentum) aux trois points ===")
    resultats = {}
    for R_init, bornes in [(0.75, (0.95, 0.999)), (0.60, (0.90, 0.999)), (0.50, (0.85, 0.999))]:
        flip = bissecter(R_init, *bornes, pas=40000, betas=(0.0, 0.999))
        k = fit_k(flip, R_init)
        resultats[R_init] = (flip, k)
        print(f"  R_init={R_init}  beta1=0  flip={flip:.6f}  k_fit={k:.4f}")

    ecart_avec_momentum = 2.449 - 1.418
    ecart_sans_momentum = resultats[0.75][1] - resultats[0.50][1]
    print(f"\n  ecart k(0.75)-k(0.50) AVEC momentum (beta1=0.9, tour 52) : {ecart_avec_momentum:.3f}")
    print(f"  ecart k(0.75)-k(0.50) SANS momentum (beta1=0)             : {ecart_sans_momentum:.3f}")
    if abs(ecart_sans_momentum) < abs(ecart_avec_momentum) * 0.5:
        print("  -> l'ecart se resserre nettement sans momentum : H_momentum soutenue.")
    else:
        print("  -> l'ecart ne se resserre pas : H_momentum refutee, la derive vient d'ailleurs.")
