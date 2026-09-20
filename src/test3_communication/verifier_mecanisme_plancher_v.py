"""Tour 53 (20/09/2026) : verification complete, de bout en bout, du
mecanisme "oscillateur de relaxation par plancher numerique de v"
propose par un agent-dipankar pour expliquer les "kicks" de R sous
Adam complet (verifier_kicks_adam_grille_fine.py, ~0.0037 d'amplitude
quasi constante toutes les ~460-500 pas).

Log dense (grille 1) de v_r=exp_avg_sq et R4 sur une fenetre qui
couvre : la fin d'un kick precedent, toute la phase "calme" qui suit,
et le declenchement + regonflement du kick suivant (trouve ici a
pas=10185).

Resultat, mecanisme confirme dans l'ordre attendu :
  1. v_r decroit de facon lisse et monotone pendant ~380 pas
     (1.403e-13 -> 1.039e-13), R4 quasi immobile.
  2. L'oscillation de R4 demarre AVANT que v_r ne remonte -- le
     declenchement precede le regonflement, pas l'inverse.
  3. v_r regonfle rapidement (1.058e-13 -> 1.48e-13 en ~25 pas)
     pendant que l'oscillation de R4 s'amortit.

Verification quantitative annexe : la decroissance de v_r sur
[9800,10160] est ~8% plus lente qu'une pure decroissance geometrique
a beta2=0.999 (v0*beta2^n) -- coherent avec un gradient de fond non
strictement nul en phase calme, pas une anomalie.
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
PAS_MAX = 10220
DEBUT_LOG = 9700


def main():
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + DELTA) / N
    poids[3] = (1.0 - DELTA) / N

    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=LR, eps=ADAM_EPS)
    p_r = r.p[0]

    trace = []
    for pas in range(PAS_MAX):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        with torch.no_grad():
            R4 = (e.loi()[4, 10] * r.loi()[10, 4]).item()
            st_r = opt.state.get(p_r, {})
            v_r = st_r["exp_avg_sq"][10, 4].item()
        if pas >= DEBUT_LOG:
            trace.append((pas, R4, v_r))

    for pas, R4, v_r in trace:
        if pas < 10160:
            if pas % 20 == 0:
                print(f"pas={pas:<6} R4={R4:.6f}  v_r={v_r:.5e}")
        else:
            print(f"pas={pas:<6} R4={R4:.6f}  v_r={v_r:.5e}")

    v0 = next(v for p, R, v in trace if p == 9800)
    v1 = next(v for p, R, v in trace if p == 10160)
    pred = v0 * (0.999 ** (10160 - 9800))
    print(f"\ndecroissance geometrique pure predite = {pred:.5e}")
    print(f"observee = {v1:.5e}  (ratio observe/predit = {v1/pred:.4f})")


if __name__ == "__main__":
    main()
