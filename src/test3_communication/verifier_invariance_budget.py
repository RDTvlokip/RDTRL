"""Tour 49 (premier point de dipankarsarkar) : mes quatre "refutations"
etaient toutes mesurees a delta=0,02, deja sature (R=1,0, H=0,0) --
cinq bras assis sur la meme frontiere ne peuvent rien separer. La seule
region ou une hypothese peut deplacer le nombre est la region graduee
(delta<=0,01 environ).

Son test : balayer delta a plusieurs budgets et voir si le bord de
bassin (delta_c) glisse avec le budget (H1 vivant, la loi molle n'est
qu'un transitoire) ou reste au meme endroit (H1 mort pour de vrai).

Grille resserree autour de la transition observee (0,01-0,02), a trois
budgets tres differents (15k comme le balayage original, 40k comme le
test decisif, 200k comme le test H1 initial).
"""

import sys
sys.path.insert(0, '.')
import math
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import N
from verifier_prior_asymetrique import continuer_sous_prior, etat

ADAM_EPS = 1e-10

if __name__ == "__main__":
    torch.set_printoptions(precision=15)
    deltas = (0.010, 0.012, 0.014, 0.016, 0.018, 0.020)
    budgets = (15000, 40000, 200000)
    print("=== bord de bassin (delta_c) a trois budgets, grille resserree ===")
    for pas in budgets:
        print(f"\n  --- budget = {pas} pas ---")
        for delta in deltas:
            poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
            poids[4] = (1.0 + delta) / N
            poids[3] = (1.0 - delta) / N
            e, r = construire_mur23(adam_eps=ADAM_EPS)
            continuer_sous_prior(e, r, BETA, pas, 0.05, ADAM_EPS, poids)
            R, H, m, Hb, S = etat(e, r)
            prediction = 1.0 / (1.0 + math.exp(-2 * delta / BETA))
            print(f"    delta={delta:<7}  R[10,4]={R[4]:.6f}  "
                  f"s[3,10]={S[3]:.6e}  prediction_molle={prediction:.6f}")
