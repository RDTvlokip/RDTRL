"""Tour 49 (deuxieme point de dipankarsarkar) : l'entropie de l'emetteur
etait une moyenne NON ponderee alors que la recompense l'est. Sa lecture :
le referent 3 perd du signal de recompense (poids reduit) mais rien ne
reduit la pression d'entropie qui devrait le retenir engage -- un ratio
recompense/entropie desequilibre, pas forcement un vrai mecanisme
d'eviction. Son correctif d'une ligne : ponderer entropie_s par
(N*poids[i]), de sorte que chaque referent factorise
poids[i]*(recompense_i + beta*H_i) et que son propre optimum ne depende
plus de delta -- seul le posterior du recepteur doit alors bouger.

Prediction s'il a raison : la loi molle sigmoid(2*delta/beta) doit
survivre bien au-dela de delta=0,02, potentiellement jusqu'au point 2:1.
Prediction si H4 (bifurcation reelle) tient quand meme : le bord de
bassin reste au meme endroit malgre le correctif.
"""

import sys
sys.path.insert(0, '.')
import math
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import N
from verifier_prior_asymetrique import continuer_sous_prior, etat

ADAM_EPS = 1e-10
PAS_SUITE_ADD = 40000

if __name__ == "__main__":
    torch.set_printoptions(precision=15)
    print("=== balayage delta, entropie de l'emetteur REPONDEREE par le prior (correctif de dipankar) ===")
    print("  poids[i] identiques a avant ; seule entropie_s change : "
          "-(N*poids[i]) * s*log(s), pas -1*s*log(s)")
    for delta in (0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.3, 0.5, 1.0):
        poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
        poids[4] = (1.0 + delta) / N
        poids[3] = (1.0 - delta) / N
        e, r = construire_mur23(adam_eps=ADAM_EPS)
        continuer_sous_prior(e, r, BETA, PAS_SUITE_ADD, 0.05, ADAM_EPS, poids,
                              reponderer_entropie_s=True)
        R, H, m, Hb, S = etat(e, r)
        prediction = 1.0 / (1.0 + math.exp(-2 * delta / BETA)) if delta > 0 else 0.5
        print(f"  delta={delta:<6}  R[10,4]={R[4]:.6f}  s[3,10]={S[3]:.9f}  "
              f"s[4,10]={S[4]:.9f}  H(27)={H:.6f}  prediction_ancienne={prediction:.6f}")
    print(f"\n  point 2:1 litteral correspond a delta=1.0 dans cette famille a poids combine fixe")
    print(f"  (poids[4]=2/N, poids[3]=0/N) -- si la loi molle y arrive encore, la lecture")
    print(f"  'posterior bayesien' tient jusqu'au bout une fois l'entropie correctement mise a l'echelle.")
