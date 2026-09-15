"""Tour 52 de dipankarsarkar : sa formule fermee ne dessine pas la
separatrice (elle ne connait que les points fixes, pas la vitesse
relative des deux joueurs). Protocole "epingler k avec UN essai, puis
falsifier avec deux autres" :

  1. Bissecter le point de bascule (s3) a R_init=0,60, delta=0,013.
  2. Lire k dans son tableau (interpolation).
  3. Predire les points de bascule a R_init=0,75 et R_init=0,50 avec ce k.
  4. Bissecter ces deux points reellement et comparer.

Table de reference (dipankar, tour 52) :
  flip@R=0.60    k       flip@R=0.75   flip@R=0.50
  0.912058      0.50      0.968331      0.862384
  0.953092      0.75      0.978188      0.936018
  0.967244      1.00      0.982594      0.958083
  0.978537      1.50      0.986715      0.974283
  0.983293      2.00      0.988682      0.980699
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import N
from verifier_prior_asymetrique import continuer_sous_prior, etat
from verifier_sonde_bassin import fixer_s3, fixer_r4, poids_delta

ADAM_EPS = 1e-10
DELTA = 0.013
PAS = 40000


def issue_a(cible_s3, R_init):
    poids = poids_delta(DELTA)
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    fixer_s3(e, cible_s3)
    fixer_r4(r, R_init)
    continuer_sous_prior(e, r, BETA, PAS, 0.05, ADAM_EPS, poids)
    R, H, m, Hb, S = etat(e, r)
    return S[3] > 0.5, S[3]


def bissecter(R_init, lo, hi, tol=1e-5):
    """lo = graduee, hi = effondre (ou l'inverse -- on le determine)."""
    grade_a_lo, _ = issue_a(lo, R_init)
    grade_a_hi, _ = issue_a(hi, R_init)
    assert grade_a_lo != grade_a_hi, f"lo et hi donnent la meme issue a R_init={R_init}"
    while hi - lo > tol:
        mid = (lo + hi) / 2
        grade_mid, s3f = issue_a(mid, R_init)
        print(f"    s3={mid:.6f}  ->  {'GRADUEE' if grade_mid else 'EFFONDRE'}  (s3_final={s3f:.6f})")
        if grade_mid == grade_a_lo:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


if __name__ == "__main__":
    torch.set_printoptions(precision=6)
    print(f"=== ETAPE 1 : bissecter le point de bascule a R_init=0,60, delta={DELTA} ===")
    flip_060 = bissecter(0.60, 0.90, 0.999)
    print(f"  point de bascule mesure a R_init=0,60 : s3={flip_060:.6f}")

    print(f"\n=== ETAPE 2 (a faire a la main) : lire k dans le tableau de dipankar ===")
    print(f"  comparer {flip_060:.6f} a sa colonne flip@R=0.60")

    print(f"\n=== ETAPE 3 : bissecter reellement a R_init=0,75 et R_init=0,50 pour comparer ===")
    print("  --- R_init=0.75 ---")
    flip_075 = bissecter(0.75, 0.90, 0.999)
    print(f"  point de bascule mesure a R_init=0,75 : s3={flip_075:.6f}")

    print("  --- R_init=0.50 ---")
    flip_050 = bissecter(0.50, 0.80, 0.999)
    print(f"  point de bascule mesure a R_init=0,50 : s3={flip_050:.6f}")

    print(f"\n=== RESUME ===")
    print(f"  flip@0.75={flip_075:.6f}  flip@0.60={flip_060:.6f}  flip@0.50={flip_050:.6f}")
