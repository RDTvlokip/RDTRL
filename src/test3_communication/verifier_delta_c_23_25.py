"""Piste 2 de ETAT.md : delta_c pour la paire 23/25 (egalite naturelle,
message 13, replay_23_25.py) est-il identique a celui de 3/4
(0,013437210, verifie 3 fois par des methodes independantes -- CARNET
§7.64/§7.65) ? La forme fermee de H7/H12 ne mentionne QUE N=27 et
beta=0,02, jamais l'identite des referents -- donc l'ALGEBRE predit deja
un delta_c identique par construction (meme systeme d'equations, memes
N/beta). Le test qui a de la valeur n'est pas algebrique : c'est de
savoir si la DYNAMIQUE REELLE d'entrainement sur CETTE collision
particuliere (23/25, graine 50000, message 13) reproduit ce seuil par
bissection empirique, exactement comme verifier_bissection_delta_c.py
l'a fait pour 3/4.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_23_25 import construire, BETA
from representable_atteignable_stable import N
from verifier_prior_asymetrique import continuer_sous_prior, etat

ADAM_EPS = 1e-10
PAS_SUITE = 40000
MESSAGE = 13
REFS = (23, 25)
DELTA_C_34 = 0.013437210  # verifie 3 fois independamment sur 3/4


def poids_pour(delta):
    p = torch.full((N,), 1.0 / N, dtype=torch.float64)
    p[25] = (1.0 + delta) / N
    p[23] = (1.0 - delta) / N
    return p


def converge_t_il(delta):
    """True si le referent 23 s'effondre (s23 pres de 1/27) plutot que
    de rester gradue (s23 pres de 1)."""
    poids = poids_pour(delta)
    e, r = construire()
    continuer_sous_prior(e, r, BETA, PAS_SUITE, 0.05, ADAM_EPS, poids)
    R, H, m, Hb, S = etat(e, r, msg=MESSAGE, refs=REFS)
    return S[23] < 0.5, R[25], S[23]


if __name__ == "__main__":
    torch.set_printoptions(precision=6)
    lo, hi = 0.013, 0.014
    print(f"=== bissection empirique de delta_c, referents 23/25, message 13 ===")
    print(f"  (comparaison : delta_c(3/4) = {DELTA_C_34})")
    sature_lo, _, _ = converge_t_il(lo)
    sature_hi, _, _ = converge_t_il(hi)
    assert not sature_lo, "lo devrait etre gradue (sous delta_c)"
    assert sature_hi, "hi devrait etre effondre (au-dessus de delta_c)"
    for _ in range(10):
        mid = (lo + hi) / 2
        sature, R25, s23 = converge_t_il(mid)
        print(f"  delta={mid:.7f}  effondre={sature}  R[13,25]={R25:.6f}  s[23,13]={s23:.6e}")
        if sature:
            hi = mid
        else:
            lo = mid
    delta_c_2325 = (lo + hi) / 2
    print(f"\n  delta_c(23/25) encadre entre {lo:.7f} et {hi:.7f}, centre = {delta_c_2325:.7f}")
    print(f"  delta_c(3/4)   = {DELTA_C_34:.7f}")
    print(f"  ecart = {delta_c_2325 - DELTA_C_34:+.7f}  ({abs(delta_c_2325-DELTA_C_34)/DELTA_C_34*100:.4f} %)")
