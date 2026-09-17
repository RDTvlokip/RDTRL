"""Test direct sur le VRAI systeme a 27 referents (pas le jouet reduit) :
perturber la masse initiale des "25 autres referents" sur le message 10
(comme r_autres_init dans le jouet) et voir si le point de bascule
(protocole pin-and-falsify du tour 52, verifier_pin_k.py) se deplace,
donc si le k lu sur la table de dipankar en est affecte.

Le jouet a trouve un mecanisme reel (masse de fond ralentit le
recepteur pres du pli) mais n'a jamais reussi a le chiffrer proprement
contre k a cause des excursions Adam + du fantome deterministe du pli.
Plutot que de construire une ODE a 3 variables pour le jouet (k lui-meme
n'a jamais ete derivable analytiquement, meme au tour 52 -- seulement
mesure empiriquement), ce script applique le MEME protocole empirique
que le tour 52 DIRECTEMENT sur le systeme reel, avec la perturbation de
masse de fond ajoutee a la recette d'origine.

Reponse recherchee : positive OU negative, les deux comptent (Theo,
18/09/2026 -- ne pas abandonner une piste de mesure juste parce qu'elle
pourrait ne rien donner).
"""

import sys
sys.path.insert(0, '.')
import math
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import N
from verifier_prior_asymetrique import continuer_sous_prior, etat
from verifier_sonde_bassin import fixer_s3, fixer_r4, poids_delta

ADAM_EPS = 1e-10
DELTA = 0.013
PAS = 40000

# table de dipankar, tour 52 (flip@R=0.60 -> k)
TABLE_K = [
    (0.912058, 0.50),
    (0.953092, 0.75),
    (0.967244, 1.00),
    (0.978537, 1.50),
    (0.983293, 2.00),
]


def fixer_masse_fond(r, referents, cible_masse_totale, message=10):
    """Fixe la masse combinee de `referents` (une liste d'indices, PAS 3
    ni 4) a `cible_masse_totale`, repartie egalement entre eux, en
    laissant les logits de r3/r4 et des autres colonnes INCHANGES --
    seule la normalisation change (meme principe que fixer_r4)."""
    with torch.no_grad():
        ligne = r.p[0][message, :].clone()
        idx_autres = [i for i in range(N) if i not in referents]
        ligne_autres = ligne[idx_autres]
        S34_et_reste = torch.exp(ligne_autres - ligne_autres.max()).sum().item() * \
            math.exp(ligne_autres.max().item())
        K = len(referents)
        exp_z_new = (cible_masse_totale / (1 - cible_masse_totale) * S34_et_reste) / K
        z_new = math.log(exp_z_new)
        for i in referents:
            r.p[0][message, i] = z_new
    with torch.no_grad():
        rr = r.loi()[message, :]
        masse_verif = sum(rr[i].item() for i in referents)
    return masse_verif


def bissecter_flip(R_init, masse_fond, referents_fond, lo=0.90, hi=0.999, tol=1e-4):
    poids = poids_delta(DELTA)

    def issue_a(cible_s3):
        e, r = construire_mur23(adam_eps=ADAM_EPS)
        fixer_s3(e, cible_s3)
        fixer_r4(r, R_init)
        if masse_fond > 0:
            fixer_masse_fond(r, referents_fond, masse_fond)
        continuer_sous_prior(e, r, BETA, PAS, 0.05, ADAM_EPS, poids)
        R, H, m, Hb, S = etat(e, r)
        return S[3] > 0.5

    g_lo, g_hi = issue_a(lo), issue_a(hi)
    assert g_lo != g_hi, f"lo/hi meme issue a R_init={R_init}, masse_fond={masse_fond}"
    while hi - lo > tol:
        mid = (lo + hi) / 2
        g_mid = issue_a(mid)
        print(f"    s3={mid:.6f}  ->  {'GRADUEE (S3>0.5)' if g_mid else 'EFFONDRE'}", flush=True)
        if g_mid == g_lo:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def k_depuis_flip(flip):
    """Interpolation lineaire dans la table de dipankar (flip@R=0.60 -> k)."""
    pts = sorted(TABLE_K)
    if flip <= pts[0][0]:
        return pts[0][1]
    if flip >= pts[-1][0]:
        return pts[-1][1]
    for (f0, k0), (f1, k1) in zip(pts, pts[1:]):
        if f0 <= flip <= f1:
            return k0 + (k1 - k0) * (flip - f0) / (f1 - f0)
    return None


if __name__ == "__main__":
    torch.set_printoptions(precision=6)
    REFERENTS_FOND = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # 10 "autres" referents (ni 3 ni 4, ni 10 le message)

    print("=== BASELINE : flip point a R_init=0,60, sans masse de fond (reproduction du tour 52) ===")
    flip_base = bissecter_flip(0.60, 0.0, REFERENTS_FOND)
    k_base = k_depuis_flip(flip_base)
    print(f"  flip={flip_base:.6f}  k_lu_sur_table={k_base}")

    print("\n=== AVEC masse de fond = 25% (10 referents, comme le jouet M=25) ===")
    flip_25 = bissecter_flip(0.60, 0.25, REFERENTS_FOND)
    k_25 = k_depuis_flip(flip_25)
    print(f"  flip={flip_25:.6f}  k_lu_sur_table={k_25}")

    print("\n=== RESUME ===")
    print(f"  k(sans fond) = {k_base}")
    print(f"  k(avec fond 25%) = {k_25}")
    if k_base and k_25:
        print(f"  ratio = {k_25/k_base:.4f}  (a comparer au facteur k(52) : 1,42 a 2,45 selon R_init)")
