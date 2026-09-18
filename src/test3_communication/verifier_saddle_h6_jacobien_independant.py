"""Verification INDEPENDANTE du calcul de jacobienne/valeurs propres avance
par l'agent-dipankar du 18/09/2026 (CARNET.md, §7.65, fin) : au point H6
(s3=0,994295, R=0,829390), le champ de vitesse CONTINU (avant Adam) aurait
deux valeurs propres reelles (+3,226e-6 et -1,128e-4, separation x35) --
un vrai col hyperbolique -- et le chaos observe sous Adam serait un
artefact de demarrage a froid (m=0,v=0), pas une propriete de l'objectif.

Methode INDEPENDANTE de celle de l'agent (qui travaillait en espace logit
z avec une approximation analytique ds/dz=s(1-s)) : ici, differences
finies DIRECTEMENT sur (s3,R) en espace probabilite, via un pas SGD
brut (pas Adam) uniforme sur TOUS les parametres (pas seulement les 2
logits vises), pour capturer aussi le retour de la normalisation
softmax sur les 25 autres referents -- plus fidele a la dynamique
reelle que la reduction analytique a 2 variables.
"""

import sys
sys.path.insert(0, '.')
import math
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import activer, parametres
from verifier_prior_asymetrique import objectif_pondere
from verifier_sonde_bassin import fixer_s3, fixer_r4, poids_delta

DELTA = 0.013
ADAM_EPS = 1e-10
S3_H6 = 0.994295
R_H6 = 0.829390


def taux(s3, R, lr_probe):
    """Construit le systeme frais a (s3,R), fait UN pas de SGD brut
    (uniforme sur TOUS les parametres, pas seulement les 2 logits vises)
    avec un lr minuscule, renvoie (ds3/dtau, dR/dtau) estime par
    difference finie / lr_probe."""
    poids = poids_delta(DELTA)
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    fixer_s3(e, s3)
    fixer_r4(r, R)
    activer(e, r)
    j, _ = objectif_pondere(e, r, BETA, poids)
    for p in parametres(e, r):
        if p.grad is not None:
            p.grad = None
    j.backward()
    with torch.no_grad():
        for p in parametres(e, r):
            p += lr_probe * p.grad
        s3_apres = e.loi()[3, 10].item()
        r_apres = r.loi()[10, 4].item()
    return (s3_apres - s3) / lr_probe, (r_apres - R) / lr_probe


def jacobien(s3_0, R_0, h, lr_probe):
    v_s3_splus, v_R_splus = taux(s3_0 + h, R_0, lr_probe)
    v_s3_smoins, v_R_smoins = taux(s3_0 - h, R_0, lr_probe)
    v_s3_Rplus, v_R_Rplus = taux(s3_0, R_0 + h, lr_probe)
    v_s3_Rmoins, v_R_Rmoins = taux(s3_0, R_0 - h, lr_probe)
    J00 = (v_s3_splus - v_s3_smoins) / (2 * h)   # d(v_s3)/d(s3)
    J01 = (v_s3_Rplus - v_s3_Rmoins) / (2 * h)   # d(v_s3)/d(R)
    J10 = (v_R_splus - v_R_smoins) / (2 * h)     # d(v_R)/d(s3)
    J11 = (v_R_Rplus - v_R_Rmoins) / (2 * h)     # d(v_R)/d(R)
    return J00, J01, J10, J11


def valeurs_propres_2x2(a, b, c, d):
    tr = a + d
    det = a * d - b * c
    disc = tr * tr - 4 * det
    if disc >= 0:
        sq = math.sqrt(disc)
        return ("reel", (tr + sq) / 2, (tr - sq) / 2)
    else:
        sq = math.sqrt(-disc)
        return ("complexe", complex(tr / 2, sq / 2), complex(tr / 2, -sq / 2))


if __name__ == "__main__":
    torch.set_printoptions(precision=6)

    print("=== Verification independante : jacobienne au point H6 ===")
    print(f"    point : s3={S3_H6}, R={R_H6}")
    for lr_probe in (1e-6, 1e-7, 1e-8):
        for h in (1e-3, 1e-4):
            J00, J01, J10, J11 = jacobien(S3_H6, R_H6, h, lr_probe)
            genre, l1, l2 = valeurs_propres_2x2(J00, J01, J10, J11)
            print(f"  lr_probe={lr_probe:.0e}  h={h:.0e}  "
                  f"J=[[{J00:.4e},{J01:.4e}],[{J10:.4e},{J11:.4e}]]  "
                  f"valeurs propres ({genre}): {l1}, {l2}")

    print()
    print("=== Meme calcul au point MIROIR (0,994305, gradue) ===")
    for lr_probe in (1e-7,):
        for h in (1e-4,):
            J00, J01, J10, J11 = jacobien(0.994305, R_H6, h, lr_probe)
            genre, l1, l2 = valeurs_propres_2x2(J00, J01, J10, J11)
            print(f"  lr_probe={lr_probe:.0e}  h={h:.0e}  "
                  f"J=[[{J00:.4e},{J01:.4e}],[{J10:.4e},{J11:.4e}]]  "
                  f"valeurs propres ({genre}): {l1}, {l2}")
