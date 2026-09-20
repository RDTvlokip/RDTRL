"""Tour 53 (20/09/2026) : test de l'hypothese standard n.1 posee dans
REPONSE_ORDRE54.md pour expliquer pourquoi l'excursion sous Adam complet
vit presque entierement dans R et pas dans s3 -- "la sensibilite locale
du recepteur en R est plus grande que celle de l'emetteur en s3".

Mesure directe (differences finies) de la courbure locale de l'objectif
au point convergu (s3=0.998963, R=0.794756), independamment en espace
LOGIT (l'espace ou Adam applique ses pas) et en espace PROBABILITE
(s3, R) via la regle de la chaine (valide ici : gradient ~0 au plateau).

Resultat : les deux espaces donnent des verdicts OPPOSES.
  - En logit, l'emetteur est ~150x PLUS PLAT que le recepteur
    (d2J/dlogit_s3^2=-7.9e-7 contre d2J/dlogit_R4^2=-1.21e-4).
  - En probabilite, la compression de saturation du softmax
    (s3(1-s3)=1.04e-3 contre R(1-R)=0.163, facteur 157x) inverse tout :
    l'emetteur devient ~160x PLUS RAIDE (-0.734 contre -0.0045).
L'hypothese standard n.1 est confirmee dans sa conclusion (s3 stable,
R mobile) mais son mecanisme initial ("s3 est intrinsequement raide")
etait faux -- c'est la saturation du softmax qui amplifie une courbure
logit en fait plus molle, pas une raideur intrinseque de l'emetteur.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import activer, parametres
from verifier_prior_asymetrique import objectif_pondere
from verifier_sonde_bassin import poids_delta

DELTA = 0.013026615
PAS_WARMUP = 40000


def converger():
    poids = poids_delta(DELTA)
    e, r = construire_mur23(adam_eps=1e-10)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=0.05, eps=1e-10)
    for _ in range(PAS_WARMUP):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
    return e, r, poids


def courbure_logit(e, r, poids, indice, h):
    def J():
        j, _ = objectif_pondere(e, r, BETA, poids)
        return j.item()

    tenseur, i, k = indice
    base = tenseur[i, k].item()
    j0 = J()
    with torch.no_grad():
        tenseur[i, k] = base + h
    jp = J()
    with torch.no_grad():
        tenseur[i, k] = base - h
    jm = J()
    with torch.no_grad():
        tenseur[i, k] = base
    return (jp - 2 * j0 + jm) / h ** 2


def main():
    torch.set_printoptions(precision=6)
    e, r, poids = converger()
    with torch.no_grad():
        s3 = e.loi()[3, 10].item()
        R4 = (e.loi()[4, 10] * r.loi()[10, 4]).item()
    print(f"plateau : s3={s3:.6f}  R4={R4:.6f}")

    for h in (1e-2, 1e-3, 1e-4):
        ce = courbure_logit(e, r, poids, (e.p[0], 3, 10), h)
        cr = courbure_logit(e, r, poids, (r.p[0], 10, 4), h)
        print(f"h={h:.0e}  d2J/dlogit_s3^2={ce:.6e}  d2J/dlogit_R4^2={cr:.6e}  "
              f"ratio |emetteur/recepteur|={abs(ce/cr):.6f}")

    comp_s3 = s3 * (1 - s3)
    comp_R4 = R4 * (1 - R4)
    print(f"\ncompression ds3/dlogit = s3(1-s3) = {comp_s3:.6e}")
    print(f"compression dR/dlogit  = R4(1-R4) = {comp_R4:.6e}")
    print(f"ratio compression R/s3 = {comp_R4/comp_s3:.4f}")

    ce_ref = courbure_logit(e, r, poids, (e.p[0], 3, 10), 1e-4)
    cr_ref = courbure_logit(e, r, poids, (r.p[0], 10, 4), 1e-4)
    curv_prob_e = ce_ref / comp_s3 ** 2
    curv_prob_r = cr_ref / comp_R4 ** 2
    print(f"\nd2J/ds3^2  (approx, chaine) = {curv_prob_e:.6e}")
    print(f"d2J/dR4^2  (approx, chaine) = {curv_prob_r:.6e}")
    print(f"ratio probabilite |emetteur/recepteur| = {abs(curv_prob_e/curv_prob_r):.4f}")


if __name__ == "__main__":
    main()
