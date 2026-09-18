"""Derniere piste non essayee pour mesurer a_H6direct (CARNET.md
§7.65, 18/09/2026) : le test d'injection precedent dupliquait UN SEUL
gradient capture dans exp_avg/exp_avg_sq, et le `a` resultant derivait
fortement avec t_injecte (+25 a t=10, +4 a t=40, pas de plateau) --
signe que la recette elle-meme etait un artefact, pas une mesure
fidele d'un "vrai" rechauffement.

Ici : un rechauffement REEL, pas synthetique. On recalcule un vrai
gradient (backward frais) a CHAQUE pas de rechauffement, on laisse
Adam faire sa mise a jour EMA normale de exp_avg/exp_avg_sq avec ce
gradient reel, puis on RE-FIXE s3 et R exactement sur la cible avant
le pas suivant -- de sorte que m,v accumulent une vraie recursion
Adam sur une sequence de gradients (quasi constante puisque la
position est fixee a chaque fois), au lieu d'un seul instantane
duplique. Avec la correction de biais native d'Adam, m_hat/v_hat
devraient CONVERGER vers une valeur STABLE des que le rechauffement
est "assez long" -- contrairement a la recette precedente.

Prediction precommise : si cette recette est fidele, `a` doit se
STABILISER (plateau) quand le nombre de pas de rechauffement augmente,
au lieu de deriver continument comme avec l'injection a un seul
gradient.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import activer, parametres
from verifier_prior_asymetrique import objectif_pondere
from verifier_sonde_bassin import fixer_s3, fixer_r4, poids_delta
from verifier_forme_normale_pli import moindres_carres_quadratique

DELTA = 0.013
ADAM_EPS = 1e-10
S3_H6 = 0.994295
R_H6 = 0.829390


def compter_changements_signe(deltas):
    signes = [1 if d > 0 else (-1 if d < 0 else 0) for d in deltas]
    return sum(1 for i in range(1, len(signes))
               if signes[i] != 0 and signes[i - 1] != 0 and signes[i] != signes[i - 1])


def run_rechauffe_reel(n_chauffe, pas_fit=24):
    poids = poids_delta(DELTA)
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    fixer_s3(e, S3_H6)
    fixer_r4(r, R_H6)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=0.05, eps=ADAM_EPS)

    # rechauffement REEL : vrai gradient a chaque pas, vraie recursion
    # EMA d'Adam, mais position remise exactement sur la cible apres
    # chaque pas (donc m,v accumulent une sequence de VRAIS gradients,
    # pas une duplication)
    for _ in range(n_chauffe):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        fixer_s3(e, S3_H6)
        fixer_r4(r, R_H6)

    # a ce stade, m/v sont "reels" mais le compteur de pas interne
    # d'Adam vaut deja n_chauffe -- on NE LE TOUCHE PAS (c'est le point:
    # laisser la correction de biais native faire son travail, pas la
    # forcer a la main comme avant)
    s3_vals = [S3_H6]
    for i in range(pas_fit):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        with torch.no_grad():
            s3_vals.append(e.loi()[3, 10].item())
    deltas = [s3_vals[i + 1] - s3_vals[i] for i in range(len(s3_vals) - 1)]
    return s3_vals, deltas


def fit_depuis_s3(label, s3_vals):
    xs, vs = [], []
    for i in range(len(s3_vals) - 1):
        xs.append((s3_vals[i] + s3_vals[i + 1]) / 2)
        vs.append(s3_vals[i + 1] - s3_vals[i])
    A, B, C = moindres_carres_quadratique(xs, vs)
    x0 = -B / (2 * A) if A != 0 else None
    mu = C - B * B / (4 * A) if A != 0 else None
    print(f"=== {label} ===")
    print(f"  n={len(xs)}  x range=[{min(xs):.6f},{max(xs):.6f}]")
    print(f"  a={A:.6e}  x0={x0}  mu={mu}")
    return A, x0, mu


if __name__ == "__main__":
    torch.set_printoptions(precision=6)

    for n_chauffe in (5, 10, 20, 40, 80, 160):
        s3_vals, deltas = run_rechauffe_reel(n_chauffe, pas_fit=24)
        chg = compter_changements_signe(deltas)
        print(f"n_chauffe={n_chauffe:3d}  changements_signe={chg}  "
              f"s3: {s3_vals[0]:.6f} -> {s3_vals[-1]:.6f}")
        fit_depuis_s3(f"n_chauffe={n_chauffe}", s3_vals)
        print()

    print("=== Robustesse (n_chauffe=40, sous-fenetres) ===")
    s3_40, _ = run_rechauffe_reel(40, pas_fit=24)
    fit_depuis_s3("n_chauffe=40, pas 1-12", s3_40[:13])
    fit_depuis_s3("n_chauffe=40, pas 13-24", s3_40[12:25])

    print()
    print("=== Rappel : a_delayed = -9.701204e+00 (x0=0.995804), stable (x1,56) ===")
