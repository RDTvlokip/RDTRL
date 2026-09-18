"""Verification independante (agent-dipankar, tour du 18/09/2026) du
protocole "temps de residence" applique a la trajectoire naturelle
trouvee ce meme jour (bissection sur s3_init SEUL, R laisse naturel a
~0.5, verifier_sonde_bassin.fixer_s3, seuil net entre 0.972646 et
0.972661, milieu retenu 0.9726535).

But : reproduire au moins deux points de la table publiee (bande=1e-3,
pas=3000), PUIS rejouer avec une bande et un budget de pas differents
(bande=5e-4, pas=5000) pour verifier que la croissance quasi-log du
temps de residence n'est pas un artefact du choix de bande/budget.

Rapporte aussi le fit log vs loi de puissance sur les DEUX jeux, et un
controle numerique a part : fitter une loi de puissance sur des
donnees PUREMENT logarithmiques synthetiques (meme a, meme grille
d'offset) pour voir si un exposant petit et un R^2 ~0.96 sont deja ce
qu'on obtient en fittant une loi de puissance sur du log pur -- auquel
cas l'exposant -0.208 ne serait PAS un signal contre le mecanisme de
noeud-col, juste l'ombre attendue d'un fit mal specifie.
"""

import sys
sys.path.insert(0, '.')
import copy
import numpy as np
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import activer, parametres, N
from verifier_prior_asymetrique import objectif_pondere
from verifier_sonde_bassin import fixer_s3, poids_delta

DELTA = 0.013
ADAM_EPS = 1e-10
LR = 0.05
SEUIL = 0.9726535
CIBLE_BANDE = 0.994300

# Construit le checkpoint mur23 UNE SEULE FOIS (10000+40000 pas Adam) et le
# clone pour chaque experience -- fixer_s3 ecrase de toute facon le logit
# [3,10] du clone, donc partir du meme checkpoint de base ne biaise rien,
# et evite de repayer 50000 pas Adam a chaque offset teste (24 fois sinon).
print("Construction du checkpoint de base (une seule fois)...", flush=True)
_E_BASE, _R_BASE = construire_mur23(adam_eps=ADAM_EPS)
print("Checkpoint de base pret.", flush=True)


def trace_residence(cible_s3_init, pas_max, bande, cible_bande=CIBLE_BANDE, verbose_traj=False):
    e, r = copy.deepcopy(_E_BASE), copy.deepcopy(_R_BASE)
    fixer_s3(e, cible_s3_init)
    activer(e, r)
    poids = poids_delta(DELTA)
    opt = torch.optim.Adam(parametres(e, r), lr=LR, eps=ADAM_EPS)
    n_dans_bande = 0
    traj = []
    for t in range(pas_max):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        with torch.no_grad():
            s3 = e.loi()[3, 10].item()
        if abs(s3 - cible_bande) < bande:
            n_dans_bande += 1
        if verbose_traj:
            traj.append(s3)
    if verbose_traj:
        return n_dans_bande, traj
    return n_dans_bande


def fit_log(offsets, ys):
    x = np.log10(np.asarray(offsets, dtype=float))
    y = np.asarray(ys, dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return coef[0], coef[1], r2


def fit_puissance(offsets, ys):
    x = np.log(np.asarray(offsets, dtype=float))
    y = np.log(np.asarray(ys, dtype=float))
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return coef[0], coef[1], r2


if __name__ == "__main__":
    torch.set_printoptions(precision=6)

    offsets_full = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5]

    print("=== Table complete, PARAMETRES PUBLIES (bande=1e-3, pas=3000) ===", flush=True)
    print("    (offset=1e-3 et 1e-4 = les 2 points 'reproduction directe' demandes)", flush=True)
    sous_pub, sur_pub = [], []
    for offset in offsets_full:
        sous = SEUIL - offset
        sur = SEUIL + offset
        n_sous = trace_residence(sous, 3000, 1e-3)
        n_sur = trace_residence(sur, 3000, 1e-3)
        sous_pub.append(n_sous)
        sur_pub.append(n_sur)
        print(f"  offset={offset:.0e}  SOUS n={n_sous:4d}   SUR n={n_sur:4d}", flush=True)

    a_log, b_log, r2_log = fit_log(offsets_full, sous_pub)
    a_pow, b_pow, r2_pow = fit_puissance(offsets_full, sous_pub)
    print(f"\n  fit log (mes donnees, params publies)  : a={a_log:.3f} b={b_log:.3f} R2={r2_log:.6f}")
    print(f"  fit puissance (mes donnees, params publies): exposant={a_pow:.4f} R2={r2_pow:.6f}")

    print("\n=== Table complete, PARAMETRES DIFFERENTS (bande=5e-4, pas=5000) ===", flush=True)
    sous_alt, sur_alt = [], []
    for offset in offsets_full:
        sous = SEUIL - offset
        sur = SEUIL + offset
        n_sous = trace_residence(sous, 5000, 5e-4)
        n_sur = trace_residence(sur, 5000, 5e-4)
        sous_alt.append(n_sous)
        sur_alt.append(n_sur)
        print(f"  offset={offset:.0e}  SOUS n={n_sous:4d}   SUR n={n_sur:4d}", flush=True)

    a_log2, b_log2, r2_log2 = fit_log(offsets_full, sous_alt)
    a_pow2, b_pow2, r2_pow2 = fit_puissance(offsets_full, sous_alt)
    print(f"\n  fit log (bande/budget differents)  : a={a_log2:.3f} b={b_log2:.3f} R2={r2_log2:.6f}")
    print(f"  fit puissance (bande/budget differents): exposant={a_pow2:.4f} R2={r2_pow2:.6f}")

    print("\n=== Controle : une loi de puissance ajustee sur du LOG PUR simule ===")
    print("    (meme grille d'offset, meme pente 'a' que le fit log mesure ci-dessus,")
    print("    zero bruit -- pour voir si exposant~-0.2/R2~0.96 est deja l'ombre")
    print("    attendue d'un fit puissance sur une vraie loi log, pas un signal a part)")
    y_synth = [a_log * np.log10(o) + b_log for o in offsets_full]
    a_pow_synth, b_pow_synth, r2_pow_synth = fit_puissance(offsets_full, y_synth)
    print(f"  donnees log-pures synthetiques (a={a_log:.3f}, b={b_log:.3f}) : {[f'{v:.1f}' for v in y_synth]}")
    print(f"  fit puissance SUR CES DONNEES SYNTHETIQUES : exposant={a_pow_synth:.4f} R2={r2_pow_synth:.6f}")
