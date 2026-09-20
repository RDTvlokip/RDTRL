"""Tour 53 (20/09/2026, repris apres 18h) : tentative de derivation de
la cassure de la loi `espacement~1/(1-beta2)` deja observee entre
beta2=0.995 et beta2=0.99 (voir CARNET.md fin de 7.65/8ter,
verifier_kicks_adam_grille_fine.py).

Hypothese fermee testee : le temps de declenchement t* d'un kick suit
t* ~ ln(v_pic/v_crit)/(1-beta2), ou v_pic est la valeur de v juste
apres regonflement (le pic post-kick) et v_crit le plancher juste
avant que l'oscillation ne demarre. Si ln(v_pic/v_crit) etait CONSTANT
en beta2, la loi naive tiendrait partout -- ce n'est pas le cas.

Resultat : ln(v_pic/v_crit) croit nettement quand beta2 diminue
(0.096 a 0.999, 0.471 a 0.995, 0.652 a 0.99), MAIS la prediction
resultante (t*~96,94,65) ne reproduit PAS les espacements medians
reellement mesures (~456-470, 110, 165). Mesure sur UN SEUL kick par
beta2 -- bruitee, pas une moyenne. PAS une derivation fermee complete,
juste un signal qualitatif dans le bon sens. Reste ouvert : moyenner
sur plusieurs kicks, ou resoudre l'equation de recurrence de v en
incluant le terme de gradient de fond (deja mesure a ~8% d'effet a
beta2=0.999 dans verifier_mecanisme_plancher_v.py, probablement plus
grand a beta2 plus petit).
"""

import sys
import math
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere

DELTA = 0.013026615
LR = 0.05
ADAM_EPS = 1e-10
BETA2_VALEURS = (0.999, 0.995, 0.99)


def mesurer_v_pic_crit(beta2, pas_max=15000, seuil=0.0015):
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + DELTA) / N
    poids[3] = (1.0 - DELTA) / N
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=LR, eps=ADAM_EPS, betas=(0.9, beta2))
    p_r = r.p[0]
    vs, Rs = [], []
    for pas in range(pas_max):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        with torch.no_grad():
            R4 = (e.loi()[4, 10] * r.loi()[10, 4]).item()
            st_r = opt.state.get(p_r, {})
            v_r = st_r["exp_avg_sq"][10, 4].item()
        vs.append(v_r)
        Rs.append(R4)
    baseline = sum(Rs[8000:12000]) / 4000
    devs = [abs(R - baseline) for R in Rs[5000:]]
    idx_crit_candidates = [i + 5000 for i in range(len(devs)) if devs[i] > seuil]
    if not idx_crit_candidates:
        return None
    first = idx_crit_candidates[0]
    v_crit = min(vs[max(0, first - 15):first])
    v_pic = max(vs[first:first + 30])
    return v_crit, v_pic, first


def main():
    for beta2 in BETA2_VALEURS:
        res = mesurer_v_pic_crit(beta2)
        if res:
            v_crit, v_pic, first = res
            ratio = math.log(v_pic / v_crit)
            print(f"beta2={beta2}  v_crit={v_crit:.5e}  v_pic={v_pic:.5e}  "
                  f"ln(v_pic/v_crit)={ratio:.4f}  first_kick_pas={first}")
        else:
            print(f"beta2={beta2}  aucun kick trouve dans la fenetre")


if __name__ == "__main__":
    main()
