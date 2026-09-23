"""Tour 58 (23/09/2026) : test precommis par le premier agent style
dipankar, fait avec mon propre code.

Sa lecture du retard de la salve quand on gele la ligne 3 (+106 pas a
delta reel, verifie par verifier_tour58_audit_valeur_propre_et_timing.py) :
    Delta t = Delta S / (dS_gap/dt),  Delta S = pente S_gap kappa,
    d ln S_gap / dt = (1 - beta2)/2   (sqrt v du recepteur decroit en beta2^(t/2))
    => Delta t = 2 pente kappa / (1 - beta2) = 95,8 pas a beta2 = 0,999.
Si c'est la decroissance de v qui fixe le franchissement, passer le beta2
du SEUL recepteur a 0,998 doit diviser le retard par deux :
PREDICTION (ecrite avant le run, la sienne) : 2 x 0,937 x 0,05112 / 0,002
= 47,9 pas (~53 si son depassement de 10 % observe a beta2=0,999 est
reel). S'il reste pres de 106, le seuil n'est pas fixe par la
decroissance de v et sa lecture est fausse.

Protocole : etat a pas=59600 (delta reel) ; optimiseur scinde en deux
groupes (emetteur beta2=0,999 ; recepteur beta2 = 0,999 ou 0,998), etat
d'Adam recopie a l'identique ; ligne 3 gelee (k=0) ou non (k=1). Retard
= pic(k=0) - pic(k=1) a beta2 fixe.
"""

import sys
sys.path.insert(0, '.')
import copy
import torch

from replay_mur23_referent3 import BETA
from verifier_prior_asymetrique import objectif_pondere
from sauver_checkpoints_mur23_tour58 import reprendre, LR, ADAM_EPS

P_SWITCH = 59600


def run(beta2_r, k, duree=900):
    e, r, opt, poids = reprendre(0.013026615)
    for _ in range(P_SWITCH - 54000):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
    p_e, p_r = e.p[0], r.p[0]
    opt2 = torch.optim.Adam([{"params": [p_e], "betas": (0.9, 0.999)},
                             {"params": [p_r], "betas": (0.9, beta2_r)}], lr=LR, eps=ADAM_EPS)
    for p in (p_e, p_r):
        opt2.state[p] = copy.deepcopy(opt.state[p])
    gaps = []
    for t in range(duree):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt2.zero_grad()
        (-j).backward()
        opt2.step()
        with torch.no_grad():
            if k != 1:
                st = opt2.state[p_e]
                p_e[3] += (1 - k) * LR * st["exp_avg"][3] / (st["exp_avg_sq"][3].sqrt() + ADAM_EPS)
            gaps.append((p_r[10, 4] - p_r[10, 3]).item())
    base = sorted(gaps)[len(gaps) // 2]
    i0 = next(i for i in range(len(gaps)) if abs(gaps[i] - base) > 0.004)
    i = max(range(i0, min(i0 + 50, len(gaps))), key=lambda i: abs(gaps[i] - base))
    return P_SWITCH + i


if __name__ == "__main__":
    torch.set_num_threads(1)
    beta2_r = float(sys.argv[1])
    p1 = run(beta2_r, 1)
    p0 = run(beta2_r, 0)
    pred = 2 * 0.937 * 0.05112 / (1 - beta2_r)
    print(f"beta2 recepteur = {beta2_r} : pic k=1 a {p1}, pic ligne 3 gelee a {p0}, retard = {p0 - p1}   "
          f"predit 2 pente kappa/(1-beta2) = {pred:.1f}")
