"""Agent (role dipankarsarkar, tour 59 simule) : le canal de TIMING (la ligne 3
avance la salve du recepteur) s'allume-t-il au meme delta que la pente ?
Predictions dans D:/tmp/agent_dk59_predictions_balayage_decalage.txt, ecrites
avant ce run.

Meme trajectoire que verifier_reponse_dipankar_tour58_balayage_delta_couplage.py
(construire_mur23 puis Adam lr=0,05 eps=1e-10 sous les poids de delta). A partir
de 58000 pas, on attend une salve (|dgap| > 0,004), on laisse passer 100 pas,
on sauve l'etat, puis on rejoue 700 pas avec k=1 (base) et k=0 (ligne 3 de
l'emetteur gelee : correction exacte apres opt.step()). Retard = debut de la
salve suivante (k=0) - debut (k=1).

Usage : python agent_dk59_balayage_decalage.py <delta>
"""
import sys
sys.path.insert(0, '.')
import copy
import math
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere

ADAM_EPS, LR = 1e-10, 0.05
AUTRES = [j for j in range(N) if j != 10]


def pas_adam(e, r, opt, poids):
    j, _ = objectif_pondere(e, r, BETA, poids)
    opt.zero_grad()
    (-j).backward()
    opt.step()


def main(delta):
    torch.set_num_threads(1)
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + delta) / N
    poids[3] = (1.0 - delta) / N
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    params = parametres(e, r)
    opt = torch.optim.Adam(params, lr=LR, eps=ADAM_EPS)
    p_e, p_r = e.p[0], r.p[0]
    pas = 0
    for _ in range(58000):
        pas_adam(e, r, opt, poids)
        pas += 1
    gap = (p_r[10, 4] - p_r[10, 3]).item()
    while True:
        pas_adam(e, r, opt, poids)
        pas += 1
        g2 = (p_r[10, 4] - p_r[10, 3]).item()
        if abs(g2 - gap) > 0.004:
            break
        gap = g2
    for _ in range(100):
        pas_adam(e, r, opt, poids)
        pas += 1
    bascule = pas
    with torch.no_grad():
        s = torch.softmax(p_e[3], 0)
        un = torch.exp(torch.logsumexp(p_e[3, AUTRES], 0) - torch.logsumexp(p_e[3], 0)).item()
        rr = torch.softmax(p_r[10], 0)
        X = (p_e[3, 10] - p_e[3, AUTRES].mean()).item()
    kappa = (1 - delta) * un * (1 - un) / BETA
    print(f"=== delta={delta} : bascule apres {bascule} pas ; X={X:.5f} 1-s3={un:.4e} "
          f"r3={rr[3].item():.5f} r4={rr[4].item():.5f} kappa=(1-d)s3(1-s3)/beta={kappa:.4e} "
          f"u=sqrt(v e3[10])={opt.state[p_e]['exp_avg_sq'][3, 10].sqrt().item():.4e}")
    sauve_p = [p.detach().clone() for p in params]
    sauve_opt = copy.deepcopy(opt.state_dict())
    res = {}
    for k in (1.0, 0.0):
        with torch.no_grad():
            for p, sv in zip(params, sauve_p):
                p.copy_(sv)
        opt.load_state_dict(copy.deepcopy(sauve_opt))
        gaps, Xs = [(p_r[10, 4] - p_r[10, 3]).item()], []
        debut, pic, amp = None, None, 0.0
        for i in range(700):
            pas_adam(e, r, opt, poids)
            if k != 1.0:
                with torch.no_grad():
                    st = opt.state[p_e]
                    m, rv = st["exp_avg"][3], st["exp_avg_sq"][3].sqrt()
                    p_e[3] -= (k - 1.0) * LR * m / (rv + ADAM_EPS)
            g = (p_r[10, 4] - p_r[10, 3]).item()
            dg = g - gaps[-1]
            gaps.append(g)
            etiquette = bascule + i   # meme convention que la trace du tour 58
            if debut is None and abs(dg) > 0.004:
                debut = etiquette
            if debut is not None and etiquette <= debut + 40 and abs(dg) > amp:
                amp, pic = abs(dg), etiquette
        res[k] = (debut, pic, amp)
        print(f"  k={k}: debut salve={debut}  pic={pic}  max|dgap|={amp:.4e}")
    if res[1.0][0] is not None and res[0.0][0] is not None:
        print(f"  RETARD (k=0 - k=1) : debut {res[0.0][0] - res[1.0][0]:+d} pas, pic {res[0.0][1] - res[1.0][1]:+d} pas")


if __name__ == "__main__":
    main(float(sys.argv[1]))
