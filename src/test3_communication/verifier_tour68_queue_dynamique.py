"""Tour 68 : la queue du reseau complet est-elle gelee ? Depuis le checkpoint de phase 1 (parametres + etat Adam, 20 000 pas
a pli - 1e-6, lr 0,05, eps 1e-10), on bascule a delta = pli - x et eps voulu, et on suit la ligne 10 du recepteur :
somme du gradient des 25 logits de queue (en convention gradient de J, comme la reduction : g3 + g4 = -somme(queue)),
Se reel = somme exp(R[10,j] - R[10,4]) sur la queue, u = R[10,3] - R[10,4], d3 = 1 - s[3,10], R_b.
Usage : python verifier_tour68_queue_dynamique.py <eps> <pas> <x> [<pas_entre_lignes>]
"""

import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from verifier_tour59_delta_c_dynamique import depart, BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere
from hasard_reduction_tour59 import PHASE1

PLI = 0.0134372100660973
MSG = 10

if __name__ == "__main__":
    eps, pas, x = float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    every = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
    torch.set_num_threads(1)
    (e, r), msg = depart("mur23")
    activer(e, r)
    params = parametres(e, r)
    c = torch.load(PHASE1)
    with torch.no_grad():
        for p, v in zip(params, c["params"]):
            p.copy_(v)
    opt = torch.optim.Adam(params, lr=0.05, eps=1e-10)
    opt.load_state_dict(c["opt"])
    opt.param_groups[0]["eps"] = eps
    delta = PLI - x
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[3] = (1.0 - delta) / N
    poids[4] = (1.0 + delta) / N
    print(f"eps {eps:.0e} x {x:.0e} : pas | g3+g4 (grad J) | somme queue (grad J) | Se reel | u | d3 | moyenne du logit de queue - l4")
    t0 = 0
    for t in range(pas + 1):
        for p in params:
            p.grad = None
        j, _ = objectif_pondere(e, r, BETA, poids)
        (-j).backward()
        if t % every == 0:
            with torch.no_grad():
                gr = -r.p[0].grad[MSG].numpy()
                lg = r.p[0][MSG].numpy()
                tail_idx = [k for k in range(27) if k not in (3, 4)]
                Se = np.exp(lg[tail_idx] - lg[4]).sum()
                S = e.loi()
                print(f"{t:8d}  {gr[3] + gr[4]:+.5e}  {gr[tail_idx].sum():+.5e}  {Se:.4e}  {lg[3] - lg[4]:+.8f}  {1 - S[3, MSG].item():.8e}  {np.mean(lg[tail_idx] - lg[4]):+.4f}", flush=True)
        if t < pas:
            opt.step()
