"""Tour 59 (29/09/2026) : E3, ou Adam agit-il ? eps grand sur UN groupe de
coordonnees, 1e-10 partout ailleurs, reseau complet (mur 23).

Masques :
  tout       eps sur toutes les coordonnees (temoin : doit reproduire torch.optim.Adam)
  ligne3     eps sur la ligne 3 de l'emetteur (logit de message ET 26 concurrents)
  recepteur  eps sur tout le recepteur
Protocole (celui de la grille) : chauffe 20 000 pas a pli-1e-6 sous eps 1e-10
(torch.optim.Adam, cache "phase 1" partage), puis 20 000 pas sous le masque
(AdamMasque, moments repris), puis delta et detection de la bascule
(R[10,4] > 0,9). Meme format de sortie que verifier_tour59_delta_c_dynamique.py.

Usage : python verifier_tour59_masque_eps.py mur23 3 4 <delta> <pas> masque=<nom> eps=<valeur>
Creer les caches par un appel a pas=0 avant de lancer en parallele.
"""

import os
import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere
from verifier_tour59_delta_c_dynamique import depart, ADAM_EPS, LR
from verifier_tour59_branches_fermees import F_PLI
from adam_eps_masque import AdamMasque

MSG = 10


def eps_masque(e, r, a, masque, eps):
    ee = torch.full_like(e.p[0], ADAM_EPS)
    er = torch.full_like(r.p[0], ADAM_EPS)
    if masque == "tout":
        ee.fill_(eps); er.fill_(eps)
    elif masque == "ligne3":
        ee[a, :] = eps
    elif masque == "recepteur":
        er.fill_(eps)
    else:
        raise ValueError(masque)
    return [ee, er]


def poids_pour(a, b, delta):
    p = torch.full((N,), 1.0 / N, dtype=torch.float64)
    p[a], p[b] = (1.0 - delta) / N, (1.0 + delta) / N
    return p


def courir(a, b, delta, pas, masque, eps, chauffe=20000, chauffe_eps=20000, beta2=0.999, cas="mur23"):
    torch.set_num_threads(1)
    (e, r), msg = depart(cas)  # msg : 10 pour mur23, 8 pour c814 (collision 6/14 de 12345 k=3)
    activer(e, r)
    params = parametres(e, r)
    tag_b = "" if beta2 == 0.999 else f"_b2{beta2}"
    ck1 = f"D:/tmp/rdtrl_tour59_phase1_{cas}_{a}_{b}_{chauffe}{tag_b}.pt"
    ck2 = f"D:/tmp/rdtrl_tour59_masque_{masque}_{cas}_{a}_{b}_eps{eps:.0e}_{chauffe}_{chauffe_eps}{tag_b}.pt"
    pc = poids_pour(a, b, F_PLI - 1e-6)
    opt = AdamMasque(params, LR, eps_masque(e, r, a, masque, eps), betas=(0.9, beta2))
    if os.path.exists(ck2):
        c = torch.load(ck2)
        with torch.no_grad():
            for p, v in zip(params, c["params"]):
                p.copy_(v)
        for m, v, mm, vv in zip(opt.m, opt.v, c["m"], c["v"]):
            m.copy_(mm); v.copy_(vv)
        opt.t = c["t"]
    else:
        if os.path.exists(ck1):
            c1 = torch.load(ck1)
            with torch.no_grad():
                for p, v in zip(params, c1["params"]):
                    p.copy_(v)
            opt1 = torch.optim.Adam(params, lr=LR, eps=ADAM_EPS, betas=(0.9, beta2))
            opt1.load_state_dict(c1["opt"])
        else:
            opt1 = torch.optim.Adam(params, lr=LR, eps=ADAM_EPS, betas=(0.9, beta2))
            for _ in range(chauffe):
                j, _ = objectif_pondere(e, r, BETA, pc)
                opt1.zero_grad(); (-j).backward(); opt1.step()
            torch.save({"params": [p.detach().clone() for p in params], "opt": opt1.state_dict()}, ck1)
        opt.charger_depuis_torch(opt1)
        for _ in range(chauffe_eps):
            j, _ = objectif_pondere(e, r, BETA, pc)
            opt.zero_grad(); (-j).backward(); opt.step()
        torch.save({"params": [p.detach().clone() for p in params], "m": [m.clone() for m in opt.m],
                    "v": [v.clone() for v in opt.v], "t": opt.t}, ck2)
    poids = poids_pour(a, b, delta)
    bascule = -1
    for t in range(pas):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad(); (-j).backward(); opt.step()
        with torch.no_grad():
            S, R = e.loi(), r.loi()
            if (S[b, msg] * R[msg, b]).item() > 0.9:
                bascule = t
                break
    with torch.no_grad():
        S, R = e.loi(), r.loi()
        Rb = (S[b, msg] * R[msg, b]).item()
        d = 1 - S[a, msg].item()
    print(f"{cas} a={a} b={b} delta={delta!r} eps={eps:.0e} masque={masque} chauffe_eps={chauffe_eps} beta2={beta2} "
          f"pas={pas} R_b={Rb:.6f} 1-s_a={d:.6e} bascule={bascule}", flush=True)


if __name__ == "__main__":
    a, b, delta, pas = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
    opts = sys.argv[6:]
    masque = next(o[7:] for o in opts if o.startswith("masque="))
    eps = next(float(o[4:]) for o in opts if o.startswith("eps="))
    # chauffe_eps= : nombre de pas de la phase sous masque (20 000 par defaut) ; le faire varier de
    # quelques pas decale la PHASE des salves au moment ou delta est applique.
    chauffe_eps = next((int(o[12:]) for o in opts if o.startswith("chauffe_eps=")), 20000)
    beta2 = next((float(o[6:]) for o in opts if o.startswith("beta2=")), 0.999)
    courir(a, b, delta, pas, masque, eps, chauffe_eps=chauffe_eps, beta2=beta2, cas=sys.argv[1])
