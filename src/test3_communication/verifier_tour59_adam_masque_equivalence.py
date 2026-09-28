"""Tour 59 (29/09/2026) : AdamMasque avec eps uniforme reproduit-il
torch.optim.Adam ? 3000 pas sur l'objectif reel du mur 23 (delta = pli -
1e-6), a partir de l'etat chauffe eps 1e-7 (moments non nuls), et aussi
chargement des moments d'un Adam torch puis poursuite. Ecart maximal des
parametres attendu au niveau de l'arrondi (<= 1e-12)."""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere
from verifier_tour59_delta_c_dynamique import depart, LR
from adam_eps_masque import AdamMasque
from verifier_tour59_branches_fermees import F_PLI

A, B_ = 3, 4


def poids(delta):
    p = torch.full((N,), 1.0 / N, dtype=torch.float64)
    p[A], p[B_] = (1 - delta) / N, (1 + delta) / N
    return p


def charger(ck):
    (e, r), _ = depart("mur23")
    activer(e, r)
    c = torch.load(ck)
    prm = parametres(e, r)
    with torch.no_grad():
        for p, v in zip(prm, c["params"]):
            p.copy_(v)
    return e, r, prm, c


if __name__ == "__main__":
    torch.set_num_threads(1)
    delta = F_PLI - 1e-6
    w = poids(delta)
    # argument optionnel : eps du cache chaud (ex. 3e-06, regime SANS salves : l'equivalence doit
    # y tenir a l'arrondi ; a 1e-07 les salves amplifient tout ecart, cf. le temoin ci-dessous)
    EPS = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-7
    ck = f"D:/tmp/rdtrl_tour59_chaud_mur23_3_4_eps{EPS:.0e}_20000_20000.pt"
    e1, r1, p1, c = charger(ck)
    opt1 = torch.optim.Adam(p1, lr=LR, eps=EPS)
    opt1.load_state_dict(c["opt"])
    opt1.param_groups[0]["eps"] = EPS
    e2, r2, p2, _ = charger(ck)
    opt2 = AdamMasque(p2, LR, [torch.full_like(p, EPS) for p in p2])
    o_tmp = torch.optim.Adam(p2, lr=LR, eps=EPS)
    o_tmp.load_state_dict(c["opt"])
    opt2.charger_depuis_torch(o_tmp)
    # temoin de chaos : un troisieme Adam torch identique au premier, perturbe de 1e-15 sur un
    # logit ; la dynamique (salves du bord de stabilite) amplifie tout ecart, donc l'ecart
    # AdamMasque/torch a 3000 pas ne se lit que contre l'ecart torch/torch-perturbe.
    e3, r3, p3, _ = charger(ck)
    opt3 = torch.optim.Adam(p3, lr=LR, eps=EPS)
    opt3.load_state_dict(c["opt"])
    opt3.param_groups[0]["eps"] = EPS
    with torch.no_grad():
        p3[0][0, 0] += 1e-15
    jalons = (1, 10, 100, 300, 1000, 3000)
    for t in range(1, 3001):
        for (e, r, o) in ((e1, r1, opt1), (e2, r2, opt2), (e3, r3, opt3)):
            j, _ = objectif_pondere(e, r, BETA, w)
            o.zero_grad()
            (-j).backward()
            o.step()
        if t in jalons:
            d12 = max(float((a - b).abs().max()) for a, b in zip(p1, p2))
            d13 = max(float((a - b).abs().max()) for a, b in zip(p1, p3))
            print(f"pas {t:5d} : |torch - AdamMasque| = {d12:.3e}   |torch - torch perturbe 1e-15| = {d13:.3e}")
