"""Tour 59 : sous le pli (pli-1e-8), d3 = 1-s3 depasse la coordonnee d3 du
jumeau instable jusqu'a 2,4x l'ecart, sans bascule. Hypotheses testees
ici, pas a pas (pas 20 000 a 24 000) :
  H59-4  l'excursion de d3 vient de l'ETALEMENT des 26 logits concurrents
         (Jensen : sum e^{z_m - z_msg} > 26 e^{mean z_m - z_msg}), pas d'un
         deplacement de X3 = z_msg - mean z_m, la variable du pli ;
  H59-5  la variable lente est le recepteur r4, qui fluctue peu.
On enregistre d3, d3_champ_moyen = 26 e^{-X3}/(1+26 e^{-X3}), X3, r4.
"""

import sys
sys.path.insert(0, '.')
import numpy as np
import torch

from verifier_tour59_delta_c_dynamique import depart, ADAM_EPS, LR, BETA
from verifier_tour59_branches_fermees import branche_et_jumeau, mesures, F_PLI
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere

A, B_, MSG = 3, 4, 10


def run(delta, debut=20000, fin=24000):
    torch.set_num_threads(1)
    (e, r), _ = depart("mur23")
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[A], poids[B_] = (1 - delta) / N, (1 + delta) / N
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=LR, eps=ADAM_EPS)
    rows = []
    for t in range(fin):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        if t >= debut:
            with torch.no_grad():
                z = e.p[0][A]
                autres = torch.cat([z[:MSG], z[MSG + 1:]])
                X = (z[MSG] - autres.mean()).item()
                d3 = 1 - e.loi()[A, MSG].item()
                mf = 26 * np.exp(-X) / (1 + 26 * np.exp(-X))
                rows.append((t, d3, mf, X, r.loi()[MSG, B_].item(), autres.std().item()))
    return np.array(rows)


if __name__ == "__main__":
    off = float(sys.argv[1]) if len(sys.argv) > 1 else -1e-8
    d = F_PLI + off
    T = run(d)
    rs = branche_et_jumeau(d)
    (dA, _, rA), (dJ, _, rJ) = mesures(rs[0], d), mesures(rs[1], d)
    XA = (1 - d) * (1 / (1 + np.exp(-rs[0]))) / 0.02
    XJ = (1 - d) * (1 / (1 + np.exp(-rs[1]))) / 0.02
    np.savetxt(f"D:/tmp/rdtrl_tour59_jensen_mur23_off{off}.txt", T)
    fd = (T[:, 1] - dA) / (dJ - dA)
    fm = (T[:, 2] - dA) / (dJ - dA)
    fX = (T[:, 3] - XA) / (XJ - XA)
    fr = (T[:, 4] - rA) / (rJ - rA)
    print(f"pli{off:+.0e} : X3 stable {XA:.6f} jumeau {XJ:.6f}")
    for nom, f in (("d3 reel", fd), ("d3 champ moyen (X3 seul)", fm), ("X3", fX), ("r4", fr)):
        print(f"  {nom:26s} frac vers jumeau : min {f.min():+.4f} max {f.max():+.4f} moyenne {f.mean():+.4f} ecart-type {f.std():.4f}")
    print(f"  ecart-type des 26 logits concurrents : min {T[:, 5].min():.3e} max {T[:, 5].max():.3e}")
    print(f"  d3 reel / d3 champ moyen : min {np.min(T[:, 1] / T[:, 2]):.6f} max {np.max(T[:, 1] / T[:, 2]):.6f}")
    print(f"  correlation(d3 reel, ecart-type concurrents) = {np.corrcoef(T[:, 1], T[:, 5])[0, 1]:+.4f}")
    print(f"  correlation(d3 reel, X3) = {np.corrcoef(T[:, 1], T[:, 3])[0, 1]:+.4f}")
