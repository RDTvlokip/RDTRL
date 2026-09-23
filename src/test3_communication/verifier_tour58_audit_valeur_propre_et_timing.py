"""Tour 58 (23/09/2026) : verification INDEPENDANTE (regle 5bis) des deux
affirmations les plus lourdes de l'agent style dipankar sur le coeur du
tour 58 -- avec mon propre code, pas le sien.

1. VALEUR PROPRE, pas seulement vecteur propre. Il affirme que ma pente
   (0,937 / 4,75e-5) n'est que le VECTEUR propre du mode instable, et que
   la valeur propre porte un decalage
       Delta S = lambda_mode - S_gap = pente * S_gap * kappa,
       kappa = H(e3[10], r10[4]) / H(r10[4], r10[4]) = (1-delta) s3(1-s3)/beta,
   lineaire en s3(1-s3), sans eps. Chiffres qu'il donne (eigh exact de la
   matrice 1458x1458 preconditionnee, delta reel) : a pas=59480 S_gap
   36,7052, lambda_mode 38,4605 ; a 59520 S_gap 37,4483, lambda 39,2368,
   Delta S 1,7885 ; kappa = 0,051118. Et mon iteration de puissance
   (verifier_tour58_bord_stabilite_salve.py) serait restee coincee sur le
   paquet de la ligne 5 : 37,99 loggue a 59480 contre 38,46 exact.
   Ici : hessien complet par autograd, eigh, aux memes pas.

2. TIMING. Geler la ligne 3 (pas d'Adam multiplie par k, correction
   exacte apres opt.step()) a partir de pas=59600 decalerait le pic de la
   salve suivante (base 59989) : k=0 -> 60095 (+106), k=2 -> 59878
   (-111) ; a delta=0, aucun decalage (59580 pour k=0 et k=2). Ses
   predictions a priori (fichier date avant son run) : +94 / -101.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import BETA
from representable_atteignable_stable import parametres
from verifier_prior_asymetrique import objectif_pondere
from sauver_checkpoints_mur23_tour58 import reprendre, LR, ADAM_EPS

N = 27


def avancer(e, r, opt, poids, n):
    for _ in range(n):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()


def spectre(e, r, opt, poids):
    params = parametres(e, r)
    formes = [p.shape for p in params]
    tailles = [p.numel() for p in params]
    plat0 = torch.cat([p.detach().flatten() for p in params])

    # hessien explicite via autograd sur une fonction pure des logits
    from representable_atteignable_stable import N as NN

    def moins_J(plat):
        pe, pr = torch.split(plat, tailles)
        pe, pr = pe.view(formes[0]), pr.view(formes[1])
        s, rr = torch.softmax(pe, 1), torch.softmax(pr, 1)
        recompense = (poids.unsqueeze(1) * s * rr.t()).sum()
        ent_s = -(s * torch.log(s.clamp_min(1e-300))).sum() / NN
        ent_r = -(rr * torch.log(rr.clamp_min(1e-300))).sum() / NN
        return -(recompense + BETA * (ent_s + ent_r))

    H = torch.autograd.functional.hessian(moins_J, plat0)
    D = torch.cat([(opt.state[p]["exp_avg_sq"].sqrt() + ADAM_EPS).flatten() for p in params])
    dmh = D.rsqrt()
    M = LR * dmh.unsqueeze(1) * H * dmh.unsqueeze(0)
    lam, vec = torch.linalg.eigh(M)
    i_r4 = tailles[0] + 10 * N + 4
    i_r3 = tailles[0] + 10 * N + 3
    i_e310 = 3 * N + 10
    # S_gap : quotient de Rayleigh preconditionne sur la direction du gap
    d = torch.zeros_like(plat0)
    d[i_r4], d[i_r3] = 1.0, -1.0
    y = d / dmh
    y = y / y.norm()
    s_gap = (y @ M @ y).item()
    # H(e3[10], r10[4]) / H(r10[4], r10[4])
    kappa = (H[i_e310, i_r4] / H[i_r4, i_r4]).item()
    top = vec[:, -1]
    part_gap = (top[i_r4] ** 2 + top[i_r3] ** 2).item()
    part_e3 = (top[3 * N:4 * N] ** 2).sum().item()
    part_e5 = (top[5 * N:6 * N] ** 2).sum().item()
    return lam, s_gap, kappa, part_gap, part_e3, part_e5, H[i_r4, i_r4].item()


def partie_1():
    e, r, opt, poids = reprendre(0.013026615)
    fait = 54000
    for pas in (59480, 59520):
        avancer(e, r, opt, poids, pas - fait)
        fait = pas
        lam, s_gap, kappa, pg, pe3, pe5, h44 = spectre(e, r, opt, poids)
        print(f"(1) delta reel, pas={pas} : lambda_max exact = {lam[-1].item():.4f}  "
              f"(suivants {lam[-2].item():.4f}, {lam[-3].item():.4f})  S_gap = {s_gap:.4f}  "
              f"Delta S = {lam[-1].item() - s_gap:.4f}")
        print(f"    vecteur propre dominant : part gap {pg:.4f}, ligne 3 {pe3:.4f}, ligne 5 {pe5:.4f}   "
              f"kappa = {kappa:.6f}   H(r4,r4) = {h44:.5e}   pente*S_gap*kappa = {0.937 * s_gap * kappa:.4f}")


def pic_suivant(delta, k, p_switch, duree=700):
    e, r, opt, poids = reprendre(delta)
    avancer(e, r, opt, poids, p_switch - 54000)
    p_e, p_r = e.p[0], r.p[0]
    gaps = []
    for t in range(duree):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        with torch.no_grad():
            if k != 1:
                st = opt.state[p_e]
                pas3 = LR * st["exp_avg"][3] / (st["exp_avg_sq"][3].sqrt() + ADAM_EPS)
                p_e[3] += (1 - k) * pas3   # opt.step() a fait -pas3 ; on laisse -k*pas3
            gaps.append((p_r[10, 4] - p_r[10, 3]).item())
    base = sorted(gaps)[len(gaps) // 2]
    # premiere salve seulement : premier depassement de 0,004, puis pic dans les 50 pas
    i0 = next(i for i in range(len(gaps)) if abs(gaps[i] - base) > 0.004)
    i = max(range(i0, min(i0 + 50, len(gaps))), key=lambda i: abs(gaps[i] - base))
    return p_switch + i, gaps[i] - base


def partie_2():
    for delta, p_switch, cibles in ((0.013026615, 59600, {1: 59989, 0: 60095, 2: 59878}),
                                    (0.0, 59200, {1: 59579, 0: 59580, 2: 59580})):
        for k in (1, 0, 2):
            pic, amp = pic_suivant(delta, k, p_switch)
            print(f"(2) delta={delta} k={k} : pic de la salve suivante a pas={pic} (amplitude {amp:+.4e})   "
                  f"agent : {cibles[k]}")


if __name__ == "__main__":
    torch.set_num_threads(1)
    quoi = sys.argv[1] if len(sys.argv) > 1 else "12"
    if "1" in quoi:
        partie_1()
    if "2" in quoi:
        partie_2()
