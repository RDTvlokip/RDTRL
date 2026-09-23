"""Tour 58 (23/09/2026) : le "referent 5 a non-convergence totale" (21/09,
H~ln27, mecanisme laisse ouvert) releu dans sa version la plus triviale.

Constat (verifie avant ce script) : le referent 5 est ORPHELIN -- aucun
message ne decode vers lui (max_m r[m,5]=6,3e-11 ; les paires synonymes
8 et 12 et la collision 3/4 consomment les 27 messages : 22+2+2+1). Sa
recompense brute vaut 8,6e-12 : sa ligne d'emetteur n'optimise que
l'entropie, et l'uniforme est son optimum EXACT. Pas une non-convergence.

Reste a expliquer pourquoi elle s'arrete a ln27 - 8,07e-7 et pas a
l'optimum. Hypothese (H58.11) : chaque coordonnee de la ligne 5 oscille
en periode 2 autour de l'optimum, au bord de stabilite d'Adam. Pour une
oscillation de periode 2 de gradient +-g : |m| = (1-b1)/(1+b1) g,
sqrt(v) = g, pas = lr (1-b1)/(1+b1) = lr/19, amplitude A = lr/38 --
independante de la courbure et d'eps (tant que sqrt(v) >> eps).
PREDICTIONS ecrites avant le run :
  (a) |pas| par coordonnee (mode commun retire) ~ lr/19 = 2,632e-3,
      signe alternant a chaque pas ;
  (b) deficit d'entropie ln27 - H = 1/2 A^2 (26/27) = 8,34e-7 (mesure 8,07e-7) ;
  (c) lr de la seule ligne 5 multiplie par f : deficit ~ f^2
      (f=0,5 -> 2,1e-7 ; f=0,25 -> 5,2e-8), pas ~ f ;
  (d) generalite : toute ligne orpheline d'une autre configuration
      (graine 77777 k=5, graine 12345 k=3, replay standard lr=0,05)
      a le meme deficit, quelle que soit la graine.
Correction pour eps non negligeable (ecrite aussi avant le run) : a
l'etat stationnaire 2A (hA + eps) = lr hA/19, donc A = lr/38 - eps/h,
h = beta (K-1)/(N K^2) = 2,642e-5 (courbure de -beta H/N a l'uniforme).
  mur23 (eps=1e-10) : A = 1,3120e-3, deficit 8,29e-7 ;
  replay standard (Adam par defaut, eps=1e-8) : A = 9,37e-4, deficit 4,23e-7.
"""

import sys
sys.path.insert(0, '.')
import math
import torch

from replay_mur23_referent3 import replay, BETA
from verifier_prior_asymetrique import objectif_pondere
from sauver_checkpoints_mur23_tour58 import reprendre, LR, ADAM_EPS

A = LR / 38
DEFICIT_PREDIT = 0.5 * A * A * 26 / 27


def deficit(ligne):
    s = torch.softmax(ligne, 0)
    return math.log(27) + (s * s.log()).sum().item()


def partie_ab():
    e, r, opt, poids = reprendre(0.013026615)
    p_e = e.p[0]
    lignes = [p_e[5].detach().clone()]
    for _ in range(200):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        lignes.append(p_e[5].detach().clone())
    L = torch.stack(lignes)
    d = L[1:] - L[:-1]
    d = d - d.mean(dim=1, keepdim=True)          # retire le mode commun
    alterne = ((d[1:] * d[:-1]) < 0).double().mean().item()
    print(f"(a) |pas| par coordonnee, mediane = {d.abs().median().item():.4e}   predit lr/19 = {LR/19:.4e}")
    print(f"    fraction de pas qui changent de signe d'un pas au suivant = {alterne:.4f}")
    print(f"(b) deficit d'entropie ligne 5 = {deficit(L[-1]):.4e}   predit 1/2 (lr/38)^2 (26/27) = {DEFICIT_PREDIT:.4e}")
    ecart = (L[-1] - L[-1].mean())
    print(f"    ecart-type des logits de la ligne 5 = {ecart.std(unbiased=False).item():.4e}   predit lr/38 = {A:.4e}")


def partie_c():
    for f in (1.0, 0.5, 0.25):
        e, r, opt, poids = reprendre(0.013026615)
        p_e = e.p[0]
        prec = p_e[5].detach().clone()
        pas_abs = []
        for k in range(4000):
            j, _ = objectif_pondere(e, r, BETA, poids)
            opt.zero_grad()
            (-j).backward()
            opt.step()
            with torch.no_grad():
                if f != 1.0:
                    st = opt.state[p_e]
                    p_e[5] += (1 - f) * LR * st["exp_avg"][5] / (st["exp_avg_sq"][5].sqrt() + ADAM_EPS)
                if k >= 3800:
                    dd = p_e[5] - prec
                    pas_abs.append((dd - dd.mean()).abs().median().item())
                prec = p_e[5].detach().clone()
        print(f"(c) lr ligne 5 x {f}: deficit apres 4000 pas = {deficit(p_e[5].detach()):.4e}   "
              f"predit {DEFICIT_PREDIT * f * f:.4e}   |pas| median = {sorted(pas_abs)[len(pas_abs)//2]:.4e}   "
              f"predit {f * LR / 19:.4e}")


def partie_d():
    for graine, k in ((77777, 3), (77777, 5), (12345, 3), (77777, 1)):
        e, r = replay(graine, k, 10000)
        with torch.no_grad():
            S, R = e.loi(), r.loi()
            orphelins = [i for i in range(27) if R[:, i].max().item() < 1e-6]
            for i in range(27):
                H = -(S[i] * S[i].log()).sum().item()
                if H > 3.0 or i in orphelins:
                    print(f"(d) graine={graine} k={k} ligne={i}  orpheline={i in orphelins}  "
                          f"max_m r[m,i]={R[:, i].max().item():.2e}  deficit={math.log(27) - H:.4e}  "
                          f"(predit {DEFICIT_PREDIT:.4e} si orpheline)")


if __name__ == "__main__":
    torch.set_num_threads(1)
    quoi = sys.argv[1] if len(sys.argv) > 1 else "abcd"
    if "a" in quoi or "b" in quoi:
        partie_ab()
    if "c" in quoi:
        partie_c()
    if "d" in quoi:
        partie_d()
