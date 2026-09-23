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


H_COURBURE = BETA / 27 * 26 / 27 / 27   # diagonale du hessien de -beta H/N a l'uniforme


def deficit_eos(lr, eps):
    """Hypothese REVISEE (23/09, apres les parties a-d : le pas median valait
    lr/39 et non lr/19, chaque coordonnee fait des salves intermittentes et
    non un cycle limite). Ce qui se fixe est sqrt(v) : auto-stabilisation a
    S = lr h/(sqrt(v)+eps) = 38, donc sqrt(v) = lr h/38 - eps ; et comme
    v = h^2 <(z - zbar)^2>, le deficit MOYEN DANS LE TEMPS vaut
    1/2 (lr/38 - eps/h)^2."""
    return 0.5 * (lr / 38 - eps / H_COURBURE) ** 2


def partie_e():
    """Moyennes dans le temps (dernier quart de 8000 pas) de sqrt(v) et du
    deficit de la ligne 5, lr de la ligne 5 multiplie par f. Predictions
    (ecrites avant le run) : sqrt(v) = f lr h/38 - eps ; deficit moyen
    8,61e-7 / 2,14e-7 / 5,29e-8 pour f = 1 / 0,5 / 0,25."""
    for f in (1.0, 0.5, 0.25):
        e, r, opt, poids = reprendre(0.013026615)
        p_e = e.p[0]
        defs, svs = [], []
        for k in range(8000):
            j, _ = objectif_pondere(e, r, BETA, poids)
            opt.zero_grad()
            (-j).backward()
            opt.step()
            with torch.no_grad():
                st = opt.state[p_e]
                if f != 1.0:
                    p_e[5] += (1 - f) * LR * st["exp_avg"][5] / (st["exp_avg_sq"][5].sqrt() + ADAM_EPS)
                if k >= 6000:
                    defs.append(deficit(p_e[5]))
                    svs.append(st["exp_avg_sq"][5].sqrt().mean().item())
        print(f"(e) lr ligne 5 x {f}: <sqrt v> = {sum(svs)/len(svs):.4e}  predit {f*LR*H_COURBURE/38 - ADAM_EPS:.4e}   "
              f"<deficit> = {sum(defs)/len(defs):.4e}  predit {deficit_eos(f*LR, ADAM_EPS):.4e}  "
              f"(min {min(defs):.2e}, max {max(defs):.2e})")


def partie_f():
    """Replay standard (graine 77777, k=3, Adam par defaut eps=1e-8, lr=0,05)
    prolonge a 80 000 pas : les lignes orphelines (4 et 5 a 10 000 pas)
    sont uniformes a 1e-11 a 10 000 pas. Prediction : v y decroit jusqu'au
    seuil, puis le regime de bord de stabilite s'installe avec un deficit
    moyen 1/2 (lr/38 - 1e-8/h)^2 = 4,39e-7 et sqrt(v) = lr h/38 - 1e-8 = 2,48e-8."""
    from representable_atteignable_stable import activer, parametres, objectif
    e, r = replay(77777, 3, 0)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=LR)
    p_e = e.p[0]
    acc = {4: [], 5: []}
    sv = {4: [], 5: []}
    for k in range(80000):
        j, _ = objectif(e, r, BETA)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        with torch.no_grad():
            for i in (4, 5):
                acc[i].append(deficit(p_e[i]))
                sv[i].append(opt.state[p_e]["exp_avg_sq"][i].sqrt().mean().item())
            if (k + 1) % 5000 == 0:
                R = r.loi()
                ligne = "  ".join(f"ligne {i}: orpheline={R[:, i].max().item() < 1e-6} "
                                  f"<deficit>={sum(acc[i])/len(acc[i]):.3e} <sqrt v>={sum(sv[i])/len(sv[i]):.3e}"
                                  for i in (4, 5))
                print(f"(f) pas {k+1:6d}  {ligne}   (predit a terme {deficit_eos(LR, 1e-8):.3e}, "
                      f"sqrt v {LR*H_COURBURE/38 - 1e-8:.3e})")
                acc = {4: [], 5: []}
                sv = {4: [], 5: []}


if __name__ == "__main__":
    torch.set_num_threads(1)
    quoi = sys.argv[1] if len(sys.argv) > 1 else "abcd"
    if "e" in quoi:
        partie_e()
    if "f" in quoi:
        partie_f()
    if "a" in quoi or "b" in quoi:
        partie_ab()
    if "c" in quoi:
        partie_c()
    if "d" in quoi:
        partie_d()
