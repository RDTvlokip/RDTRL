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


H_PROPRE = BETA / 27 / 27   # valeur propre de -beta H/N sur le sous-espace a somme nulle


def partie_g():
    """CORRECTION trouvee apres (e)-(f) : la courbure qui fixe le seuil de
    stabilite est la valeur propre du hessien sur le sous-espace a somme
    nulle (celui ou vit le softmax), h = beta/(N K) = 2,7435e-5, pas
    l'element diagonal beta(K-1)/(N K^2) utilise en (a)-(f). Avec elle,
    sqrt(v) predit = lr h/38 - eps colle a +1,1 % en (e) et (f).
    Consequence (JUSQU'OU), ecrite avant le run : au-dessus de
    eps_c = lr beta/(38 N K) = 3,61e-8, S = lr h/(sqrt v + eps) ne peut
    jamais atteindre 38 -> pas de bord de stabilite, l'orphelin converge
    EXACTEMENT. Predictions (replay standard, graine 77777 k=3, 40 000 pas,
    moyenne sur les 10 000 derniers) : deficit = 1/2 (lr/38 - eps/h)^2
    = 2,47e-8 a eps=3e-8 ; ~0 (niveau 1e-10 ou moins) a eps=5e-8 et 1e-7 ;
    4,52e-7 a eps=1e-8 (controle)."""
    from representable_atteignable_stable import activer, parametres, objectif
    for eps in (1e-8, 3e-8, 5e-8, 1e-7):
        e, r = replay(77777, 3, 0)
        activer(e, r)
        opt = torch.optim.Adam(parametres(e, r), lr=LR, eps=eps)
        p_e = e.p[0]
        defs = []
        for k in range(40000):
            j, _ = objectif(e, r, BETA)
            opt.zero_grad()
            (-j).backward()
            opt.step()
            if k >= 30000:
                with torch.no_grad():
                    defs.append(deficit(p_e[5]))
        with torch.no_grad():
            orph = r.loi()[:, 5].max().item() < 1e-6
        pred = 0.5 * max(0.0, LR / 38 - eps / H_PROPRE) ** 2
        print(f"(g) eps={eps:.0e}  ligne 5 orpheline={orph}  <deficit>={sum(defs)/len(defs):.4e}  "
              f"predit {pred:.4e}  (min {min(defs):.2e}, max {max(defs):.2e})")


def partie_h():
    """(g) est CADUC tel que concu : changer eps pour TOUT l'entrainement
    change le code appris (a eps >= 3e-8 le referent 5 recoit son propre
    message, H=0) -- le test ne porte plus sur un orphelin. Refait comme
    l'ablation de la ligne 3 : depuis le checkpoint mur23 (ligne 5
    orpheline, deja au bord de stabilite), eps change sur la SEULE ligne 5
    (correction exacte du pas apres opt.step()). Predictions ecrites avant
    le run, moyenne sur les 2000 derniers de 10 000 pas :
      eps5 = 1e-8 : retombe sur la valeur de (f), ~4,6e-7 ;
      eps5 = 3e-8 : S max = lr h/eps = 45,7 > 38 -> bord de stabilite
                    maintenu, deficit ~2,5-2,7e-8 ;
      eps5 = 5e-8 : S max = 27,4 < 38 -> convergence EXACTE, deficit -> ~0 ;
      eps5 = 1e-7 : S max = 13,7 < 38 -> convergence exacte."""
    for eps5 in (1e-8, 3e-8, 5e-8, 1e-7):
        e, r, opt, poids = reprendre(0.013026615)
        p_e = e.p[0]
        defs = []
        for k in range(10000):
            j, _ = objectif_pondere(e, r, BETA, poids)
            opt.zero_grad()
            (-j).backward()
            opt.step()
            with torch.no_grad():
                st = opt.state[p_e]
                m, rv = st["exp_avg"][5], st["exp_avg_sq"][5].sqrt()
                p_e[5] += LR * m / (rv + ADAM_EPS) - LR * m / (rv + eps5)
                if k >= 8000:
                    defs.append(deficit(p_e[5]))
        pred = 0.5 * max(0.0, LR / 38 - eps5 / H_PROPRE) ** 2
        print(f"(h) eps ligne 5 = {eps5:.0e}  S max sans v = {LR * H_PROPRE / eps5:6.1f}  "
              f"<deficit> = {sum(defs)/len(defs):.4e}  predit {pred:.4e}  "
              f"(min {min(defs):.2e}, max {max(defs):.2e})")


if __name__ == "__main__":
    torch.set_num_threads(1)
    quoi = sys.argv[1] if len(sys.argv) > 1 else "abcd"
    if "h" in quoi:
        partie_h()
    if "g" in quoi:
        partie_g()
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
