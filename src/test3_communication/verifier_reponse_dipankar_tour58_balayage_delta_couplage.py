"""Tour 58 (23/09/2026, VRAIE critique de dipankarsarkar) : JUSQU'OU le
couplage s3 <-> salve du recepteur s'allume-t-il quand delta monte de 0
a sa valeur reelle ?

Mecanisme (verifie par ablation d'eps sur la ligne 3,
verifier_reponse_dipankar_tour58_ablation_eps_ligne3.py) : pente dX/dgap
= 1/2 [u/(u+eps) + (u/26)/(u/26+eps)], u = sqrt(v) d'Adam sur e3[10].
Equilibre en forme fermee (reproduit l'etat delta reel a tous les
chiffres) : r3 = sigmoid(-((1+d)s4-(1-d)s3)/beta), X* = (1-d) r3/beta,
1-s3 = 26 exp(-X*)/(1+26 exp(-X*)) ; et u ~ s3(1-s3)(1-d) r3 r4, etalonne
sur u=1,88e-8 a delta reel.

PREDICTIONS A PRIORI, ecrites avant de lancer :
  delta   0,002    0,004    0,006    0,008   0,010   0,012
  u       1,3e-13  1,5e-12  1,5e-11  1,3e-10 9,6e-10 6,3e-9
  pente   6,6e-4   7,5e-3   6,8e-2   0,31    0,59    0,85
Transition (pente = moitie du plafond) vers delta ~ 0,009.
Deux niveaux de test : (1) la pente mesuree contre la prediction a
priori ; (2) la pente mesuree contre la formule evaluee avec le u
MESURE (teste la formule seule, independamment de la loi d'echelle de u
et du fait que l'etat ait fini de relaxer vers X*).

Usage : python verifier_reponse_dipankar_tour58_balayage_delta_couplage.py <delta>
"""

import sys
sys.path.insert(0, '.')
import math
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere

ADAM_EPS = 1e-10
LR = 0.05
DEBUT, PAS_MAX = 56000, 62000


def formule(u, eps=ADAM_EPS):
    return 0.5 * (u / (u + eps) + (u / 26) / (u / 26 + eps))


def main(delta):
    torch.set_num_threads(1)
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + delta) / N
    poids[3] = (1.0 - delta) / N
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=LR, eps=ADAM_EPS)
    p_e, p_r = e.p[0], r.p[0]
    autres = [j for j in range(N) if j != 10]
    gaps, Xs, us = [], [], []
    for pas in range(PAS_MAX):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        if pas >= DEBUT:
            with torch.no_grad():
                gaps.append((p_r[10, 4] - p_r[10, 3]).item())
                Xs.append((p_e[3, 10] - p_e[3, autres].mean()).item())
                us.append(opt.state[p_e]["exp_avg_sq"][3, 10].sqrt().item())
    with torch.no_grad():
        s3 = torch.softmax(p_e[3], dim=0)[10].item()
        un_moins_s3 = torch.exp(torch.logsumexp(p_e[3, autres], 0) - torch.logsumexp(p_e[3], 0)).item()
        rr = torch.softmax(p_r[10], dim=0)
    print(f"=== delta={delta} === etat final : X={Xs[-1]:.5f}  1-s3={un_moins_s3:.4e}  "
          f"r3={rr[3].item():.5f}  r4={rr[4].item():.5f}")

    dg = [gaps[i] - gaps[i - 1] for i in range(1, len(gaps))]
    dx = [Xs[i] - Xs[i - 1] for i in range(1, len(Xs))]
    idx = [i for i, v in enumerate(dg) if abs(v) > 0.004]
    salves = []
    for i in idx:
        if salves and i - salves[-1][-1] <= 50:
            salves[-1].append(i)
        else:
            salves.append([i])
    lignes = []
    for s in salves:
        a, b = max(0, s[0] - 5), min(len(dg), s[-1] + 6)
        hors = sorted(dx[max(0, a - 40):a])
        c = hors[len(hors) // 2] if hors else 0.0
        pente = sum((dx[i] - c) * dg[i] for i in range(a, b)) / sum(dg[i] ** 2 for i in range(a, b))
        pic = max(range(a, b), key=lambda i: abs(dg[i]))
        u = us[pic + 1]
        lignes.append((DEBUT + s[0] + 1, max(abs(dg[i]) for i in range(a, b)), pente, u, formule(u)))
        print(f"  salve pas={DEBUT + s[0] + 1}  max|dgap|={lignes[-1][1]:.4e}  pente={pente:+.4e}  "
              f"u={u:.4e}  formule(u mesure)={formule(u):.4e}  rapport={pente / formule(u):.4f}")
    if lignes:
        med = lambda v: sorted(v)[len(v) // 2]
        print(f"  RESUME delta={delta} : {len(lignes)} salves, pente mediane={med([l[2] for l in lignes]):.4e}  "
              f"u median={med([l[3] for l in lignes]):.4e}  "
              f"rapport mesure/formule median={med([l[2] / l[4] for l in lignes]):.4f}")


if __name__ == "__main__":
    main(float(sys.argv[1]))
