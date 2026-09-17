"""Corrige le probleme trouve en fin de session precedente : la pente
de relaxation de r3 pres du pli ne se stabilise jamais parce que
r3_final (derniere valeur d'un run D'ENTRAINEMENT fini) est elle-meme
une cible mobile (ralentissement critique). Ici, le point fixe (s3*,
s4*, r3*, r4*) du jouet a M categories de fond est resolu ALGEBRIQUEMENT
(systeme couple a 4 equations, iteration de point fixe -- pas de
gradient, pas de ralentissement critique de descente de gradient) :

  s3 = sigmoid(N*poids3*r3/beta)
  s4 = sigmoid(N*poids4*r4/beta)
  r3 = exp(N*poids3*s3/beta) / Z
  r4 = exp(N*poids4*s4/beta) / Z
  Z  = exp(N*poids3*s3/beta) + exp(N*poids4*s4/beta) + M

Ce point fixe sert de reference FIXE (pas la derniere valeur d'un run)
pour remesurer proprement le taux de relaxation de r3 vers sa vraie
valeur finale, sans le biais de cible mobile.
"""

import sys
sys.path.insert(0, '.')
import math
import torch

from verifier_jouet_n_variable import construire_toy_m, objectif_toy_m, N


def point_fixe(M, delta, tol=1e-15, max_iter=100000):
    poids3 = (1.0 - delta) / N
    poids4 = (1.0 + delta) / N
    s3, s4, r3, r4 = 0.999999, 0.999999, 0.5, 0.5
    for it in range(max_iter):
        s3n = 1.0 / (1.0 + math.exp(-N * poids3 * r3 / 0.02))
        s4n = 1.0 / (1.0 + math.exp(-N * poids4 * r4 / 0.02))
        e3 = math.exp(min(N * poids3 * s3n / 0.02, 700))
        e4 = math.exp(min(N * poids4 * s4n / 0.02, 700))
        Z = e3 + e4 + M
        r3n = e3 / Z
        r4n = e4 / Z
        delta_max = max(abs(s3n - s3), abs(s4n - s4), abs(r3n - r3), abs(r4n - r4))
        s3, s4, r3, r4 = s3n, s4n, r3n, r4n
        if delta_max < tol:
            break
    return s3, s4, r3, r4, it


def relaxation_vers_point_fixe(M, r_autres_init, delta, r3_cible, pas=400, lr=0.2, adam_eps=1e-10):
    poids3 = torch.tensor((1.0 - delta) / N, dtype=torch.float64)
    poids4 = torch.tensor((1.0 + delta) / N, dtype=torch.float64)
    p3, p4, q = construire_toy_m(M, delta, r_autres_init=r_autres_init)
    opt = torch.optim.Adam([p3, p4, q], lr=lr, eps=adam_eps)
    r3_tr = []
    with torch.no_grad():
        r_all0 = torch.softmax(q, dim=0)
        r3_tr.append(r_all0[0].item())
    for i in range(pas):
        j = objectif_toy_m(p3, p4, q, poids3, poids4)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        with torch.no_grad():
            r_all = torch.softmax(q, dim=0)
            r3_tr.append(r_all[0].item())
    return r3_tr


def pente_vs_cible_fixe(r3_tr, r3_cible, t_debut, t_fin):
    xs, ys = [], []
    for t in range(t_debut, min(t_fin, len(r3_tr))):
        ecart = abs(r3_tr[t] - r3_cible)
        if ecart > 1e-14:
            xs.append(t)
            ys.append(math.log(ecart))
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx


def fenetre_adaptative(r3_tr, r3_cible, plancher_bruit=1e-6):
    """Trouve automatiquement, pour CE jouet, la fenetre [t_debut, t_fin)
    du dernier extremum local de r3(t) jusqu'a ce que |r3-cible| tombe
    sous le plancher de bruit numerique -- evite (a) de mesurer une
    pente sur une fenetre qui traverse encore un rebond/depassement,
    (b) de mesurer du bruit flottant une fois converge."""
    ecarts = [r3_tr[t] - r3_cible for t in range(len(r3_tr))]
    # dernier changement de signe de la DERIVEE (extremum local) sur
    # les 200 premiers pas (au-dela, la dynamique est censee etre lisse)
    derniere_extremum = 0
    for t in range(1, min(200, len(ecarts) - 1)):
        if (ecarts[t] - ecarts[t - 1]) * (ecarts[t + 1] - ecarts[t]) < 0:
            derniere_extremum = t
    t_debut = derniere_extremum + 2
    t_fin = t_debut
    for t in range(t_debut, len(r3_tr)):
        if abs(r3_tr[t] - r3_cible) < plancher_bruit:
            break
        t_fin = t + 1
    return t_debut, t_fin


if __name__ == "__main__":
    configs = [
        ("M=0 (sans fond)", 0, 1e-6, 0.95 * 0.018672),
        ("M=25, masse totale=25%", 25, 0.01, 0.95 * 0.018516),
    ]

    resultats = {}
    for label, M, r_autres_init, delta in configs:
        s3f, s4f, r3f, r4f, it = point_fixe(M, delta)
        print(f"=== {label} (delta={delta:.6f}) ===")
        print(f"  point fixe algebrique (converge en {it} iterations) : "
              f"s3*={s3f:.10f}  s4*={s4f:.10f}  r3*={r3f:.10f}  r4*={r4f:.10f}")

        r3_tr = relaxation_vers_point_fixe(M, r_autres_init, delta, r3f, pas=3000)
        print(f"  r3 mesure apres 3000 pas d'entrainement : {r3_tr[-1]:.10f}  "
              f"(ecart au point fixe : {abs(r3_tr[-1]-r3f):.2e})")

        for lo, hi in [(1, 9), (16, 34), (50, 100), (100, 200), (200, 400)]:
            p = pente_vs_cible_fixe(r3_tr, r3f, lo, hi)
            print(f"  pente vs point fixe FIXE, fenetre [{lo},{hi}) : {p}")

        lo_adapt, hi_adapt = fenetre_adaptative(r3_tr, r3f)
        p_adapt = pente_vs_cible_fixe(r3_tr, r3f, lo_adapt, hi_adapt)
        print(f"  FENETRE ADAPTATIVE [{lo_adapt},{hi_adapt}) (apres dernier extremum, avant plancher de bruit 1e-6) : pente={p_adapt}")
        resultats[label] = (r3_tr, p_adapt)
        print()

    print("=== comparaison des pentes en fenetre adaptative ===")
    pentes = [v[1] for v in resultats.values()]
    if all(p is not None for p in pentes) and pentes[1] != 0:
        print(f"  ratio (M=0/M=25) = {pentes[0]/pentes[1]:.4f}")
