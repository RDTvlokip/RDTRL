"""Correction et approfondissement d'un test ad hoc (python -c, jamais
sauve -- erreur en soi, corrigee ici) sur le mecanisme transitoire de
la masse de fond (piste 3 de ETAT.md). Un agent-dipankar (17/09/2026) a
trouve deux problemes reels dans le test original :

1. Bug d'indexage : `if i % check_tous == 0` s'executait APRES
   `opt.step()`, donc le "t=0" enregistre etait deja apres UN pas
   d'Adam complet (avec lr=0,2, ce premier pas est quasi un pas plein
   en espace de logit, donc ce n'est pas cosmetique).
2. La trajectoire de r3 n'est PAS monotone -- elle depasse largement
   sa valeur finale avant de revenir (M=0 : minimum a 0,0527, 62% sous
   la valeur finale 0,1384). `t90%` (premier croisement du seuil) mesure
   donc le FRONT d'un depassement, pas une convergence reelle.

Mecanisme identifie par l'agent, verifie ici : la masse de fond
s'EVACUE geometriquement vite (ratio ~0,71-0,74/pas, demi-vie ~2 pas).
PENDANT cette evacuation, r3 ET r4 montent ENSEMBLE (ils sont tous les
deux en concurrence avec le fond a recompense nulle, pas encore l'un
contre l'autre) -- ce n'est qu'une fois le fond evacue que
l'asymetrie poids3<poids4 prend le relais et tire r3 vers le bas.

Ce script mesure separement : (a) le delai d'apparition avant que le
fond soit evacue (onset), (b) le taux de decroissance POST-evacuation
de r3 vers sa valeur finale (le vrai analogue de k), pour comparer
CE taux (pas le ratio brut de t90%) au k in[1,42 ; 2,45] du tour 52.
"""

import sys
sys.path.insert(0, '.')
import math
import torch

from verifier_jouet_n_variable import construire_toy_m, objectif_toy_m, N


def trace_complet(M, r_autres_init, delta, pas, lr=0.2, adam_eps=1e-10, r_tie=0.5):
    """Enregistre AVANT le premier pas (t=0 reel), puis a chaque pas."""
    poids3 = torch.tensor((1.0 - delta) / N, dtype=torch.float64)
    poids4 = torch.tensor((1.0 + delta) / N, dtype=torch.float64)
    p3, p4, q = construire_toy_m(M, delta, r_tie=r_tie, r_autres_init=r_autres_init)
    opt = torch.optim.Adam([p3, p4, q], lr=lr, eps=adam_eps)

    def etat():
        with torch.no_grad():
            s3 = torch.sigmoid(p3).item()
            r_all = torch.softmax(q, dim=0)
            r3, r4 = r_all[0].item(), r_all[1].item()
            masse_fond = 1.0 - r3 - r4
        return s3, r3, r4, masse_fond

    s3_tr, r3_tr, r4_tr, fond_tr = [], [], [], []
    s3_0, r3_0, r4_0, fond_0 = etat()
    s3_tr.append(s3_0); r3_tr.append(r3_0); r4_tr.append(r4_0); fond_tr.append(fond_0)
    for i in range(pas):
        j = objectif_toy_m(p3, p4, q, poids3, poids4)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        s3, r3, r4, masse_fond = etat()
        s3_tr.append(s3); r3_tr.append(r3); r4_tr.append(r4); fond_tr.append(masse_fond)
    return s3_tr, r3_tr, r4_tr, fond_tr


def fin_evacuation(fond_tr, seuil_relatif=0.01):
    """Premier pas ou la masse de fond tombe sous seuil_relatif * masse initiale."""
    fond_0 = fond_tr[0]
    if fond_0 <= 0:
        return 0
    for t, f in enumerate(fond_tr):
        if f <= seuil_relatif * fond_0:
            return t
    return None


def taux_post_evacuation(r3_tr, t_debut, t_fin, r3_final):
    """Pente de log|r3-r3_final| entre t_debut et t_fin (regression lineaire simple)."""
    xs, ys = [], []
    for t in range(t_debut, min(t_fin, len(r3_tr))):
        ecart = abs(r3_tr[t] - r3_final)
        if ecart > 1e-12:
            xs.append(t)
            ys.append(math.log(ecart))
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx  # pente (negative = decroissance exponentielle, taux = -pente)


if __name__ == "__main__":
    torch.set_printoptions(precision=6)

    configs = [
        ("M=0 (sans fond)", 0, 1e-6, 0.95 * 0.018672, 500),
        ("M=25, masse totale=25%", 25, 0.01, 0.95 * 0.018516, 500),
    ]

    resultats = {}
    for label, M, r_autres_init, delta, pas in configs:
        s3_tr, r3_tr, r4_tr, fond_tr = trace_complet(M, r_autres_init, delta, pas)
        r3_final = r3_tr[-1]
        t_evac = fin_evacuation(fond_tr)
        print(f"=== {label} (delta={delta:.6f}) ===")
        print(f"  t=0 (reel, avant tout pas)  : s3={s3_tr[0]:.6f}  r3={r3_tr[0]:.6f}  "
              f"r4={r4_tr[0]:.6f}  masse_fond={fond_tr[0]:.6f}")
        print(f"  r3_final={r3_final:.6f}")
        print(f"  fin d'evacuation (masse_fond < 1% de l'initiale) : pas={t_evac}")
        r3_min_idx = min(range(len(r3_tr)), key=lambda t: r3_tr[t])
        print(f"  minimum de r3 sur la trace : {r3_tr[r3_min_idx]:.6f} au pas {r3_min_idx} "
              f"(final={r3_final:.6f}, {'EN DESSOUS' if r3_tr[r3_min_idx] < r3_final else 'au-dessus'} de la valeur finale -> non-monotone)")

        if t_evac is not None:
            fenetre_fin = min(t_evac + 20, pas)
            pente = taux_post_evacuation(r3_tr, t_evac, fenetre_fin, r3_final)
        else:
            pente = taux_post_evacuation(r3_tr, 1, 10, r3_final)
        print(f"  pente post-evacuation (log|r3-r3_final| vs pas, fenetre [{t_evac},{fenetre_fin if t_evac is not None else '?'}]) : {pente}")
        resultats[label] = (t_evac, pente)
        print()

    print("=== comparaison des taux post-evacuation ===")
    pentes = [v[1] for v in resultats.values()]
    if all(p is not None for p in pentes) and pentes[1] != 0:
        ratio_taux = pentes[0] / pentes[1]
        print(f"  ratio des pentes (M=0 / M=25) = {ratio_taux:.4f}")
        print(f"  (a comparer a k in [1,42 ; 2,45] du tour 52 sur le systeme complet)")
