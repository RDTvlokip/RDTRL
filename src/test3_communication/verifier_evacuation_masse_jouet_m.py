"""Challenge agent-dipankar (18/09/2026, tour post-3b) : verification
independante de trois points avant d'accepter la lecture "evacuation de
masse" comme mecanisme des 6,5 decades d'ecart M=0 vs M=25.

1. Tracer masse_autres(t) = somme des M logits de fond au cours de
   l'entrainement (M=25, r_autres_init=0,01 -- config EXACTE de
   verifier_point_fixe_jouet_m.py) : combien de pas pour aller de 25%
   a ~4e-11 ? Si c'est trivialement rapide (dizaines de pas), le
   mecanisme "evacuation lente" est refute avant meme de creuser plus.

2. Convergence numerique de a=F''(x*)/2 (differences finies centrees)
   a plusieurs h : h=1e-4, 1e-5, 1e-6, 1e-7, 1e-8 -- verifie si le
   chiffre a~99.06 publie est stable ou artefact de troncature/arrondi.

3. Recalcul independant de delta_c(M) par bissection brute (PAS Newton,
   PAS le meme code que localiser_pli) pour M=0 et M=25, pour confirmer
   l'identite a 1e-21 annoncee sans reutiliser le meme algorithme.
"""

import sys
sys.path.insert(0, '.')
import math
import torch

from verifier_jouet_n_variable import construire_toy_m, objectif_toy_m, N
from verifier_ode_jouet_m import x_br, R_br, localiser_pli, BETA


def tracer_masse_autres(M, r_autres_init, delta, pas=3000, lr=0.2, adam_eps=1e-10):
    poids3 = torch.tensor((1.0 - delta) / N, dtype=torch.float64)
    poids4 = torch.tensor((1.0 + delta) / N, dtype=torch.float64)
    p3, p4, q = construire_toy_m(M, delta, r_autres_init=r_autres_init)
    opt = torch.optim.Adam([p3, p4, q], lr=lr, eps=adam_eps)
    traj = []
    with torch.no_grad():
        r_all0 = torch.softmax(q, dim=0)
        traj.append(r_all0[2:].sum().item())
    for i in range(pas):
        j = objectif_toy_m(p3, p4, q, poids3, poids4)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        with torch.no_grad():
            r_all = torch.softmax(q, dim=0)
            traj.append(r_all[2:].sum().item())
    return traj


def courbure_a(M, delta_c, x_star, hs):
    def F(x):
        return x_br(R_br(x, M, delta_c), delta_c) - x
    out = {}
    for h in hs:
        d2 = (F(x_star + h) - 2 * F(x_star) + F(x_star - h)) / (h * h)
        out[h] = d2 / 2
    return out


def bissection_delta_c(M, lo=0.0180, hi=0.0195, tol=1e-16, x0_guess=0.0042146):
    """Bissecte sur delta le NOMBRE de racines de F(x,delta)=0 (3 racines
    si delta<delta_c stable/instable/stable ; 1 si delta>delta_c) --
    algorithme independant de localiser_pli (pas de Newton sur le
    systeme de tangence, juste comptage de racines par balayage fin)."""
    def nb_racines(delta):
        n = 0
        prec = None
        for i in range(1, 400_000):
            x = i / 400_000
            f = x_br(R_br(x, M, delta), delta) - x
            if prec is not None and (f > 0) != (prec > 0):
                n += 1
            prec = f
        return n

    for _ in range(80):
        mid = (lo + hi) / 2
        n = nb_racines(mid)
        if n >= 3:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-17:
            break
    return (lo + hi) / 2


if __name__ == "__main__":
    print("=== 1. Trajectoire masse_autres(t), M=25, r_autres_init=0.01, delta=0.95*0.018516 ===")
    delta = 0.95 * 0.018516
    traj = tracer_masse_autres(25, 0.01, delta, pas=3000)
    print(f"  t=0     masse_autres = {traj[0]:.6e}")
    for t_cible in (1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000, 2000, 3000):
        if t_cible < len(traj):
            print(f"  t={t_cible:5d}  masse_autres = {traj[t_cible]:.6e}")
    # premier pas ou masse_autres < 1e-6, < 1e-9, < 4.09e-11*10
    for seuil in (1e-3, 1e-6, 1e-9, 1e-10):
        for t, v in enumerate(traj):
            if v < seuil:
                print(f"  premier t avec masse_autres < {seuil:.0e} : t={t}")
                break
        else:
            print(f"  masse_autres n'atteint jamais < {seuil:.0e} en {len(traj)-1} pas (derniere valeur {traj[-1]:.3e})")

    print()
    print("=== 2. Sensibilite de a=F''(x*)/2 a h (differences finies centrees) ===")
    out = localiser_pli(0)
    if out is not None:
        dc0, xstar0 = out
        print(f"  delta_c(M=0)={dc0:.15f}  x*={xstar0:.12e}")
        for h in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 3e-9):
            a_vals = courbure_a(0, dc0, xstar0, [h])
            print(f"    h={h:.0e}  a={a_vals[h]:.6f}")
    out25 = localiser_pli(25)
    if out25 is not None:
        dc25, xstar25 = out25
        print(f"  delta_c(M=25)={dc25:.15f}  x*={xstar25:.12e}")
        for h in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 3e-9):
            a_vals = courbure_a(25, dc25, xstar25, [h])
            print(f"    h={h:.0e}  a={a_vals[h]:.6f}")

    print()
    print("=== 3. delta_c(M) par bissection independante (comptage de racines) ===")
    for M in (0, 25):
        dc_bis = bissection_delta_c(M)
        print(f"  M={M:2d}  delta_c (bissection racines) = {dc_bis:.15f}")
