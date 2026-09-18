"""Mecanisme final de la piste "masse de fond" (CARNET.md §7.65,
18/09/2026, Theo : « continue, je n'aime pas rester sans reponse »).

Trace dense s3(t)/r4(t) pour M=0 et M=25, chacun a son propre
delta_c(M) (deja etabli dans verifier_ode_jouet_m.py -- pli
algebriquement M-independant). Montre que M=0 se fixe sur la branche
graduee stable et y reste, tandis que M=25 traverse la meme region
puis tombe et se fixe a s3=0,5.

Verifie ensuite que s3=0,5 (M=25) n'est PAS un attracteur separe : la
fonction de branche x_br (deja validee en 3a) suit s3 mesure a chaque
instant, et s3=0,5=x_br(R=1) exactement -- le recepteur (r4) file
jusqu'a R=1 (saturation totale) au lieu de se stabiliser sur la valeur
intermediaire de la branche graduee comme le fait M=0.

Complete par une verification structurelle : les derivees de branche
(x_br'(R), R_br'(x;M)) a la valeur R de la branche STABLE (pas
seulement au pli) sont identiques a 15 chiffres pour tout M -- la
reduction (x,R) est donc M-independante PARTOUT, pas seulement au
pli, ce qui montre que le canal manquant ne peut PAS etre dans cette
reduction a 2 variables, quel que soit le point sonde.
"""

import sys
sys.path.insert(0, '.')
import torch

from verifier_jouet_n_variable import construire_toy_m, objectif_toy_m, N
from verifier_ode_jouet_m import x_br, R_br, localiser_pli

DELTA_C_M0 = 0.01867676
DELTA_C_M25 = 0.01855957


def trace_s3_r4(M, delta, lr, r_autres_init, pas, points):
    poids3 = torch.tensor((1.0 - delta) / N, dtype=torch.float64)
    poids4 = torch.tensor((1.0 + delta) / N, dtype=torch.float64)
    p3, p4, q = construire_toy_m(M, delta, r_autres_init=r_autres_init)
    opt = torch.optim.Adam([p3, p4, q], lr=lr, eps=1e-10)
    out = []
    pset = set(points)
    for i in range(pas):
        j = objectif_toy_m(p3, p4, q, poids3, poids4)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        if (i + 1) in pset:
            with torch.no_grad():
                s3 = torch.sigmoid(p3).item()
                r_all = torch.softmax(q, dim=0)
                r3, r4 = r_all[0].item(), r_all[1].item()
            out.append((i + 1, s3, r3, r4))
    return out


def trouver_racine_stable(M, delta, x_lo=1e-6, x_hi=1e-2, n=2000):
    """Racine STABLE (petite x, cote graduee) de x=x_br(R_br(x;M)),
    par balayage + bissection -- distincte de la racine instable
    (le pli) deja localisee par localiser_pli."""
    def F(x):
        return x_br(R_br(x, M, delta), delta) - x
    xs = [x_lo + i * (x_hi - x_lo) / n for i in range(n + 1)]
    prec = None
    for x in xs:
        f = F(x)
        if prec is not None and (f > 0) != (prec[1] > 0):
            lo, hi = prec[0], x
            for _ in range(60):
                mid = (lo + hi) / 2
                fm = F(mid)
                if (fm > 0) == (prec[1] > 0):
                    lo = mid
                else:
                    hi = mid
            return (lo + hi) / 2
        prec = (x, f)
    return None


def derivees_branche_stable(M, delta):
    x_s = trouver_racine_stable(M, delta)
    R_s = R_br(x_s, M, delta)
    h = 1e-7
    x_br_prime = (x_br(R_s + h, delta) - x_br(R_s - h, delta)) / (2 * h)
    R_br_prime = (R_br(x_s + h, M, delta) - R_br(x_s - h, M, delta)) / (2 * h)
    return x_s, R_s, x_br_prime, R_br_prime


if __name__ == "__main__":
    torch.set_printoptions(precision=8)

    points = [10, 50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 3000,
              5000, 8000, 12000, 20000, 30000, 40000]

    print("=== M=0, delta_c(M=0)=0,01867676, lr=0,2 ===")
    for t, s3, r3, r4 in trace_s3_r4(0, DELTA_C_M0, 0.2, 0.01, 40000, points):
        print(f"  t={t:6d}  s3={s3:.8f}  r3={r3:.8f}  r4={r4:.8f}")

    print()
    print("=== M=25, delta_c(M=25)=0,01855957, lr=0,2 ===")
    for t, s3, r3, r4 in trace_s3_r4(25, DELTA_C_M25, 0.2, 0.01, 40000, points):
        print(f"  t={t:6d}  s3={s3:.8f}  r3={r3:.8f}  r4={r4:.8f}  "
              f"x_br(r4)={x_br(r4, DELTA_C_M25):.8f}")

    print()
    print("=== derivees de branche a la racine STABLE (pas le pli), M=0..25 ===")
    for M in (0, 1, 3, 8, 25):
        delta_c, _ = localiser_pli(M)
        delta_test = delta_c * 0.95
        x_s, R_s, xp, Rp = derivees_branche_stable(M, delta_test)
        print(f"  M={M:2d}  x_s={x_s:.8e}  R_s={R_s:.10f}  s3_s={1-x_s:.8f}  "
              f"x_br_prime={xp:.6e}  R_br_prime={Rp:.6e}  produit={xp*Rp:.6e}")
