"""Suite de verifier_k13_bissection_instable.py : la falaise
plateau/plat du jouet a K variable (voir CARNET.md, section jouet a K
variable) a ete resserree entre K=12 (plateau) et K=13 (plat, une fois
la bissection corrigee). Ce script teste des valeurs de K NON
ENTIERES entre 12 et 13 pour localiser plus finement, et repondre a
la question ouverte : coin net ou transition lisse ?

Le terme d'entropie entropie_k(p,K) n'utilise K que via un log(K)
simple (pas d'operation combinatoire/factorielle), donc le jouet est
mathematiquement bien defini pour K non entier -- pas d'extrapolation
artificielle.

Resultat (chaque point bissecte a tol=1e-7, ~12 min/point) :
  K=12    : premier=300  (plateau, deja etabli)
  K=12.5  : premier=300  (plateau)
  K=12.8  : premier=300  (plateau)
  K=12.95 : premier=60   (PLAT)
  K=13    : premier=60   (plat, une fois corrige)

La transition est localisee entre K=12.8 et K=12.95 -- une fenetre de
seulement 0.15 en K, sur un intervalle total teste de 1.0 (K=12 a
K=13). C'est un signal NET, pas une pente lisse etalee sur tout
l'intervalle -- coherent avec une vraie transition serree plutot
qu'une decroissance graduelle continue.
"""

import sys
import time
sys.path.insert(0, '.')
import torch

from verifier_jouet_k_emetteur_variable import construire_toy, objectif_toy, BETA, N, bissecter_delta_c

POINTS_K = (12.5, 12.8, 12.95)


def trace(K, delta, check_tous=20, pas_max=20000):
    poids3 = torch.tensor((1.0 - delta) / N, dtype=torch.float64)
    poids4 = torch.tensor((1.0 + delta) / N, dtype=torch.float64)
    p3, p4, q = construire_toy(K, s3_init=0.999, s4_init=0.999, r_init=0.5)
    opt = torch.optim.Adam([p3, p4, q], lr=0.05, eps=1e-10)
    trajectoire = []
    for pas in range(pas_max):
        j = objectif_toy(p3, p4, q, poids3, poids4, K)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        if pas % check_tous == 0:
            with torch.no_grad():
                r4 = torch.sigmoid(q).item()
            trajectoire.append((pas, r4))
    with torch.no_grad():
        r4_final = torch.sigmoid(q).item()
    return trajectoire, r4_final


def premier_pas_1pct(traj, r4_final):
    cible = 0.99 * r4_final if r4_final < 1.0 else 0.99
    for pas, r4 in traj:
        if r4 >= cible:
            return pas
    return None


def main():
    for K in POINTS_K:
        t0 = time.time()
        dc = bissecter_delta_c(K, r_init=0.5, lo=0.0130, hi=0.0200, tol=1e-7, pas=40000)
        delta = dc * (1 - 0.0003)
        traj, r4_final = trace(K, delta)
        p = premier_pas_1pct(traj, r4_final)
        print(f"K={K}  delta_c={dc:.9f}  premier(0.03%)={p}  r4_final={r4_final:.9f}  "
              f"time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
