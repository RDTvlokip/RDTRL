"""Locate the K boundary between 'flat' (K=20,26) and K=1's strong
critical slowing. K=8 shows a weak signal only at 0.03% below its own
delta_c (320 vs 60/60 baseline). Test K=2,4,6 with the same exact
protocol (three relative distances, check_tous=20, PAS_MAX=20000) to
see where the effect appears/disappears.

delta_c values must be found first for each K via bisection (same
seuil formula as verifier_jouet_k_emetteur_variable.bissecter_delta_c),
matching the precision used for K=1/8/20/26 (~1e-5, pas=40000).
"""
import sys
import time
sys.path.insert(0, 'src/test3_communication')
import torch

from verifier_jouet_k_emetteur_variable import (
    construire_toy, objectif_toy, BETA, N, bissecter_delta_c
)

PAS_MAX = 20000
CHECK_TOUS = 20
DISTANCES_RELATIVES = (0.03, 0.003, 0.0003)


def trace(K, delta, check_tous=CHECK_TOUS, pas_max=PAS_MAX):
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
    for K in (2, 4, 6):
        t0 = time.time()
        # known brackets: delta_c(K=1)=0.018699, delta_c(K=8)=0.015098
        # monotone decreasing in K (26->0.013438, 20->0.01378, 8->0.015098, 1->0.018699)
        delta_c = bissecter_delta_c(K, r_init=0.5, lo=0.0130, hi=0.0200, tol=1e-6, pas=40000)
        print(f"=== K={K}  delta_c={delta_c:.6f}  (bisected, {time.time()-t0:.1f}s) ===", flush=True)
        for frac in DISTANCES_RELATIVES:
            delta = delta_c * (1 - frac / 100.0)
            t1 = time.time()
            traj, r4_final = trace(K, delta)
            p = premier_pas_1pct(traj, r4_final)
            print(f"  frac={frac}% delta={delta:.7f}  r4_final={r4_final:.6f}  premier={p}  "
                  f"time={time.time()-t1:.1f}s", flush=True)


if __name__ == "__main__":
    main()
