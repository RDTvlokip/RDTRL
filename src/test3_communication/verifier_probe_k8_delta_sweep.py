"""Probe: is the K=8 '320 pas' signal at 0.03% below delta_c real
critical slowing, or a knife-edge/discretization artifact? Rerun exact
protocol, then sweep delta finely, then use a finer time grid, then
dump the raw trajectory near the threshold-crossing to see its shape
(monotonic approach vs oscillatory kick)."""
import sys
import time
sys.path.insert(0, 'src/test3_communication')
import torch

from verifier_jouet_k_emetteur_variable import construire_toy, objectif_toy, BETA, N

PAS_MAX = 20000


def trace(K, delta, check_tous, pas_max=PAS_MAX):
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
            return pas, cible
    return None, cible


def main():
    K = 8
    delta_c = 0.015098

    print("=== step 0: exact reproduction (determinism check) ===")
    for rep in range(2):
        traj, r4_final = trace(K, delta_c * (1 - 0.0003), check_tous=20)
        p, cible = premier_pas_1pct(traj, r4_final)
        print(f"  rep={rep}  r4_final={r4_final!r}  premier={p}  cible={cible!r}")

    print()
    print("=== step 1: fine delta sweep around the 0.03% point (K=8) ===")
    for frac in (0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.2, 0.3):
        delta = delta_c * (1 - frac / 100.0)
        t0 = time.time()
        traj, r4_final = trace(K, delta, check_tous=20)
        p, cible = premier_pas_1pct(traj, r4_final)
        print(f"  frac={frac:.4f}%  delta={delta:.8f}  r4_final={r4_final:.8f}  premier={p}  "
              f"time={time.time()-t0:.1f}s")

    print()
    print("=== step 2: finer time grid (check_tous=1) at the 0.03% point ===")
    delta = delta_c * (1 - 0.0003)
    traj, r4_final = trace(K, delta, check_tous=1)
    p, cible = premier_pas_1pct(traj, r4_final)
    print(f"  r4_final={r4_final:.10f}  cible={cible:.10f}  premier(grille1)={p}")
    # dump trajectory around the crossing region to see its shape
    lo = max(0, (p or 400) - 60)
    hi = (p or 400) + 60
    print("  --- raw trajectory near crossing (grille 1) ---")
    for pas, r4 in traj:
        if lo <= pas <= hi:
            print(f"    pas={pas:6d}  r4={r4:.10f}")


if __name__ == "__main__":
    main()
