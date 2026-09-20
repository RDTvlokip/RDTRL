"""Does the K-variable toy (K=26) show critical slowing near its OWN
delta_c, the way verifier_ralentissement_critique.py tested (and found
FLAT, not diverging) for the real 27-referent system? My independent
mean-field/equilibrium analysis of this exact toy found det(J) -> 0
smoothly approaching delta_c=0.01344 (a genuine fold in the reduced
3-variable equilibrium equations), which -- if it governs the Adam
dynamics -- predicts convergence time ~ (delta_c-delta)^-1/2, diverging.
This is the decisive test: if the toy is flat too, the equilibrium fold
does not drive the toy's dynamical threshold either (consistent with
the real system's boundary-crisis reading) and the near-exact numeric
match between fold and delta_c is more coincidental than mechanistic.
If the toy DOES slow down critically, the toy's mechanism differs from
the real system's despite matching delta_c almost exactly.
"""
import sys
sys.path.insert(0, 'src/test3_communication')
import torch
from verifier_jouet_k_emetteur_variable import construire_toy, objectif_toy, BETA, N

K = 26
DELTA_C = 0.013438
PAS_MAX = 20000
CHECK_TOUS = 20


def trace(delta, pas_max=PAS_MAX):
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
        if pas % CHECK_TOUS == 0:
            with torch.no_grad():
                r4 = torch.sigmoid(q).item()
            trajectoire.append((pas, r4))
    with torch.no_grad():
        r4_final = torch.sigmoid(q).item()
    return trajectoire, r4_final


if __name__ == "__main__":
    distances_relatives = (0.03, 0.003, 0.0003)
    print(f"=== ralentissement critique, toy K=26, delta_c ~ {DELTA_C} (variable tracee: r4) ===")
    for frac in distances_relatives:
        delta = DELTA_C * (1 - frac)
        traj, r4_final = trace(delta)
        cible = 0.99 * r4_final if r4_final < 1.0 else 0.99
        premier_pas = None
        for pas, r4 in traj:
            if r4 >= cible:
                premier_pas = pas
                break
        print(f"  delta={delta:.7f}  ({frac*100:.3f}% sous delta_c)  "
              f"r4_final={r4_final:.6f}  premier pas a 1% de la cible = {premier_pas}")
