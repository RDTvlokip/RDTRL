"""Suite de verifier_localisation_falaise_k10_k14.py : un agent-dipankar
a trouve que le K=13 rapporte "plateau" (premier=300) etait en fait un
artefact -- la bissection de delta_c(K=13) a tol=1e-6 n'est PAS
reproductible selon le bracket de depart, et cette petite difference
suffit a faire basculer la classification plateau/plat.

Rejoue les deux bissections (bracket large [0.0130,0.0200], bracket
etroit +-3e-5 autour du resultat large, tol=1e-7 pour le second) et
la classification qui en decoule.

Resultat, verifie independamment bit pour bit avec l'agent :
  bracket large  : delta_c=0.014388123  premier(0.03%)=300
  bracket etroit : delta_c=0.014387859  premier(0.03%)=60

Deux bissections du meme seuil, tolerances nominales comparables,
brackets differents -- ne convergent PAS a la meme 6e decimale, et
cette difference (~2.6e-7 sur delta_c) suffit a inverser le verdict
plateau/plat. Ce n'est pas du bruit de calcul (determinisme deja
verifie ailleurs), c'est un manque de precision de bissecter_delta_c
a tol=1e-6 pour CE K precis. Le verdict correct (a precision
suffisante) est PLAT, pas plateau -- la falaise deja localisee entre
K=10 et K=14 se resserre donc entre K=12 et K=13.
"""

import sys
import time
sys.path.insert(0, '.')
import torch

from verifier_jouet_k_emetteur_variable import construire_toy, objectif_toy, BETA, N, bissecter_delta_c

K = 13


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
    t0 = time.time()
    dc_large = bissecter_delta_c(K, r_init=0.5, lo=0.0130, hi=0.0200, tol=1e-6, pas=40000)
    print(f"bracket large: delta_c={dc_large:.9f}  time={time.time()-t0:.1f}s")

    t0 = time.time()
    dc_narrow = bissecter_delta_c(K, r_init=0.5, lo=dc_large - 3e-5, hi=dc_large + 3e-5, tol=1e-7, pas=40000)
    print(f"bracket etroit tol=1e-7: delta_c={dc_narrow:.9f}  time={time.time()-t0:.1f}s")

    for label, dc in [("large", dc_large), ("etroit", dc_narrow)]:
        delta = dc * (1 - 0.0003)
        traj, r4_final = trace(K, delta)
        p = premier_pas_1pct(traj, r4_final)
        print(f"{label}: delta_c={dc:.9f}  premier(0.03%)={p}  r4_final={r4_final:.9f}")


if __name__ == "__main__":
    main()
