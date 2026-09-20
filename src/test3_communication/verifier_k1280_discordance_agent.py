"""Suite de verifier_localisation_fine_k_non_entier.py : un
agent-dipankar a conteste le "coin net [12.80;12.95]" en affirmant que
K=12.80 lui-meme bascule de plateau (300) a plat (60) sous un bracket
de bissection etroit -- le meme defaut deja confirme a K=13
(verifier_k13_bissection_instable.py), mais avec une perturbation
encore plus fine.

Rejoue ce test precis pour K=12.80. Resultat : DISCORDANCE NON
RESOLUE. Le delta_c du bracket etroit obtenu ici (0.014409856)
correspond EXACTEMENT a celui rapporte par l'agent, au chiffre pres --
mais premier(0.03%) donne 300 (plateau) dans ce script, pas 60 (plat)
comme l'agent l'affirme, pour la MEME paire (K, delta_c) sur un
systeme entierement deterministe.

Statut : INDETERMINE. Le defaut de reproductibilite de la bissection
a K=13 reste confirme (deux verifications independantes en accord).
Ce qui est remis en question, c'est seulement l'extension de ce
defaut a K=12.80 -- affirmee par l'agent, non reproduite ici. Voir
CARNET.md, section jouet a K variable, pour le detail complet et la
discussion de cette discordance.
"""

import sys
import time
sys.path.insert(0, '.')
import torch

from verifier_jouet_k_emetteur_variable import construire_toy, objectif_toy, BETA, N, bissecter_delta_c

K = 12.80


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
    dc_wide = bissecter_delta_c(K, r_init=0.5, lo=0.0130, hi=0.0200, tol=1e-7, pas=40000)
    print(f"wide: delta_c={dc_wide:.9f}  time={time.time()-t0:.1f}s")

    t0 = time.time()
    dc_narrow = bissecter_delta_c(K, r_init=0.5, lo=dc_wide - 3e-5, hi=dc_wide + 3e-5, tol=1e-7, pas=40000)
    print(f"narrow: delta_c={dc_narrow:.9f}  time={time.time()-t0:.1f}s")

    for label, dc in [("wide", dc_wide), ("narrow", dc_narrow)]:
        delta = dc * (1 - 0.0003)
        traj, r4_final = trace(K, delta)
        p = premier_pas_1pct(traj, r4_final)
        print(f"{label}: delta_c={dc:.9f}  premier(0.03%)={p}")


if __name__ == "__main__":
    main()
