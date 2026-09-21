"""Resout la discordance entre ma localisation "coin net [12.80;12.95]"
(verifier_localisation_fine_k_non_entier.py) et la contestation d'un
agent-dipankar (qui affirmait que K=12.80 bascule sous bracket de
bissection etroit, verifier_k1280_discordance_agent.py -- discordance
initialement non resolue).

Diagnostic : rebissecte delta_c(K=12.80) en bracket etroit en gardant
la PLEINE precision (pas la valeur affichee a 9 decimales), puis
teste la classification premier(delta) en tronquant cette valeur a
des precisions croissantes (7 a 14 decimales) plus la pleine
precision elle-meme.

Resultat, decisif : la classification REBONDIT entre 60 et 300 de
facon NON MONOTONE meme pour des perturbations de delta_c de l'ordre
de 1e-15 (quinze ordres de grandeur sous delta_c lui-meme). Aucune
precision testee ne stabilise le resultat -- ce n'est pas un probleme
de "pas assez de decimales", c'est une vraie sensibilite chaotique :
a K≈12.80, ecart≈0.03% sous son propre delta_c, la trajectoire est
posee quasi exactement SUR une separatrice de la dynamique continue
sous-jacente. Ni l'agent ni moi n'etions "faux" -- chacun tombait
d'un cote different de cette frontiere sensible selon le chemin
flottant exact de son propre script.

Consequence methodologique : premier_pas_1pct (seuil discret sur une
trajectoire unique) est un mauvais instrument tout pres d'une
separatrice -- la bonne mesure serait un temps de residence continu
ou un taux de relaxation linearise, pas un seuil binaire.
"""

import sys
sys.path.insert(0, '.')
import torch

from verifier_jouet_k_emetteur_variable import construire_toy, objectif_toy, BETA, N, bissecter_delta_c

K = 12.80
DC_WIDE = 0.01440988540649414  # deja etabli, bracket [0.0130, 0.0200]
NDECS = (7, 8, 9, 10, 12, 14)


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


def classer(K, dc):
    delta = dc * (1 - 0.0003)
    traj, r4_final = trace(K, delta)
    return premier_pas_1pct(traj, r4_final)


def main():
    dc_narrow = bissecter_delta_c(K, r_init=0.5, lo=DC_WIDE - 3e-5, hi=DC_WIDE + 3e-5, tol=1e-7, pas=40000)
    print(f"dc_narrow (pleine precision) = {dc_narrow!r}")
    print(f"diff vs dc_wide = {dc_narrow - DC_WIDE:.6e}")

    for ndec in NDECS:
        dc_trunc = round(dc_narrow, ndec)
        diff = dc_trunc - dc_narrow
        p = classer(K, dc_trunc)
        print(f"ndec={ndec:2d}  dc_trunc={dc_trunc!r}  diff={diff:+.3e}  premier={p}")

    p_full = classer(K, dc_narrow)
    print(f"PLEINE PRECISION: premier={p_full}")


if __name__ == "__main__":
    main()
