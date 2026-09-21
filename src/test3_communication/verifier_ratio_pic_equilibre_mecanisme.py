"""SYNTHESE du fil "plateau/falaise/coin" du jouet a K variable (voir
CARNET.md, section finale). Un agent-dipankar a trouve le vrai
mecanisme sous-jacent a TOUT le tableau discret "plateau vs plat" :
chaque trajectoire (tout K confondu) monte vite, depasse un pic vers
t=50-60, redescend legerement, puis remonte lentement vers son
equilibre. Le ratio(K) = r4(pic)/r4_eq croit de facon LISSE et
MONOTONE avec K -- premier_pas_1pct ne mesure pas un taux de
relaxation, il mesure si le PREMIER pic depasse deja 99% de la valeur
finale. ratio(K) traverse 0.99 quelque part entre K=10 et K=12.5,
exactement la ou premier_pas_1pct s'effondre de ~300 a ~50.

Ce script mesure ratio(K) directement (au lieu du seuil discret) pour
plusieurs K, et diagnostique le cas limite trouve a K=12.5 : le pic
mesure (grille 20) est SOUS la cible de seulement 0.033% relatif --
un cas limite exact, pas une coincidence. C'est pourquoi K=12.5
classifie differemment selon la resolution de grille utilisee pour
trouver le pic (moi: grille 20, premier=300 ; agent: grille plus fine,
premier~54) -- les deux mesures sont correctes sur leurs propres
donnees, le systeme est reellement a cheval sur le seuil a cet endroit.
"""

import sys
sys.path.insert(0, '.')
import torch

from verifier_jouet_k_emetteur_variable import construire_toy, objectif_toy, BETA, N


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


def ratio_pic_equilibre(traj, r4_final, fenetre_pic=(20, 200)):
    a, b = fenetre_pic
    pic = max(r4 for pas, r4 in traj if a <= pas <= b)
    return pic / r4_final


DC = {10: 0.014766, 12.5: 0.014443638, 14: 0.014283029}


def main():
    for K, dc in DC.items():
        delta = dc * (1 - 0.0003)
        traj, r4_final = trace(K, delta)
        ratio = ratio_pic_equilibre(traj, r4_final)
        p = premier_pas_1pct(traj, r4_final)
        cible = 0.99 * r4_final
        pic = max(r4 for pas, r4 in traj if 20 <= pas <= 200)
        marge = pic - cible
        print(f"K={K:5}  ratio(pic/eq)={ratio:.6f}  premier={p}  "
              f"pic={pic:.6f}  cible={cible:.6f}  marge={marge:+.6f}")


if __name__ == "__main__":
    main()
