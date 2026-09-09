"""Recette exacte pour reproduire la config "referent 3 vs referent 4,
message 10" (l'egalite du tour 7.60 / le k_3 invalide du tour 7.59).

Retrouvee le 09/09/2026 en grepant le transcript JSONL de session (meme
methode que replay_idx5.py) : graine maitresse `default_rng(77777)`,
3 paires sautees, checkpoint a 10000 pas. La perturbation qui a produit
l'egalite pousse `e.p[0][4, 10]` (referent 4, message 10) de +30, puis
continue l'entrainement sous un `adam_eps` reduit (le pas par defaut
1e-8 gele la cellule).

Contexte : c'est une graine et un generateur DIFFERENTS de idx5
(`replay_idx5.py`, graine 999) -- ne pas confondre les deux murs.
"""

import sys
sys.path.insert(0, '.')
import numpy as np
import torch

from representable_atteignable_stable import EmetteurTabulaire, Recepteur, monter, objectif, activer, parametres


BETA = 0.02


def replay(graine_base, k, pas):
    g = np.random.default_rng(graine_base)
    for _ in range(k):
        EmetteurTabulaire(g)
        Recepteur(g)
    e, r = EmetteurTabulaire(g), Recepteur(g)
    monter(e, r, BETA, pas, lr=0.05)
    return e, r


def monter_avec_eps_adam(e, r, beta, pas, lr, adam_eps):
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=lr, eps=adam_eps)
    for _ in range(pas):
        j, _ = objectif(e, r, beta)
        opt.zero_grad()
        (-j).backward()
        opt.step()


def construire_mur23(pas_checkpoint=10000, eps_perturb=30.0, adam_eps=1e-10, pas_suite=40000):
    """Reproduit l'etat "egalite/mur 2" : referent 3 (titulaire) et
    referent 4 (challenger pousse) sur le message 10, graine 77777, k=3."""
    e, r = replay(77777, 3, pas_checkpoint)
    with torch.no_grad():
        e.p[0][4, 10] += eps_perturb
    monter_avec_eps_adam(e, r, BETA, pas_suite, 0.05, adam_eps)
    return e, r


if __name__ == "__main__":
    torch.set_printoptions(precision=15)
    print("Referent 3 vs referent 4, message 10, checkpoint 10k, graine 77777 k=3")
    for adam_eps in (1e-8, 1e-10, 1e-12, 1e-14):
        e, r = construire_mur23(adam_eps=adam_eps)
        with torch.no_grad():
            s_, r_ = e.loi(), r.loi()
            R10_4 = (s_[4, 10] * r_[10, 4]).item()
            un_moins_s4 = 1.0 - s_[4, 10].item()
            un_moins_s3 = 1.0 - s_[3, 10].item()
            print(f"  adam_eps={adam_eps:.0e}  R[10,4]={R10_4:.6f}  "
                  f"1-s[4,10]={un_moins_s4:.6e}  1-s[3,10]={un_moins_s3:.6e}")
