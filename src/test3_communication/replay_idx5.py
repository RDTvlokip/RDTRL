"""Recette exacte pour reproduire idx5 (mur 1, "referent 18 / message 0").

Retrouvee le 08/09/2026 en grepant le transcript JSONL de session apres
avoir perdu la reconstruction (CARNET.md 7.60bis) : la graine maitresse
est `np.random.default_rng(999)`, on saute 5 paires emetteur/recepteur
avant de prendre la sixieme. Ce n'est PAS le meme generateur que
`recensement_egalite_mur.py` (qui utilise 31415) -- idx5 vient d'une
exploration ad hoc separee, jamais sauvegardee en fichier avant ce jour.

A checkpoint 40000 pas (lr=0.05 par defaut, monter() cree son propre
Adam a chaque appel) :
    S[18,0] = 0.499479   S[18,8] = 0.500521   (referent 18, scission libre)
    S[0].max() = 0.037113, argmax=7            (referent 0, mur/jamais engage)
Valeurs verifiees identiques a celles publiees dans docs/CARNET.md 7.48-7.49.

Le "mur" oppose le referent 0 (jamais commis, ligne quasi uniforme) au
message 0, contre le referent 18 qui envoie ce message la moitie du
temps sans que ca lui coute rien (scission gratuite avec le message 8,
tous deux non contestes par ailleurs). Pousser `e.p[0][0, 0]` (referent 0,
message 0) au dela d'un seuil entre eps=23 et eps=24 fait basculer le
referent 0 en capture complete du message 0, forcant le referent 18 a se
retirer proprement vers le message 8 seul.
"""

import sys
sys.path.insert(0, '.')
import numpy as np
import torch

from representable_atteignable_stable import EmetteurTabulaire, Recepteur, monter, objectif, activer, parametres


BETA = 0.02


def replay_idx5(pas, lr=0.05):
    """Reconstruit l'etat de idx5 apres `pas` pas de montee exacte."""
    g = np.random.default_rng(999)
    for _ in range(5):
        EmetteurTabulaire(g)
        Recepteur(g)
    e, r = EmetteurTabulaire(g), Recepteur(g)
    monter(e, r, BETA, pas, lr=lr)
    return e, r


def monter_avec_eps_adam(e, r, beta, pas, lr, adam_eps):
    """Comme monter(), mais avec un epsilon Adam explicite (pour degeler
    une cellule dont le pas par defaut (1e-8) reste sous le plancher)."""
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=lr, eps=adam_eps)
    for _ in range(pas):
        j, _ = objectif(e, r, beta)
        opt.zero_grad()
        (-j).backward()
        opt.step()


if __name__ == "__main__":
    e, r = replay_idx5(40000)
    with torch.no_grad():
        s_ = e.loi()
        print("verification :")
        print(f"  S[18,0]={s_[18,0].item():.6f}  S[18,8]={s_[18,8].item():.6f}")
        print(f"  S[0].max()={s_[0].max().item():.6f}  argmax={s_[0].argmax().item()}")
