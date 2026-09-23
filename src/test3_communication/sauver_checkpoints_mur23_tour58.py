"""Tour 58 (23/09/2026) : checkpoint complet (tous les logits de
l'emetteur et du recepteur + etat d'Adam) de la trajectoire mur23 a
pas=54000, aux deux deltas, pour pouvoir rejouer n'importe quelle
fenetre de [54000,62000) en quelques secondes au lieu de reconstruire
110 000 pas a chaque question.

Meme convention de comptage que tracer_mur23_lignes3_10_complet.py :
l'etat sauve est celui APRES le pas numero PAS-1 (PAS pas executes
apres construire_mur23). reprendre() verifie la reproduction bit pour
bit contre la trace deja enregistree.

Usage : python sauver_checkpoints_mur23_tour58.py <delta>
Sortie : D:/tmp/rdtrl_tour58_checkpoint_mur23_g77777_k3_delta<delta>_pas<PAS>.pt
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, replay, BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere

ADAM_EPS = 1e-10
LR = 0.05
PAS = 54000


def poids_pour(delta):
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + delta) / N
    poids[3] = (1.0 - delta) / N
    return poids


def chemin(delta):
    return f"D:/tmp/rdtrl_tour58_checkpoint_mur23_g77777_k3_delta{delta}_pas{PAS}.pt"


def reprendre(delta):
    """Reconstruit (e, r, opt, poids) a l'etat sauve."""
    torch.set_num_threads(1)
    ck = torch.load(chemin(delta))
    e, r = replay(77777, 3, 10)  # structure seulement, valeurs ecrasees ci-dessous
    activer(e, r)
    params = parametres(e, r)
    with torch.no_grad():
        for p, v in zip(params, ck["params"]):
            p.copy_(v)
    opt = torch.optim.Adam(params, lr=LR, eps=ADAM_EPS)
    opt.load_state_dict(ck["opt"])
    return e, r, opt, poids_pour(delta)


if __name__ == "__main__":
    torch.set_num_threads(1)
    delta = float(sys.argv[1])
    poids = poids_pour(delta)
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    params = parametres(e, r)
    opt = torch.optim.Adam(params, lr=LR, eps=ADAM_EPS)
    for pas in range(PAS):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
    torch.save({"params": [p.detach().clone() for p in params],
                "opt": opt.state_dict(), "delta": delta, "pas": PAS}, chemin(delta))
    print(f"ecrit : {chemin(delta)}")
