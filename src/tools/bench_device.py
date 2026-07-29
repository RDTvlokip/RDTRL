"""Mesure le temps par episode sur CPU et sur GPU, pour justifier le choix du device.

Le modele est minuscule (GRU 128, lot de 1, 12 pas) : le calcul reel est
negligeable devant le cout fixe de lancement des noyaux CUDA, qui est paye
12 fois par episode a l'aller plus autant au retour de la retropropagation.
"""

import time
import torch

import os
import sys

sys.path.insert(0, os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    "src", "test1_copy"))

from rl_copie import (CIBLE_PRINCIPALE, CIBLE_PERTURBEE, Vocabulaire,
                      construire_vocabulaire, nouvelle_politique,
                      recompense_positions)


def mesurer(device, taille_lot, n_episodes=200):
    vocabulaire = Vocabulaire(construire_vocabulaire([CIBLE_PRINCIPALE, CIBLE_PERTURBEE]))
    cible = vocabulaire.encoder(CIBLE_PRINCIPALE)
    politique = nouvelle_politique(vocabulaire, 0, device)
    optimiseur = torch.optim.Adam(politique.parameters(), lr=1e-3)

    def un_pas():
        actions, log_probs, entropies, _ = politique.generer(
            len(cible), taille_lot=taille_lot, device=device)
        recompenses = [recompense_positions(s, cible) for s in actions.tolist()]
        r = torch.tensor(recompenses, device=device)
        perte = -(log_probs.sum(1) * (r - r.mean()).detach()).mean() \
                - 0.01 * entropies.sum(1).mean()
        optimiseur.zero_grad()
        perte.backward()
        optimiseur.step()

    for _ in range(20):          # chauffe (compilation des noyaux, allocateur)
        un_pas()
    if device == "cuda":
        torch.cuda.synchronize()

    debut = time.perf_counter()
    for _ in range(n_episodes):
        un_pas()
    if device == "cuda":
        torch.cuda.synchronize()
    duree = time.perf_counter() - debut

    # Un "pas" traite taille_lot episodes en parallele
    return duree / n_episodes * 1000, duree / (n_episodes * taille_lot) * 1000


print(f"torch {torch.__version__} | threads CPU = {torch.get_num_threads()}")
if torch.cuda.is_available():
    print(f"GPU : {torch.cuda.get_device_name(0)}")
print()
print(f"{'device':6s} {'lot':>5s} {'ms / mise a jour':>18s} {'ms / episode':>14s}")
for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
    for taille_lot in (1, 32, 256):
        par_pas, par_episode = mesurer(device, taille_lot)
        print(f"{device:6s} {taille_lot:5d} {par_pas:18.2f} {par_episode:14.3f}")
