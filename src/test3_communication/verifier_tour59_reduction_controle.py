"""Tour 59 (29/09/2026) : V2. Rejoue la reduction a deux lignes du premier
agent style dipankar (reduction_deux_lignes_tour59.py = red2.py, simulateur Adam reduit :
lignes 3 et 4 de l'emetteur = logit de message + 26 concurrents symetriques,
logits l3 et l4 du recepteur, queue de 25 referents gelee) depuis l'etat
chaud 20 000 + 20 000 pas a eps 1e-10 du reseau complet, aux cinq delta ou
il annonce des temps de bascule identiques entre red2 et son port numba :
    1e-8 / 7e-9 / 6,5e-9 / 6e-9 / 5e-9  ->  490 574 593 614 665.
Sort aussi le temps de bascule du RESEAU COMPLET aux memes delta depuis le
meme etat (verifier_tour59_delta_c_dynamique.py, cache chaud identique),
pour juger l'ecart reduction / reseau complet sur une seule realisation.
"""

import sys
sys.path.insert(0, ".")
import numpy as np
import torch

from reduction_deux_lignes_tour59 import charger, simuler, PLI  # copie de red2.py (agent style dipankar)

CHAUD = "D:/tmp/rdtrl_tour59_chaud_mur23_3_4_eps1e-10_20000_20000.pt"
SES_TEMPS = {1e-8: 490, 7e-9: 574, 6.5e-9: 593, 6e-9: 614, 5e-9: 665}


if __name__ == "__main__":
    s0 = charger(CHAUD)
    print(f"etat charge : pas d'Adam = {s0['step']:.0f}")
    print("offset      bascule (red2, ma execution)   ses temps annonces")
    for off, ses in SES_TEMPS.items():
        _, b, _ = simuler(s0, PLI + off, 1e-10, 3000, trace=False)
        print(f"{off:8.1e}    {b:6d}                        {ses}")
