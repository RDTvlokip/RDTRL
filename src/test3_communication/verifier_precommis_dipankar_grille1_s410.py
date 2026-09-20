"""Tour 54 (20/09/2026, vraie critique de dipankarsarkar) : test 1 des
trois tests precommis par lui. Rejoue le vrai systeme (pas l'hybride)
a grille 1 (aucun sous-echantillonnage) sur une fenetre de 2000 pas
centree sur pas=60000, pour verifier si 1-s[4,10] depasse 5e-12
(seuil qu'il a lui-meme fixe pour rouvrir sa defense
"mauvais instant d'echantillonnage").

Resultat : 1-s[4,10] reste dans [4.663e-15, 4.885e-15] sur toute la
fenetre, y compris exactement au pic du kick -- trois ordres de
grandeur SOUS son seuil de reouverture. Sa defense est definitivement
close, pas juste affaiblie.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere

DELTA = 0.013026615
LR = 0.05
ADAM_EPS = 1e-10
PAS_MAX = 61000
DEBUT = 59000


def main():
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + DELTA) / N
    poids[3] = (1.0 - DELTA) / N

    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=LR, eps=ADAM_EPS)

    p_e = e.p[0]
    s410_min, s410_max = 1.0, 0.0
    for pas in range(PAS_MAX):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        if pas >= DEBUT:
            with torch.no_grad():
                s410 = torch.sigmoid(p_e[4, 10]).item()
            s410_min = min(s410_min, s410)
            s410_max = max(s410_max, s410)

    print(f"s[4,10] range: [{s410_min}, {s410_max}]")
    print(f"1-s[4,10] range: [{1-s410_max:.3e}, {1-s410_min:.3e}]")


if __name__ == "__main__":
    main()
