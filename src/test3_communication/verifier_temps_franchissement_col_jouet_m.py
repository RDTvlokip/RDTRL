"""Mesure du temps passe dans la region lente du VRAI pli (CARNET.md
§7.65, 18/09/2026) pour le jouet a M categories de fond.

Le pli algebrique (verifier_ode_jouet_m.py) est a x*~0,0042146, donc
s3*~0,9958 -- PAS a s3=0,5 comme une premiere tentative (par un
agent-dipankar) l'avait suppose a tort. Ce script mesure combien de
pas la trajectoire reste dans la fenetre [0,990;0,999] autour du vrai
point selle, pour M=0 et M=25, aux deux lr utilises dans ce tour.

Resultat deja obtenu (a rejouer pour verifier) : M=25 passe 11-13x
MOINS de temps dans cette fenetre que M=0 -- un vrai effet dynamique,
distinct de la geometrie statique du pli (identique entre M=0 et
M=25, verifie independamment plusieurs fois).
"""

import sys
sys.path.insert(0, '.')
import torch

from verifier_jouet_n_variable import construire_toy_m, objectif_toy_m, N

DELTA_C_M0 = 0.01867676
DELTA_C_M25 = 0.01855957


def temps_dans_fenetre(M, delta, lr, r_autres_init, pas=5000, lo=0.990, hi=0.999):
    poids3 = torch.tensor((1.0 - delta) / N, dtype=torch.float64)
    poids4 = torch.tensor((1.0 + delta) / N, dtype=torch.float64)
    p3, p4, q = construire_toy_m(M, delta, r_autres_init=r_autres_init)
    opt = torch.optim.Adam([p3, p4, q], lr=lr, eps=1e-10)
    t_entree, t_sortie, n_dans = None, None, 0
    for i in range(pas):
        j = objectif_toy_m(p3, p4, q, poids3, poids4)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        with torch.no_grad():
            s3 = torch.sigmoid(p3).item()
        if lo <= s3 <= hi:
            if t_entree is None:
                t_entree = i
            t_sortie = i
            n_dans += 1
    return t_entree, t_sortie, n_dans


if __name__ == "__main__":
    print("=== temps passe dans la fenetre s3 in [0,990;0,999] "
          "(le vrai col, x*=0,0042 -> s3*=0,9958) ===")
    for M, r_autres_init, delta in [(0, 0.01, DELTA_C_M0), (25, 0.01, DELTA_C_M25)]:
        for lr in (0.2, 0.05):
            t_in, t_out, n = temps_dans_fenetre(M, delta, lr, r_autres_init)
            print(f"  M={M:2d}  lr={lr:4}  entree={t_in}  sortie={t_out}  "
                  f"n_pas_dans_fenetre={n}")
