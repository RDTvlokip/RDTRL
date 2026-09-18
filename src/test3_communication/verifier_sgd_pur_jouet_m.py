"""Test (CARNET.md §7.65, 18/09/2026) pour savoir si le comportement
M-dependant trouve dans verifier_mecanisme_r_sature_jouet_m.py (R file
jusqu'a 1 pour M=25, se stabilise pour M=0) est une propriete du flot
de gradient de l'objectif lui-meme, ou un artefact specifique a Adam.

Resultat obtenu (INCONCLUSIF, reglage a corriger) : a lr=0,02, SGD pur
laisse s3 bloque a 1,00000000 pendant 100000 pas pour M=0 ET M=25 --
l'emetteur ne quitte jamais la saturation a ce taux (deja documente
ailleurs dans ce projet : Adam necessaire pour bouger une sigmoide
saturee). Le test ne peut donc pas repondre a la question posee avec
ce reglage. Pistes pour une prochaine tentative : lr SGD plus grand,
budget bien plus long, ou repartir d'un s3_init deja hors saturation.
"""

import sys
sys.path.insert(0, '.')
import torch

from verifier_jouet_n_variable import construire_toy_m, objectif_toy_m, N

DELTA_C_M0 = 0.01867676
DELTA_C_M25 = 0.01855957


def trace_sgd(M, delta, lr, r_autres_init, pas, points):
    poids3 = torch.tensor((1.0 - delta) / N, dtype=torch.float64)
    poids4 = torch.tensor((1.0 + delta) / N, dtype=torch.float64)
    p3, p4, q = construire_toy_m(M, delta, r_autres_init=r_autres_init)
    params = [p3, p4, q]
    out = []
    pset = set(points)
    for i in range(pas):
        j = objectif_toy_m(p3, p4, q, poids3, poids4)
        for p in params:
            if p.grad is not None:
                p.grad = None
        (-j).backward()
        with torch.no_grad():
            for p in params:
                p -= lr * p.grad
        if (i + 1) in pset:
            with torch.no_grad():
                s3 = torch.sigmoid(p3).item()
                r_all = torch.softmax(q, dim=0)
                r4 = r_all[1].item()
                masse_autres = 1.0 - r_all[0].item() - r4
            out.append((i + 1, s3, r4, masse_autres))
    return out


if __name__ == "__main__":
    points = [100, 500, 1000, 3000, 8000, 20000, 50000, 100000]

    print("=== SGD pur (pas Adam), lr=0,02, M=0, delta_c(M=0)=0,01867676 ===")
    for t, s3, r4, ma in trace_sgd(0, DELTA_C_M0, 0.02, 0.01, 100000, points):
        print(f"  t={t:7d}  s3={s3:.8f}  r4={r4:.8f}  masse_autres={ma:.3e}")

    print()
    print("=== SGD pur, lr=0,02, M=25, delta_c(M=25)=0,01855957 ===")
    for t, s3, r4, ma in trace_sgd(25, DELTA_C_M25, 0.02, 0.01, 100000, points):
        print(f"  t={t:7d}  s3={s3:.8f}  r4={r4:.8f}  masse_autres={ma:.3e}")
