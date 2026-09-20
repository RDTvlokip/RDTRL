"""Tour 53 (20/09/2026) : apres la retractation du "cycle limite
periodique" (les trois occurrences a pas=60000/160000/260000 n'etaient
qu'un biais de selection -- des fluctuations comparables existent
partout, cf. verifier_periodicite_excursion.py et le journal
CARNET.md), question suivante du cycle QUAND/COMBIEN/DEPUIS QUAND :
le bruit de R sous Adam complet est-il stationnaire sur toute la
trajectoire, ou sa frequence/amplitude derive-t-elle ?

Decoupe [0,400000] en 4 segments de 100000 pas et compare, dans
chacun, le nombre de fluctuations depassant un seuil, la deviation
moyenne, et la deviation maximale.

Resultat : compte quasi identique (9-11) sur les quatre segments,
moyenne quasi identique (0.000148-0.000172), maximum dans la meme
gamme (0.0025-0.0036 sans tendance monotone) -- le processus de bruit
est STATIONNAIRE sur toute la trajectoire mesuree, pas de derive.
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
PAS_MAX = 400000
CHECK = 500
THRESH = 0.0008
SEGMENTS = [(0, 100000), (100000, 200000), (200000, 300000), (300000, 400000)]


def main():
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + DELTA) / N
    poids[3] = (1.0 - DELTA) / N

    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=LR, eps=ADAM_EPS)

    vals = []
    for pas in range(PAS_MAX):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        if pas % CHECK == 0:
            with torch.no_grad():
                R4 = (e.loi()[4, 10] * r.loi()[10, 4]).item()
            vals.append((pas, R4))

    tail = [v for p, v in vals if 20000 <= p < 40000]
    baseline = sum(tail) / len(tail)
    print(f"baseline = {baseline:.10f}")

    for a, b in SEGMENTS:
        seg = [(p, v) for p, v in vals if a <= p < b and p > 5000]
        devs = [abs(v - baseline) for p, v in seg]
        n_above = sum(1 for d in devs if d > THRESH)
        mean_d = sum(devs) / len(devs) if devs else 0
        max_d = max(devs) if devs else 0
        print(f"segment [{a},{b}) : n={len(seg)}  count>thresh({THRESH})={n_above}  "
              f"mean|dev|={mean_d:.6f}  max|dev|={max_d:.6f}")


if __name__ == "__main__":
    main()
