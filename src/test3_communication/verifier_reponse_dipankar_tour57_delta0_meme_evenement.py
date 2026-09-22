"""Tour 57 (22/09/2026, VRAIE critique de dipankarsarkar) : sa question
de cloture -- le kick a delta=0 (gap r4-r3 monte de +0,0030) est-il le
MEME evenement que le kick a delta reel (gap descend de -0,0221),
juste redirige en signe par l'asymetrie de recompense, ou un evenement
DIFFERENT qui partage seulement le pas ?

Jusqu'ici, mes mesures a delta=0 (verifier_precommis_dipankar_sigma_row10.py)
etaient prises aux MEMES pas cles que le kick delta reel (58000,59000,
59989,60432,61000,61999) -- des points PRE-SELECTIONNES sur la trajectoire
delta reel, jamais verifies comme etant un vrai evenement DETECTE
independamment sur la trajectoire delta=0 elle-meme.

Ce script detecte les evenements reels (comme
verifier_kicks_adam_grille_fine.py) sur UNE SEULE longue trace a
delta=0, grille fine, et verifie si un evenement tombe pile aux memes
pas (59989, 60432) ou a des pas differents.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere
from verifier_kicks_adam_grille_fine import detecter_evenements

ADAM_EPS = 1e-10
LR = 0.05
PAS_MAX = 62000
DEBUT_DETECTION = 55000


def tracer_delta0():
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    # delta=0 : poids uniformes exactement
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=LR, eps=ADAM_EPS)
    vals_gap = []
    vals_R4 = []
    for pas in range(PAS_MAX):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        with torch.no_grad():
            gap = (r.p[0][10, 4] - r.p[0][10, 3]).item()
            R4 = (e.loi()[4, 10] * r.loi()[10, 4]).item()
        vals_gap.append((pas, gap))
        vals_R4.append((pas, R4))
    return vals_gap, vals_R4


def main():
    vals_gap, vals_R4 = tracer_delta0()

    baseline_gap = sum(v for p, v in vals_gap if DEBUT_DETECTION <= p < PAS_MAX) / len(
        [v for p, v in vals_gap if DEBUT_DETECTION <= p < PAS_MAX]
    )
    baseline_R4 = sum(v for p, v in vals_R4 if DEBUT_DETECTION <= p < PAS_MAX) / len(
        [v for p, v in vals_R4 if DEBUT_DETECTION <= p < PAS_MAX]
    )

    pics_gap = detecter_evenements(vals_gap, baseline_gap, seuil=0.0008, pas_min=DEBUT_DETECTION)
    pics_R4 = detecter_evenements(vals_R4, baseline_R4, seuil=0.0003, pas_min=DEBUT_DETECTION)

    print(f"baseline_gap(delta=0) = {baseline_gap:.10f}")
    print(f"evenements detectes sur le GAP (r4-r3), fenetre [{DEBUT_DETECTION},{PAS_MAX}):")
    for p, d in pics_gap:
        marque = "  <-- PROCHE DE 59989/60432 ?" if 59500 <= p <= 60900 else ""
        print(f"  pas={p}  delta_gap={d:.6e}{marque}")

    print(f"\nbaseline_R4(delta=0) = {baseline_R4:.10f}")
    print(f"evenements detectes sur R4, fenetre [{DEBUT_DETECTION},{PAS_MAX}):")
    for p, d in pics_R4:
        marque = "  <-- PROCHE DE 59989/60432 ?" if 59500 <= p <= 60900 else ""
        print(f"  pas={p}  delta_R4={d:.6e}{marque}")

    print(f"\nvaleur du gap exactement a pas=59989 et 60432 (deja publiees) :")
    d_gap_59989 = dict(vals_gap)[59989] - baseline_gap
    d_gap_60432 = dict(vals_gap)[60432] - baseline_gap
    print(f"  pas=59989 : delta_gap={d_gap_59989:.6e}")
    print(f"  pas=60432 : delta_gap={d_gap_60432:.6e}")


if __name__ == "__main__":
    main()
