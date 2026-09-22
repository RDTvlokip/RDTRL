"""Tour 57 (22/09/2026) : meme detection d'evenement (grille fine,
seuil, fusion) que verifier_reponse_dipankar_tour57_delta0_meme_evenement.py,
mais sur la trajectoire delta REELLE (config mur23 standard), pour
comparer directement le pattern de signe des kicks entre delta=0 et
delta reel -- necessaire pour repondre correctement a la question de
cloture de dipankar (meme evenement retourne, ou evenement different).
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere
from verifier_kicks_adam_grille_fine import detecter_evenements

DELTA = 0.013026615
ADAM_EPS = 1e-10
LR = 0.05
PAS_MAX = 62000
DEBUT_DETECTION = 55000


def tracer_delta_reel():
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + DELTA) / N
    poids[3] = (1.0 - DELTA) / N
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=LR, eps=ADAM_EPS)
    vals_gap = []
    for pas in range(PAS_MAX):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        with torch.no_grad():
            gap = (r.p[0][10, 4] - r.p[0][10, 3]).item()
        vals_gap.append((pas, gap))
    return vals_gap


def main():
    vals_gap = tracer_delta_reel()
    baseline_gap = sum(v for p, v in vals_gap if DEBUT_DETECTION <= p < PAS_MAX) / len(
        [v for p, v in vals_gap if DEBUT_DETECTION <= p < PAS_MAX]
    )
    pics_gap = detecter_evenements(vals_gap, baseline_gap, seuil=0.0008, pas_min=DEBUT_DETECTION)

    print(f"baseline_gap(delta reel) = {baseline_gap:.10f}")
    print(f"evenements detectes sur le GAP (r4-r3), fenetre [{DEBUT_DETECTION},{PAS_MAX}):")
    for p, d in pics_gap:
        marque = "  <-- PROCHE DE 59989/60432 ?" if 59500 <= p <= 60900 else ""
        print(f"  pas={p}  delta_gap={d:.6e}{marque}")

    d_gap_59989 = dict(vals_gap)[59989] - baseline_gap
    d_gap_60432 = dict(vals_gap)[60432] - baseline_gap
    print(f"\nvaleur du gap exactement a pas=59989 et 60432 (deja publiees) :")
    print(f"  pas=59989 : delta_gap={d_gap_59989:.6e}")
    print(f"  pas=60432 : delta_gap={d_gap_60432:.6e}")


if __name__ == "__main__":
    main()
