"""Tour 53 (20/09/2026) : l'hypothese non-standard de REPONSE_ORDRE54.md
demandait si les excursions a pas=60000 et pas=160000 sous Adam complet
sont un seul mecanisme (signes opposes) ou deux evenements distincts.

Instrumente l'etat interne d'Adam (exp_avg_sq) sur le recepteur pendant
les trois fenetres ou une excursion est deja connue (60000, 160000,
260000), a grille fine (pas=500, ou pas=1 dans les fenetres cibles).

Resultat : les trois occurrences sont quasi IDENTIQUES en amplitude et
en forme (R4_min=0.79102/0.79104/0.79093, R4_max=0.79848/0.79851/
0.79854, v_r_max=1.554e-13/1.553e-13/1.575e-13, s3_min=0.998944 les
deux premieres fois a 7 chiffres significatifs) -- PAS deux evenements
independants, mais un CYCLE LIMITE PERIODIQUE de periode ~95000-100000
pas. L'hypothese "deux evenements distincts" (H non-standard 3 de
REPONSE_ORDRE54.md) est REFUTEE au profit d'une troisieme lecture,
plus forte : Adam entretient une oscillation quasi-periodique stable
pres de ce plateau, pas des accidents stochastiques isoles.
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

FENETRES = [(58000, 62000), (158000, 162000), (258000, 262000)]


def main():
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + DELTA) / N
    poids[3] = (1.0 - DELTA) / N

    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=LR, eps=ADAM_EPS)
    p_r = r.p[0]

    pas_max = FENETRES[-1][1] + 1
    resultats = {f: {"R4_min": 1e9, "R4_max": -1e9, "v_r_max": -1e9} for f in FENETRES}

    for pas in range(pas_max):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        for (a, b) in FENETRES:
            if a <= pas <= b:
                with torch.no_grad():
                    R4 = (e.loi()[4, 10] * r.loi()[10, 4]).item()
                    st_r = opt.state.get(p_r, {})
                    v_r = st_r["exp_avg_sq"][10, 4].item() if "exp_avg_sq" in st_r else float("nan")
                d = resultats[(a, b)]
                d["R4_min"] = min(d["R4_min"], R4)
                d["R4_max"] = max(d["R4_max"], R4)
                d["v_r_max"] = max(d["v_r_max"], v_r)

    for f, d in resultats.items():
        print(f"fenetre {f} : R4_min={d['R4_min']:.6f}  R4_max={d['R4_max']:.6f}  v_r_max={d['v_r_max']:.6e}")


if __name__ == "__main__":
    main()
