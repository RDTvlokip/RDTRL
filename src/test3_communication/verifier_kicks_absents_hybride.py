"""Tour 53 (20/09/2026) : dernier point non teste du rapport de
l'agent-dipankar sur le mecanisme "plancher de v" (voir
verifier_mecanisme_plancher_v.py, verifier_eps_supprime_kicks.py) --
son pari, formule mais jamais rejoue par lui : le kick de R (~0.0037
tous les ~460-500 pas sous Adam complet) doit etre SPECIFIQUE a Adam
cote recepteur, donc absent sous l'optimiseur hybride
(verifier_optimiseur_hybride.py : Adam sur l'emetteur seul, SGD pur
sur le recepteur seul), puisque le recepteur n'y a alors aucun etat
"v" a faire s'effondrer vers un plancher.

Resultat : CONFIRME. 0 evenement detecte sur 20000 pas, meme a un
seuil plus bas (0.0005) que celui utilise partout ailleurs (0.0015) --
donc pas un artefact de seuil trop strict. Cohere avec l'excursion
residuelle deja documentee sous l'hybride (~6.3e-7, bien plus petite,
a des pas differents), qui vient d'un canal different : l'etat Adam
de l'EMETTEUR (cf. CARNET.md 7.65, test precommis de dipankar refute).
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
LR_ADAM_E = 0.05
LR_SGD_R = 50.0
PAS_MAX = 20000
SEUIL = 0.0005


def main():
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + DELTA) / N
    poids[3] = (1.0 - DELTA) / N

    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    opt_e = torch.optim.Adam(e.p, lr=LR_ADAM_E, eps=ADAM_EPS)
    opt_r = torch.optim.SGD(r.p, lr=LR_SGD_R)

    vals = []
    for pas in range(PAS_MAX):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt_e.zero_grad()
        opt_r.zero_grad()
        (-j).backward()
        opt_e.step()
        opt_r.step()
        with torch.no_grad():
            R4 = (e.loi()[4, 10] * r.loi()[10, 4]).item()
        vals.append((pas, R4))

    baseline = sum(v for p, v in vals if 15000 <= p < PAS_MAX) / len(
        [v for p, v in vals if 15000 <= p < PAS_MAX]
    )
    print(f"baseline = {baseline:.10f}")
    pics = detecter_evenements(vals, baseline, seuil=SEUIL, pas_min=5000)
    print(f"n_events (seuil {SEUIL}) = {len(pics)}")


if __name__ == "__main__":
    main()
