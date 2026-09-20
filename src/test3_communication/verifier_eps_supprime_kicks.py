"""Tour 53 (20/09/2026) : test decisif proposant de trancher entre le
mecanisme "plancher de v" (confirme dans verifier_mecanisme_plancher_v.py)
et une lecture concurrente "plancher d'eps", question posee par un
agent-dipankar.

ADAM_EPS=1e-10 (defaut du projet) est tres en dessous de sqrt(v) au
plancher observe (~3e-7) -- n'entre normalement pas en jeu dans le
denominateur lr*m/(sqrt(v)+eps). Si le mecanisme est bien pilote par
v qui s'effondre vers son propre plancher numerique, relever eps a une
valeur COMPARABLE a ce plancher (~1e-6) devrait supprimer les kicks :
le denominateur ne pourrait plus jamais devenir assez petit, quel que
soit l'etat de v.

Resultat : CONFIRME, et net. adam_eps=1e-10 donne 29 kicks sur 20000
pas (espacement median 456, coherent avec verifier_kicks_adam_grille_fine.py).
adam_eps=1e-6 donne 0 kick sur 20000 pas -- disparition complete, pas
une simple attenuation. eps agit comme un plancher protecteur, pas
comme une explication concurrente du phenomene.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere
from verifier_kicks_adam_grille_fine import detecter_evenements

DELTA = 0.013026615
LR = 0.05
PAS_MAX = 20000
EPS_VALEURS = (1e-10, 1e-6)


def tracer_eps(pas_max, adam_eps):
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + DELTA) / N
    poids[3] = (1.0 - DELTA) / N
    e, r = construire_mur23(adam_eps=adam_eps)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=LR, eps=adam_eps)
    vals = []
    for pas in range(pas_max):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        with torch.no_grad():
            R4 = (e.loi()[4, 10] * r.loi()[10, 4]).item()
        vals.append((pas, R4))
    return vals


def main():
    for eps in EPS_VALEURS:
        vals = tracer_eps(PAS_MAX, eps)
        baseline = sum(v for p, v in vals if 15000 <= p < PAS_MAX) / len(
            [v for p, v in vals if 15000 <= p < PAS_MAX]
        )
        pics = detecter_evenements(vals, baseline, pas_min=5000)
        print(f"adam_eps={eps}  n_events={len(pics)}")
        if len(pics) > 1:
            esp = sorted(pics[i + 1][0] - pics[i][0] for i in range(len(pics) - 1))
            print(f"  espacement median = {esp[len(esp) // 2]}")
        amps = [abs(d) for p, d in pics]
        if amps:
            print(f"  amplitude moyenne = {sum(amps)/len(amps):.5f}  max={max(amps):.5f}")


if __name__ == "__main__":
    main()
