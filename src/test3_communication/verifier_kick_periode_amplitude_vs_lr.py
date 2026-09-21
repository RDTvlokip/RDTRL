"""Generalisation au ML au sens large, 21/09/2026 (voir CARNET.md,
section "Generalisation au ML au sens large") : le mecanisme
plancher-de-v (oscillateur de relaxation d'Adam deja confirme de bout
en bout dans ce projet) ressemble structurellement aux "loss spikes"
periodiques observes dans les tres longs entrainements de gros
modeles, generalement attribues vaguement a des problemes de donnees
ou de precision, rarement a un mecanisme precis et testable.

Prediction fermee, posee AVANT le test : la recurrence
v_{t+1}=beta2*v_t+(1-beta2)*g_t^2 ne fait jamais intervenir lr --
seul le gradient brut alimente v. Donc la PERIODE du cycle devrait
etre quasi independante de lr. Mais le deplacement reel du parametre
est lr*m/(sqrt(v)+eps) -- donc l'AMPLITUDE du kick devrait croitre
LINEAIREMENT avec lr.

Teste sur le VRAI systeme (pas le jouet), lr in {0.025, 0.05, 0.1},
30000 pas chacun. Resultat :
  lr=0.025  espacement_median=469  amplitude_moyenne=0.001784
  lr=0.05   espacement_median=459  amplitude_moyenne=0.003518
  lr=0.1    espacement_median=421  amplitude_moyenne=0.006549

Espacement quasi independant de lr (variation <10% sur un facteur x4
de lr). Amplitude quasi proportionnelle a lr (ratio amplitude/lr
stable a 0.065-0.071 sur le meme facteur x4) -- les deux predictions
CONFIRMEES.

Consequence pratique pour le ML au sens large : baisser lr pour
reduire des loss spikes periodiques (pratique courante) ne reduirait,
selon ce mecanisme, que leur AMPLITUDE, pas leur FREQUENCE -- un
praticien observant "spikes plus petits mais toujours aussi frequents"
apres avoir baisse lr aurait ici une explication mecanistique precise.
"""

import sys
import time
sys.path.insert(0, '.')
import torch

from verifier_kicks_adam_grille_fine import detecter_evenements
from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere

DELTA = 0.013026615
ADAM_EPS = 1e-10
PAS_MAX = 30000
LR_VALEURS = (0.025, 0.05, 0.1)


def tracer_lr(lr, pas_max=PAS_MAX):
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + DELTA) / N
    poids[3] = (1.0 - DELTA) / N

    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=lr, eps=ADAM_EPS)
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
    for lr in LR_VALEURS:
        t0 = time.time()
        vals = tracer_lr(lr)
        baseline = sum(v for p, v in vals if 15000 <= p < 25000) / len(
            [v for p, v in vals if 15000 <= p < 25000]
        )
        pics = detecter_evenements(vals, baseline, seuil=0.0008, pas_min=5000)
        n = len(pics)
        med = None
        if n > 1:
            esp = sorted(pics[i + 1][0] - pics[i][0] for i in range(n - 1))
            med = esp[len(esp) // 2]
        amps = [abs(d) for p, d in pics]
        amp_moy = sum(amps) / len(amps) if amps else None
        ratio = amp_moy / lr if amp_moy else None
        print(f"lr={lr}  n_events={n}  espacement_median={med}  "
              f"amplitude_moyenne={amp_moy}  amplitude/lr={ratio}  time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
