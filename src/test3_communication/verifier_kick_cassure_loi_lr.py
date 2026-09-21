"""Suite de verifier_kick_periode_amplitude_vs_lr.py : un agent-dipankar
a montre que mes chiffres originaux (lr=0.025/0.05/0.1) etaient
contamines par un transitoire (l'espacement rampe de ~190 a ~470-500
pas sur les 19 premiers evenements, jamais stationnaire sur la fenetre
[5000,30000)), et que la loi "amplitude proportionnelle a lr" casse
des lr in [0.1,0.2], pas a un lr lointain jamais teste.

Ce script mesure lr=0.2 (et peut etre etendu a 0.3) avec un
diagnostic brut ET stationnaire ([20000,30000)) pour confirmer la
cassure independamment.

Resultat (confirme deux fois, agent + moi, chiffres identiques) :
  lr=0.2  espacement=325  amp/lr=0.0581  (deja hors de la zone
                                            "quasi constante" 0.065-0.071)
  steady-state [20000,30000): n=31 med=329 -- proche du brut, la
                                              cassure est reelle, pas
                                              un artefact transitoire
                                              a ce lr

Mecanisme de cassure NON explique par la saturation du pas normalise
d'Adam (refute par mesure directe, voir CARNET.md) -- hypothese
alternative de l'agent (compression softmax en aval, mecanisme comp1
deja etabli ailleurs) precommise pour le prochain tour, pas encore
testee ici.
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
LR_VALEURS = (0.2, 0.3)


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
        esp = sorted(pics[i + 1][0] - pics[i][0] for i in range(n - 1)) if n > 1 else []
        med = esp[len(esp) // 2] if esp else None
        amps = [abs(d) for p, d in pics]
        amp_moy = sum(amps) / len(amps) if amps else None
        ratio = amp_moy / lr if amp_moy else None
        print(f"lr={lr}  n_events={n}  espacement_median={med}  "
              f"amplitude_moyenne={amp_moy}  ratio_amp_lr={ratio}  time={time.time()-t0:.1f}s")

        pics_ss = detecter_evenements(vals, baseline, seuil=0.0008, pas_min=20000)
        n_ss = len(pics_ss)
        esp_ss = sorted(pics_ss[i + 1][0] - pics_ss[i][0] for i in range(n_ss - 1)) if n_ss > 1 else []
        med_ss = esp_ss[len(esp_ss) // 2] if esp_ss else None
        print(f"  steady-state [20000,30000): n={n_ss}  med={med_ss}")


if __name__ == "__main__":
    main()
