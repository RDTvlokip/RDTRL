"""Tour 53 (vraie critique de dipankarsarkar, 20/09/2026) : test precommis
par lui pour trancher si l'excursion sous Adam complet et l'excursion sous
l'optimiseur hybride (Adam emetteur seul + SGD recepteur seul,
verifier_optimiseur_hybride.py) sont le MEME mecanisme a deux amplitudes,
ou deux mecanismes differents.

Prediction precommise de dipankar : si le rapport local mesure dans le
run hybride (s3_dip/R_dip = 21,3) tient aussi sous Adam complet, alors le
dip de R deja connu sous Adam complet (~1,957e-3) devrait s'accompagner
d'un dip de s3 d'environ 0,0417 (s3 tombant de ~0,9990 vers ~0,957).

Resultat (rejoue le 20/09/2026, delta=0,013026615, construire_mur23 sans
fixer_s3/fixer_r4, Adam complet lr=0,05 adam_eps=1e-10, 400000 pas,
s3 et R loggues ENSEMBLE tous les 20000 pas -- la trace originale de
dipankar (REPONSE_ORDRE52) n'avait logge que R) :

s3 reste quasi fige (variations de 1e-5 a 2e-5) pendant que R fait des
excursions de ~2e-3 -- PAS la chute vers ~0,957 predite. La prediction
precommise de dipankar est refutee : les excursions sous Adam complet
vivent presque entierement dans R, pas dans la direction (s3,R) mesuree
sous l'hybride. Localise mieux que l'amplitude ne l'avait fait : les deux
runs ne montrent probablement pas le meme mecanisme a deux echelles,
mais deux mecanismes differents partageant l'etat Adam de l'emetteur
comme seul ingredient commun necessaire (sans lui, l'optimiseur hybride
lui-meme ne montre aucune excursion cote recepteur SGD).
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import activer, parametres
from verifier_prior_asymetrique import objectif_pondere
from verifier_sonde_bassin import poids_delta

DELTA = 0.013026615
ADAM_EPS = 1e-10


def trace_s3_et_r(delta=DELTA, pas=400000, check_tous=20000, lr=0.05):
    poids = poids_delta(delta)
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=lr, eps=ADAM_EPS)
    out = []
    for i in range(pas):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        if (i + 1) % check_tous == 0:
            with torch.no_grad():
                s3 = e.loi()[3, 10].item()
                R4 = r.loi()[10, 4].item()
            out.append((i + 1, R4, s3))
    return out


if __name__ == "__main__":
    torch.set_printoptions(precision=6)
    print("=== Adam complet, delta=0,013026615, s3 et R loggues ensemble ===")
    baseline_R, baseline_s3 = None, None
    for step, R4, s3 in trace_s3_et_r():
        if baseline_R is None:
            baseline_R, baseline_s3 = R4, s3
        print(f"  step={step:7d}  R={R4:.6f}  s3={s3:.6f}  "
              f"dR={R4-baseline_R:+.3e}  ds3={s3-baseline_s3:+.3e}")
