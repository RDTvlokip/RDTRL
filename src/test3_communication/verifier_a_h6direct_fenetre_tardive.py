"""Suite directe de la piste "identite du point selle" (CARNET.md §7.65,
18/09/2026) : le chaos de la fenetre H6-direct pas<=24 vient de l'etat
interne d'Adam (m,v) pas encore verrouille sur la direction propre du
col, pas de la position de R (tranche par le test d'injection de
moments, 5 -> 0 changements de signe).

Prediction directe de ce mecanisme, PAS ENCORE TESTEE : si on laisse
Adam tourner plus longtemps a partir du MEME point de depart froid
(sans injection synthetique, juste plus de pas REELS), l'etat interne
devrait finir par se "rechauffer" tout seul et la fin de la trajectoire
devrait devenir lisse (monotone), meme si le debut reste chaotique.
Deja visible dans les deltas du cas froid (verifier_injection_moments_
adam.py) : les pas 18-24 sont TOUS negatifs (0 changement de signe),
contre 5 changements sur l'ensemble pas 1-24. Ce script verifie cette
prediction avec plus de donnees (pas 1-40) et tente un fit quadratique
STABLE sur la fenetre tardive/monotone plutot que sur la fenetre
complete deja essayee (et instable) precedemment.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import activer, parametres
from verifier_prior_asymetrique import objectif_pondere
from verifier_sonde_bassin import fixer_s3, fixer_r4, poids_delta
from verifier_forme_normale_pli import moindres_carres_quadratique, trace_vers_xv, analyser_xv

DELTA = 0.013
ADAM_EPS = 1e-10
S3_H6 = 0.994295
R_H6 = 0.829390


def compter_changements_signe(deltas):
    signes = [1 if d > 0 else (-1 if d < 0 else 0) for d in deltas]
    return sum(1 for i in range(1, len(signes))
               if signes[i] != 0 and signes[i - 1] != 0 and signes[i] != signes[i - 1])


def trace_longue(pas=40):
    poids = poids_delta(DELTA)
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    fixer_s3(e, S3_H6)
    fixer_r4(r, R_H6)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=0.05, eps=ADAM_EPS)
    trace = [(0, S3_H6)]
    for i in range(pas):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        with torch.no_grad():
            s3 = e.loi()[3, 10].item()
        trace.append((i + 1, s3))
    return trace


if __name__ == "__main__":
    torch.set_printoptions(precision=6)

    trace = trace_longue(pas=40)
    print("=== Trace complete pas=0-40 (etat froid, sans injection) ===")
    for (t, s3) in trace:
        print(f"  t={t:3d}  s3={s3:.6f}")

    deltas_tous = [trace[i + 1][1] - trace[i][1] for i in range(len(trace) - 1)]
    print()
    print(f"changements de signe sur pas 1-24 : {compter_changements_signe(deltas_tous[:24])}")
    print(f"changements de signe sur pas 1-40 : {compter_changements_signe(deltas_tous)}")

    for (lo, hi) in [(1, 24), (14, 24), (18, 24), (14, 40), (18, 40), (25, 40)]:
        sous_deltas = [trace[i + 1][1] - trace[i][1] for i in range(lo - 1, min(hi, len(trace) - 1))]
        chg = compter_changements_signe(sous_deltas)
        print(f"  fenetre pas={lo}-{hi} : n={len(sous_deltas)}  changements_signe={chg}")

    print()
    print("=== Fits quadratiques sur les fenetres tardives (monotones) ===")
    for (lo, hi) in [(14, 24), (18, 24), (14, 40), (18, 40), (25, 40), (18, 35)]:
        sous = [(t, s3) for (t, s3) in trace if lo <= t <= hi]
        if len(sous) < 4:
            continue
        xs, vs = trace_vers_xv(sous)
        A, B, C, x0, mu = analyser_xv(f"H6-direct tardif pas={lo}-{hi}", xs, vs)

    print()
    print("=== Rappel pour comparaison : a_delayed = -9.701204e+00 (x0=0.995804), stable (x1,56 entre sous-fenetres) ===")
