"""Suite de verifier_masse_fond_systeme_reel.py : le renversement de
bassin sous masse de fond (seuil precis entre s3_init=0,9994521 et
0,9994526, R_init=0,60, delta=0,013) n'est visible dans AUCUN
instantane statique (gradient emetteur OU recepteur, a t=0 -- verifie
et refute deux fois avec un agent-dipankar). Le mecanisme doit donc se
jouer PENDANT l'entrainement, pas au depart.

Ce script instrumente l'entrainement complet (pas seulement l'etat
final) : a chaque pas, enregistre s3, r3, r4, la masse de fond, ET les
quatre gradients (emetteur ref3, emetteur ref4, recepteur ref3,
recepteur ref4) juste apres backward() -- pour voir OU et QUAND, au
cours des 40000 pas, un croisement ou un evenement notable apparait,
en comparant un point juste SOUS le seuil (reste gradue) a un point
juste AU-DESSUS (s'effondre).
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import activer, parametres
from verifier_prior_asymetrique import objectif_pondere
from verifier_sonde_bassin import fixer_s3, fixer_r4, poids_delta
from verifier_masse_fond_systeme_reel import fixer_masse_fond

REFERENTS_FOND = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
ADAM_EPS = 1e-10
DELTA = 0.013


def entrainer_trace(cible_s3, R_init=0.60, masse_fond=0.25, pas=40000, check_tous=50):
    poids = poids_delta(DELTA)
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    fixer_s3(e, cible_s3)
    fixer_r4(r, R_init)
    if masse_fond > 0:
        fixer_masse_fond(r, REFERENTS_FOND, masse_fond)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=0.05, eps=ADAM_EPS)

    trace = []
    for i in range(pas):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        if i % check_tous == 0:
            with torch.no_grad():
                g_e3 = e.p[0].grad[3, 10].item()
                g_e4 = e.p[0].grad[4, 10].item()
                g_r3 = r.p[0].grad[10, 3].item()
                g_r4 = r.p[0].grad[10, 4].item()
                s_ = e.loi()
                r_ = r.loi()
                s3 = s_[3, 10].item()
                r3 = r_[10, 3].item()
                r4 = r_[10, 4].item()
                idx_fond = [k for k in range(27) if k not in (3, 4)]
                masse_f = sum(r_[10, k].item() for k in idx_fond)
            trace.append((i, s3, r3, r4, masse_f, g_e3, g_e4, g_r3, g_r4))
        opt.step()
    return trace


if __name__ == "__main__":
    torch.set_printoptions(precision=6)

    for label, cible_s3 in [("SOUS le seuil (reste gradue)", 0.999450),
                             ("AU-DESSUS du seuil (s'effondre)", 0.999455)]:
        print(f"\n=== {label}, cible_s3={cible_s3} ===")
        trace = entrainer_trace(cible_s3, check_tous=50)
        for (i, s3, r3, r4, masse_f, g_e3, g_e4, g_r3, g_r4) in trace[:60]:
            print(f"  pas={i:5d}  s3={s3:.6f}  r3={r3:.6f}  r4={r4:.6f}  masse_fond={masse_f:.6f}  "
                  f"g_e3={g_e3:.4e}  g_e4={g_e4:.4e}  g_r3={g_r3:.4e}  g_r4={g_r4:.4e}")
        print("  ...")
        i, s3, r3, r4, masse_f, g_e3, g_e4, g_r3, g_r4 = trace[-1]
        print(f"  pas={i:5d} (dernier)  s3={s3:.6f}  r3={r3:.6f}  r4={r4:.6f}  masse_fond={masse_f:.6f}")
