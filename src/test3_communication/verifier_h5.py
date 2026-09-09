"""H5 (corrige) : la signature de bascule se retrouve-t-elle sur un
autre mur pousse jusqu'a une VRAIE capture plutot qu'une egalite ?

Correction du 09/09/2026 : la premiere version masquait le logit gagnant
a -1e9 puis sommait la ligne entiere par erreur (bug de calcul, pas
d'exclusion). Corrige ici. Egalement : eps=60 ne suffisait pas a casser
l'egalite (reste a 0,500000 exactement apres 40000 pas) -- teste plus
fort (100, 200, 400) pour voir si une vraie capture est meme possible
sur cette paire, ou si elle est structurellement bloquee en egalite.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import replay, monter_avec_eps_adam, BETA

torch.set_printoptions(precision=15)

for eps_perturb in (60.0, 100.0, 200.0, 400.0):
    print(f"=== referents 3/4, message 10, eps_perturb={eps_perturb}, adam_eps=1e-10 ===")
    for pas_suite in (10000, 20000, 40000):
        e, r = replay(77777, 3, 10000)
        with torch.no_grad():
            e.p[0][4, 10] += eps_perturb
        monter_avec_eps_adam(e, r, BETA, pas_suite, 0.05, 1e-10)
        with torch.no_grad():
            s_, r_ = e.loi(), r.loi()
            row = e.p[0][4]
            somme_hors_gagnant = (row.sum() - row[10]).item()
            un_moins_s4 = 1.0 - s_[4, 10].item()
            R = (s_[4, 10] * r_[10, 4]).item()
        print(f"  pas={pas_suite:6d}  R[10,4]={R:.6f}  1-s[4,10]={un_moins_s4:.6e}  "
              f"somme_hors_gagnant={somme_hors_gagnant:.6f}  r[10,4]={r_[10,4].item():.6f}")
