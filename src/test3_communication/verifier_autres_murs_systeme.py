"""Decouverte du 21/09/2026, sur consigne de Theo ("continue et
continue a creuser" dans l'espace 27x27) : le "mur 23" (referents 3/4,
message 10) n'est PAS le seul desequilibre non resolu du systeme
entraine (config mur23 standard, seed 77777, k=3, checkpoint 10k,
+30 sur referent 4, 59989 pas supplementaires sous l'objectif pondere).

Trouve en creusant pourquoi le referent 5 avait un s[5,10] anormalement
eleve (0,037) dans le fil sur la derive de fond de la ligne 10 du
recepteur -- au lieu d'une "confusion avec le message 10" specifique,
toute la ligne 5 de l'emetteur s'est reveleee quasi-uniforme sur les
27 messages.

Inspection de toutes les 27 lignes de l'emetteur (jamais fait avant,
seule la ligne 10 du RECEPTEUR avait ete auditee) : 23 lignes
totalement effondrees (max_prob=1,0, H=0), et QUATRE lignes non
resolues :
- ligne 3 : le mur 23 deja etudie avec dipankarsarkar (asymetrique,
  construit par le +30 sur le referent 4)
- ligne 5 : NON-CONVERGENCE TOTALE, H proche de ln(27) -- cas
  qualitativement nouveau, jamais un simple 2-voies
- ligne 8 : egalite a 2 voies, messages 23/19, legerement asymetrique
  (+0,0065%)
- ligne 12 : egalite a 2 voies, messages 1/21, quasi EXACTEMENT a
  0,5/0,5 (precision machine) -- un mur spontane, pas construit

Reste a tester : H7/H6/comp1/plancher-de-v s'appliquent-ils a ces
trois autres murs comme au mur 23 ? Le referent 5 est-il un vrai
point-selle de dimension elevee, distinct des egalites a 2 voies
etudiees jusqu'ici ?
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
PAS_MAX = 59989


def tracer():
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + DELTA) / N
    poids[3] = (1.0 - DELTA) / N
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=LR, eps=ADAM_EPS)
    for pas in range(PAS_MAX):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
    return e, r


def main():
    e, r = tracer()
    with torch.no_grad():
        s_loi = e.loi()
        print("Lignes de l'emetteur (0-26) : max_prob et entropie")
        murs = []
        for ligne in range(27):
            row = s_loi[ligne, :]
            maxp = row.max().item()
            H = -(row * torch.log(row.clamp_min(1e-300))).sum().item()
            print(f"ligne={ligne:2d}  max_prob={maxp:.6f}  H={H:.6f}")
            if H > 1e-6:
                murs.append(ligne)

        print(f"\nLignes non effondrees : {murs}")
        for ligne in murs:
            row = s_loi[ligne, :]
            vals = [(m, row[m].item()) for m in range(27)]
            vals.sort(key=lambda x: -x[1])
            print(f"  ligne={ligne}  top4={vals[:4]}")


if __name__ == "__main__":
    main()
