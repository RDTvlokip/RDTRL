"""Suite tour 56 (21/09/2026, VRAIE critique de dipankarsarkar) :
repond a la question posee dans REPONSE_ORDRE56.md -- la ligne 10 du
recepteur a-t-elle aussi besoin de la somme complete du Jacobien
softmax (27 entrees) pour fermer son residu de 0,65% sur
dR = R(1-R)*(dlogit_r4-dlogit_r3), comme la ligne 3 de l'emetteur en
avait besoin pour fermer son propre residu (47% -> 0,9%) ?

Reponse : NON. La formule a 2 entrees (d4-d3) et la somme complete sur
les 27 entrees de la ligne 10 donnent EXACTEMENT le meme resultat
(0,65% et 0,64% d'ecart aux deux pas testes). Contrairement a la ligne
3 (ou s[3,10]~0,999 ecrase a lui seul les 26 autres, donc "message 10
vs son complement seul" est une mauvaise reduction), la ligne 10 a
DEUX probabilites substantielles (r3 et r4) qui dominent deja la somme
ponderee du Jacobien -- les 25 autres entrees bougent beaucoup en
LOGIT (verifier_derive_fond_ligne10.py) mais leurs PROBABILITES
individuelles restent trop petites pour peser dans la somme ponderee
Sum_j r_j*dlogit_j. Le residu de 0,65% est probablement le plancher de
precision de la linearisation au premier ordre, pas un terme manquant.
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
PAS_MAX = 62000
DEBUT = 58000


def tracer():
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + DELTA) / N
    poids[3] = (1.0 - DELTA) / N
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=LR, eps=ADAM_EPS)
    p_r = r.p[0]
    logits = {}
    probs = {}
    for pas in range(PAS_MAX):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        if pas >= DEBUT:
            with torch.no_grad():
                logits[pas] = p_r[10, :].clone()
                probs[pas] = r.loi()[10, :].clone()
    return logits, probs


def main():
    logits, probs = tracer()
    p0, p1 = 59000, 61000
    for pas_test in (59989, 60432):
        t = (pas_test - p0) / (p1 - p0)
        base_logit = logits[p0] + t * (logits[p1] - logits[p0])
        base_prob = probs[p0] + t * (probs[p1] - probs[p0])
        d_logit = logits[pas_test] - base_logit
        dr4_mesure = (probs[pas_test][4] - base_prob[4]).item()

        somme_ponderee = (base_prob * d_logit).sum().item()
        dr4_complet = (base_prob[4] * (d_logit[4] - somme_ponderee)).item()

        R_base = base_prob[4].item()
        dr4_deux_entrees = R_base * (1 - R_base) * (d_logit[4].item() - d_logit[3].item())

        print(f"pas={pas_test}")
        print(f"  dr4 mesure directement = {dr4_mesure:.6e}")
        print(f"  dr4 formule complete (27 entrees) = {dr4_complet:.6e}  "
              f"ecart={100*abs(dr4_complet-dr4_mesure)/abs(dr4_mesure):.2f}%")
        print(f"  dr4 approx 2-entrees (d4-d3) = {dr4_deux_entrees:.6e}  "
              f"ecart={100*abs(dr4_deux_entrees-dr4_mesure)/abs(dr4_mesure):.2f}%")


if __name__ == "__main__":
    main()
