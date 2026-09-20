"""Tour 53 (20/09/2026) : test precommis par un agent-dipankar pour
trancher entre deux lectures de la stabilite de s3 sous Adam complet.

Lecture "Hessienne" (verifier_courbure_s3_vs_r.py, ma premiere version) :
la courbure de J est ~160x plus raide en s3 qu'en R en espace
probabilite (via une conversion au CARRE du facteur de compression du
softmax), donc s3 bouge moins parce que le puits est plus etroit.

Lecture de l'agent : Adam normalise chaque coordonnee par
m/(sqrt(v)+eps), donc la courbure brute (Hessienne) n'a AUCUN effet
direct sur la taille du pas -- seul compte le ratio m/sqrt(v). Si Adam
egalise les marches en espace LOGIT (ce qui est son but de conception),
alors une SEULE puissance du facteur de compression du softmax explique
l'ecart d'amplitude visible en probabilite, pas son carre.

Test precommis par l'agent, execute ici : mesurer Delta(logit_s3) et
Delta(logit_R4) directement (pas les probabilites) sur toute la duree
d'une excursion reelle (fenetre [57500,62500], autour du dip a
pas=60000). Prediction agent (lecture lineaire) : ratio < ~3x.
Prediction Hessienne (ma premiere lecture) : ratio proche de 150x deja
en espace logit.

Resultat : |Delta_logit_R4 / Delta_logit_s3| = 0.985 -- quasi 1:1, pas
150x. La lecture Hessienne (comp^2) est REFUTEE. La lecture de l'agent
(comp^1, marches logit comparables) est CONFIRMEE.
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
DEBUT, FIN = 57500, 62500


def main():
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + DELTA) / N
    poids[3] = (1.0 - DELTA) / N

    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=LR, eps=ADAM_EPS)

    p_e, p_r = e.p[0], r.p[0]
    logit_s3_min, logit_s3_max = 1e9, -1e9
    logit_R4_min, logit_R4_max = 1e9, -1e9

    for pas in range(FIN):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        if pas >= DEBUT:
            with torch.no_grad():
                ls3 = p_e[3, 10].item()
                lR4 = p_r[10, 4].item()
            logit_s3_min = min(logit_s3_min, ls3)
            logit_s3_max = max(logit_s3_max, ls3)
            logit_R4_min = min(logit_R4_min, lR4)
            logit_R4_max = max(logit_R4_max, lR4)

    d_s3 = logit_s3_max - logit_s3_min
    d_R4 = logit_R4_max - logit_R4_min
    print(f"logit_s3 range = {d_s3:.6f}")
    print(f"logit_R4 range = {d_R4:.6f}")
    print(f"ratio |delta_logit_R4/delta_logit_s3| = {d_R4/d_s3:.6f}")


if __name__ == "__main__":
    main()
