"""Suite du tour 56 (21/09/2026, VRAIE critique de dipankarsarkar) :
precommis dans REPONSE_ORDRE56.md, pas encore execute a l'envoi.

On a trouve que r[10,3] (ligne 10 du recepteur, sur les REFERENTS) a
un miroir quasi-exact : d_logit_r[10,3] ~= -d_logit_r[10,4] a ~0,007%
pres. Et que l'approximation standard ds3(prob) ~= s3(1-s3)*dlogit_s3
est fausse d'un facteur ~1,9x -- suggere que logit_s3 (ligne 3 de
l'EMETTEUR, sur les MESSAGES, colonne 10) a lui aussi un partenaire
anti-correle parmi les 26 autres messages, jamais mesure.

Ce script log la ligne 3 complete de l'emetteur (27 messages) sur la
meme fenetre [58000,62000) que la trace 10-decimales deja envoyee, et
cherche le/les partenaire(s) miroir de logit_s3 (colonne 10) -- si
l'hypothese tient, un ou plusieurs autres logit_s[3,j] devraient
bouger de facon anti-correlee avec logit_s3 pendant le kick.
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
POINTS_CLES = (58000, 59000, 59989, 60432, 61000, 61999)


def tracer():
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + DELTA) / N
    poids[3] = (1.0 - DELTA) / N

    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=LR, eps=ADAM_EPS)

    p_e = e.p[0]
    logits = {}
    for pas in range(PAS_MAX):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        if pas >= DEBUT:
            with torch.no_grad():
                logits[pas] = p_e[3, :].clone()
    return logits


def main():
    logits = tracer()

    p0, p1 = 59000, 61000
    ligne3_0 = logits[p0]
    ligne3_1 = logits[p1]

    for pas_test in (59989, 60432):
        ligne3_t = logits[pas_test]
        t = (pas_test - p0) / (p1 - p0)
        baseline_interp = ligne3_0 + t * (ligne3_1 - ligne3_0)
        d = (ligne3_t - baseline_interp)

        d_s3 = d[10].item()
        autres_idx = [j for j in range(27) if j != 10]
        vals = [(j, d[j].item()) for j in autres_idx]
        vals.sort(key=lambda x: abs(x[1]), reverse=True)

        somme_autres = sum(abs(v) for _, v in vals)
        print(f"pas={pas_test}  d_logit_s[3,10]={d_s3:.6e}")
        print(f"  top6 autres messages (ligne 3): {vals[:6]}")
        print(f"  somme |d_logit_s[3,j]| pour j!=10 = {somme_autres:.6e}")
        if vals:
            j_top, d_top = vals[0]
            ratio = d_top / d_s3 if d_s3 else float('nan')
            print(f"  plus gros mouvement: message {j_top}, ratio d_s[3,{j_top}]/d_s[3,10] = {ratio:.6f}")


if __name__ == "__main__":
    main()
