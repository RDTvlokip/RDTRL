"""Suite du tour 56 (21/09/2026, VRAIE critique de dipankarsarkar) :
precommis dans REPONSE_ORDRE56.md, pas encore execute a l'envoi.

On a trouve que r[10,3] (ligne 10 du recepteur, sur les REFERENTS) a
un miroir quasi-exact : d_logit_r[10,3] ~= -d_logit_r[10,4] a ~0,007%
pres. Et que l'approximation standard ds3(prob) ~= s3(1-s3)*dlogit_s3
est fausse d'un facteur ~1,9x -- suggere que logit_s3 (ligne 3 de
l'EMETTEUR, sur les MESSAGES, colonne 10) a lui aussi un partenaire
anti-correle parmi les 26 autres messages, jamais mesure.

Ce script log la ligne 3 complete de l'emetteur (27 messages) sur la
meme fenetre [58000,62000) que la trace 10-decimales deja envoyee.

Resultat (pas ce qui etait attendu) : PAS un partenaire miroir unique
comme r3<->r4. LES 26 autres messages bougent tous par la MEME
quantite, a 10 chiffres significatifs pres (CV~2,6e-11, verifie
independamment deux fois -- une fois par un agent d'audit qui a
corrige ma premiere version qui ne rapportait qu'un "top-6" trompeur,
une fois par moi directement). Redistribution uniforme, pas un
partenaire dominant -- coherent avec s[3,10]~0,999 ecrasant 26
alternatives quasi-degenerees (meme signature K=26 que H7, confirmee
par un agent comme un lien reel, pas une coincidence 27-1=26 :
H7 utilise deja depuis le 14/09 la meme reduction "message 10 contre
les 26 autres groupees").

La quasi-degenerescence des 26 (CV~2e-11) n'est PAS figee des
l'initialisation -- verifie (agent + moi, trajectoire reconciliee
dans verifier_cv_decroissance_ligne3.py) : le CV decroit de facon
dynamique et monotone depuis ~2,1 juste apres la perturbation +30
jusqu'a ~0 apres ~60000-100000 pas cumules -- une vraie convergence
d'entrainement, pas un artefact de tirage aleatoire au depart.
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

        # CORRECTION (agent d'audit, 21/09/2026) : le top-6 par magnitude
        # est trompeur -- LES 26 bougent de facon quasi identique (a
        # 10 chiffres pres), le classement top-6 n'a aucune signification
        # structurelle (juste l'ordre du bruit flottant au 12e/13e
        # chiffre). On rapporte les 26 valeurs et le CV, pas un top-6.
        autres_vals = [v for _, v in vals]
        import statistics
        moyenne = statistics.mean(autres_vals)
        ecart_type = statistics.pstdev(autres_vals)
        cv = ecart_type / abs(moyenne) if moyenne else float('nan')

        somme_autres = sum(abs(v) for v in autres_vals)
        print(f"pas={pas_test}  d_logit_s[3,10]={d_s3:.6e}")
        print(f"  LES 26 autres messages (ligne 3) : min={min(autres_vals):.15e}  "
              f"max={max(autres_vals):.15e}")
        print(f"  moyenne={moyenne:.15e}  ecart-type={ecart_type:.6e}  CV={cv:.6e}")
        print(f"  somme |d_logit_s[3,j]| pour j!=10 = {somme_autres:.6e}")
        ratio = moyenne / d_s3 if d_s3 else float('nan')
        print(f"  ratio moyenne(26)/d_s[3,10] = {ratio:.6f}")


if __name__ == "__main__":
    main()
