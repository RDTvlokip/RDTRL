"""Tour 51 de dipankarsarkar : test precommis qui separe H6 (noeud-col,
avec un jumeau instable calculable) de H11 (crise de bord, bassin
fractal). A delta=0,013 (dans la region graduee), sa forme fermee
predit un point instable a s[3,10]=0,994300 -- l'endroit exact ou la
trajectoire devrait bifurquer entre "monte vers la branche graduee
(~0,999000)" et "s'effondre vers 1/27".

Prediction s'il a raison (noeud-col, seuil net) :
  init s3=0,9945 -> branche graduee (0,999000)
  init s3=0,9943 -> branche graduee
  init s3=0,9940 -> effondre (1/27)
  init s3=0,9900 -> effondre (1/27)
  et le seuil reel doit tomber a moins de 1e-4 de 0,994300.

Si le seuil est flou / imprevisible depuis la taille de la perturbation
plutot qu'un seuil net a cet endroit precis : bassin fractal, H11 tient.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import N
from verifier_prior_asymetrique import continuer_sous_prior, etat

ADAM_EPS = 1e-10
DELTA = 0.013
PAS = 40000


def fixer_s3(e, cible_s3, referent=3, message=10):
    """Fixe s[referent,message] a la valeur cible en ne modifiant QUE le
    logit de cette case, les 26 autres logits de la ligne restant
    inchanges (donc leurs poids relatifs entre eux aussi)."""
    with torch.no_grad():
        ligne = e.p[0][referent, :].clone()
        z10_actuel = ligne[message].item()
        ligne_sans = torch.cat([ligne[:message], ligne[message+1:]])
        somme_autres = torch.exp(ligne_sans - ligne_sans.max()).sum().item() * \
            torch.exp(ligne_sans.max()).item()
        import math
        nouveau_z10 = math.log(cible_s3 / (1 - cible_s3) * somme_autres)
        e.p[0][referent, message] = nouveau_z10
    with torch.no_grad():
        s_verif = e.loi()[referent, message].item()
    return s_verif


def fixer_r4(r, cible_r4, message=10, referent=4):
    """Meme principe que fixer_s3, mais sur la ligne du RECEPTEUR (message
    10), colonne referent 4 -- pour placer l'etat initial sur le meme
    point que celui predit par la forme fermee (s3 ET R ensemble), pas
    seulement s3 en laissant R a son point de depart (l'egalite a 0,5)."""
    with torch.no_grad():
        ligne = r.p[0][message, :].clone()
        ligne_sans = torch.cat([ligne[:referent], ligne[referent+1:]])
        somme_autres = torch.exp(ligne_sans - ligne_sans.max()).sum().item() * \
            torch.exp(ligne_sans.max()).item()
        import math
        nouveau_z = math.log(cible_r4 / (1 - cible_r4) * somme_autres)
        r.p[0][message, referent] = nouveau_z
    with torch.no_grad():
        r_verif = r.loi()[message, referent].item()
    return r_verif


def poids_delta(delta):
    p = torch.full((N,), 1.0 / N, dtype=torch.float64)
    p[4] = (1.0 + delta) / N
    p[3] = (1.0 - delta) / N
    return p


if __name__ == "__main__":
    torch.set_printoptions(precision=6)
    poids = poids_delta(DELTA)

    print(f"=== ROUND 1 : perturber s3 SEUL, R laisse a l'egalite de depart (0,5) ===")
    print(f"    jumeau instable predit = 0.994300")
    for cible in (0.99500, 0.99450, 0.99430, 0.99420, 0.99410, 0.99400, 0.99000, 0.98000):
        e, r = construire_mur23(adam_eps=ADAM_EPS)
        s_verif = fixer_s3(e, cible)
        continuer_sous_prior(e, r, BETA, PAS, 0.05, ADAM_EPS, poids)
        R, H, m, Hb, S = etat(e, r)
        issue = "BRANCHE GRADUEE" if S[3] > 0.5 else "EFFONDRE (1/27)"
        print(f"  init s3={cible:.5f} (verif={s_verif:.6f})  ->  "
              f"s3_final={S[3]:.6f}  R[10,4]={R[4]:.6f}  [{issue}]")

    print(f"\n=== ROUND 2 : perturber s3 ET R ENSEMBLE, sur la variete predite ===")
    print(f"    point instable joint predit : s3=0.994300, R[10,4]=0.829390")
    R_UNSTABLE = 0.829390
    for cible_s3 in (0.99500, 0.99450, 0.99430, 0.99420, 0.99400, 0.99000, 0.95000, 0.90000):
        e, r = construire_mur23(adam_eps=ADAM_EPS)
        s_verif = fixer_s3(e, cible_s3)
        r_verif = fixer_r4(r, R_UNSTABLE)
        continuer_sous_prior(e, r, BETA, PAS, 0.05, ADAM_EPS, poids)
        R, H, m, Hb, S = etat(e, r)
        issue = "BRANCHE GRADUEE" if S[3] > 0.5 else "EFFONDRE (1/27)"
        print(f"  init s3={cible_s3:.5f} (verif={s_verif:.6f})  R4 init verif={r_verif:.6f}  ->  "
              f"s3_final={S[3]:.6f}  R[10,4]={R[4]:.6f}  [{issue}]")
