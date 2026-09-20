"""Tour 54 (20/09/2026, vraie critique de dipankarsarkar) : teste sa
prediction falsifiable sur le mecanisme H2 -- au point stabilise de
l'optimiseur hybride (Adam emetteur seul, SGD recepteur seul), il
predit que le ratio |dR|/|ds3|=0.0470 mesure s'explique par
1-s[4,10]~6.12e-05 (l'emetteur presque mais pas tout a fait sature),
via dR = R*(1-s[4,10])*delta (formule qui suppose dr[10,4]=0, le
recepteur SGD fige pendant le nudge commun cote emetteur).

Log s[4,10] et r[10,4] SEPAREMENT (pas seulement leur produit R) sur
toute la trace hybride de 400000 pas, y compris au pas exact du dip
qu'il cite (pas=300000, R:0.7947556089->0.7947549760).

Resultat : 1-s[4,10] reste dans la gamme 1e-12 a 3e-12 sur TOUTE la
trajectoire, y compris exactement au dip cite -- sept ordres de
grandeur plus sature que sa prediction (6.12e-05), et plus extreme
encore que son scenario de rejet alternatif (~0.89). Sa prediction est
REFUTEE. Un agent-dipankar a ensuite trace la cause : s[4,10] est
sature DES construire_mur23() (perturbation +30 sur le logit donne
1-s[4,10]~26*exp(-30)~3e-11 par construction, avant meme que
l'optimiseur hybride ne fasse un pas), et dR=r*ds4+s4*dr avec s4~1
fixe donne dR~=dr[10,4] a 12 chiffres significatifs (verifie sur un
vrai evenement de dip). Son hypothese dr[10,4]=0 etait inversee --
c'est d(s[4,10])~=0 qui tient. Sa conclusion qualitative (deficit cote
recepteur) survit, par un mecanisme plus simple que sa formule.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere

DELTA = 0.013026615
PAS_MAX = 400_000
CHECK_TOUS = 20_000
LR_ADAM_E = 0.05
ADAM_EPS = 1e-10
LR_SGD_R = 50.0


def main():
    torch.set_printoptions(precision=10)
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + DELTA) / N
    poids[3] = (1.0 - DELTA) / N

    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    opt_e = torch.optim.Adam(e.p, lr=LR_ADAM_E, eps=ADAM_EPS)
    opt_r = torch.optim.SGD(r.p, lr=LR_SGD_R)

    for pas in range(PAS_MAX):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt_e.zero_grad()
        opt_r.zero_grad()
        (-j).backward()
        opt_e.step()
        opt_r.step()
        if pas % CHECK_TOUS == 0:
            with torch.no_grad():
                s410 = e.loi()[4, 10].item()
                r104 = r.loi()[10, 4].item()
            print(f"pas={pas:<7}  s[4,10]={s410:.10f}  r[10,4]={r104:.10f}  "
                  f"1-s[4,10]={1-s410:.6e}  R={s410*r104:.10f}")

    with torch.no_grad():
        s410 = e.loi()[4, 10].item()
        r104 = r.loi()[10, 4].item()
        s3_final = e.loi()[3, 10].item()
    print(f"FINAL  s[4,10]={s410:.10f}  r[10,4]={r104:.10f}  "
          f"1-s[4,10]={1-s410:.6e}  s3={s3_final:.10f}  R={s410*r104:.10f}")


if __name__ == "__main__":
    main()
