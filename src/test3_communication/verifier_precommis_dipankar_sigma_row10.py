"""Tour 55 (21/09/2026, VRAIE critique de dipankarsarkar, pas simulee) :
test precommis par lui pour trancher son mecanisme "sigma" (l'off-diagonal
du softmax partage de la ligne 10 du recepteur).

Il derive, a partir de l'identite exacte du Jacobien softmax
(dr4 = r4[(1-r4)d4 - sum_{j!=4} r_j dj]) :

  |dR|/|ds3| = 157.4623 * (rho - sigma)
  rho   := dlogit_r[10,4] / dlogit_s3
  sigma := dlogit_r[10,3] / dlogit_s3

et retro-calcule, a partir du ratio mesure 147.69 (pas=60000) et de ses
propres deviations logit deja publiees (d4, ds3 a pas=59989/60432) :

  sigma(59989) = 0.24138   ->   d3 predit = -2.2642e-03
  sigma(60432) = 0.20508   ->   d3 predit = -1.9603e-03

Test precommis par lui, precis : imprimer logit_r[10,3] (et la ligne 10
complete) sur EXACTEMENT la meme fenetre/config que la trace 10
decimales deja envoyee (verifier_logits_s3_s4_r4_60000.py), et verifier
si la deviation mesuree de logit_r[10,3] a pas=59989 tombe pres de
-2.2642e-03 (a ~10% pres selon lui).

Deuxieme point souleve par lui, independant du test precommis : il
rapporte un DESACCORD entre trois lectures de rho a partir de MES
PROPRES publications (0.938 depuis le ratio mesure, 0.985 depuis mon
test precommis Delta-logit original, ~1.18/1.14 depuis ma propre trace
10-decimales). Ce script calcule rho et sigma directement depuis la
trace logit (pas depuis un ratio de deviation indirect) pour trancher
lequel de ces trois chiffres correspond a quoi.
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


def tracer(delta):
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + delta) / N
    poids[3] = (1.0 - delta) / N

    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=LR, eps=ADAM_EPS)

    p_e, p_r = e.p[0], r.p[0]
    logits = {}
    for pas in range(PAS_MAX):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        if pas >= DEBUT:
            with torch.no_grad():
                ligne10 = p_r[10, :].clone()
                s3_prob = e.loi()[3, 10].item()
                R4_prob = (e.loi()[4, 10] * r.loi()[10, 4]).item()
                logits[pas] = (p_e[3, 10].item(), p_e[4, 10].item(),
                               p_r[10, 4].item(), p_r[10, 3].item(), ligne10,
                               s3_prob, R4_prob)
    return logits


BRACKET_COEF = 157.4623


def analyser(logits, label):
    print(f"=== {label} ===")
    for pas in POINTS_CLES:
        if pas in logits:
            ls3, ls4, lr4, lr3, ligne10, s3p, R4p = logits[pas]
            print(f"pas={pas}  logit_s3={ls3:.10f}  logit_s4={ls4:.10f}  "
                  f"logit_r4={lr4:.10f}  logit_r3={lr3:.10f}  s3={s3p:.10f}  R4={R4p:.10f}")

    p0, p1 = 59000, 61000
    if p0 in logits and p1 in logits:
        ls3_0, ls4_0, lr4_0, lr3_0, _, s3p_0, R4p_0 = logits[p0]
        ls3_1, ls4_1, lr4_1, lr3_1, _, s3p_1, R4p_1 = logits[p1]
        for pas_test in (59989, 60432):
            if pas_test not in logits:
                continue
            t = (pas_test - p0) / (p1 - p0)
            base_s3 = ls3_0 + t * (ls3_1 - ls3_0)
            base_r4 = lr4_0 + t * (lr4_1 - lr4_0)
            base_r3 = lr3_0 + t * (lr3_1 - lr3_0)
            base_R4p = R4p_0 + t * (R4p_1 - R4p_0)
            base_s3p = s3p_0 + t * (s3p_1 - s3p_0)
            ls3, ls4, lr4, lr3, ligne10, s3p, R4p = logits[pas_test]
            d_s3_prob = s3p - base_s3p
            d_s3 = ls3 - base_s3
            d_r4 = lr4 - base_r4
            d_r3 = lr3 - base_r3
            d_R4p = R4p - base_R4p
            d_s3_brut = ls3 - ls3_0
            d_r4_brut = lr4 - lr4_0
            d_r3_brut = lr3 - lr3_0
            rho_interp = d_r4 / d_s3 if d_s3 else float('nan')
            sigma_interp = d_r3 / d_s3 if d_s3 else float('nan')
            rho_brut = d_r4_brut / d_s3_brut if d_s3_brut else float('nan')
            sigma_brut = d_r3_brut / d_s3_brut if d_s3_brut else float('nan')
            print(f"\n  pas={pas_test} (baseline interpolee lineaire [59000,61000)) :")
            print(f"    d_logit_s3={d_s3:.6e}  d_logit_r4={d_r4:.6e}  d_logit_r3={d_r3:.6e}")
            print(f"    rho (interp)={rho_interp:.6f}  sigma (interp)={sigma_interp:.6f}")
            print(f"  pas={pas_test} (baseline brute, difference contre pas=59000 seul) :")
            print(f"    d_logit_s3={d_s3_brut:.6e}  d_logit_r4={d_r4_brut:.6e}  d_logit_r3={d_r3_brut:.6e}")
            print(f"    rho (brut)={rho_brut:.6f}  sigma (brut)={sigma_brut:.6f}")

            # test direct de sa formule |dR|/|ds3| = 157.4623*(rho-sigma),
            # avec dR mesure EN PROBABILITE sur la MEME excursion (pas
            # un ratio importe d'un autre tour/fenetre)
            ratio_mesure = abs(d_R4p) / abs(d_s3_prob) if d_s3_prob else float('nan')
            prediction_dipankar = BRACKET_COEF * (rho_interp - sigma_interp)
            prediction_rho_seul = BRACKET_COEF * rho_interp
            print(f"    d_R4(probabilite)={d_R4p:.6e}  d_s3(probabilite)={d_s3_prob:.6e}")
            print(f"    |dR|/|ds3| MESURE directement sur cette excursion (dR et ds3 tous deux en PROBABILITE) = {ratio_mesure:.4f}")
            print(f"    prediction 157.4623*rho (sigma=0, lecture naive)  = {prediction_rho_seul:.4f}")
            print(f"    prediction 157.4623*(rho-sigma) (formule complete de dipankar) = {prediction_dipankar:.4f}")

            # somme des mouvements de la ligne 10 (verification que r3+r4 dominent)
            _, _, _, _, ligne10_test, _, _ = logits[pas_test]
            _, _, _, _, ligne10_base0, _, _ = logits[p0]
            d_ligne10 = ligne10_test - ligne10_base0
            autres = d_ligne10.clone()
            autres[3] = 0.0
            autres[4] = 0.0
            norme_autres = autres.abs().sum().item()
            print(f"    somme |d_logit_r[10,j]| pour j!=3,4 (brut vs pas=59000) = {norme_autres:.6e}")
            print(f"    (a comparer a |d_logit_r3|={abs(d_r3_brut):.6e}, |d_logit_r4|={abs(d_r4_brut):.6e})")


def main():
    logits_reel = tracer(DELTA)
    analyser(logits_reel, f"delta={DELTA} (config reelle, mur 23)")

    print()
    logits_delta0 = tracer(0.0)
    analyser(logits_delta0, "delta=0 (controle de dipankar : sigma doit s'effondrer, rho doit rester)")


if __name__ == "__main__":
    main()
