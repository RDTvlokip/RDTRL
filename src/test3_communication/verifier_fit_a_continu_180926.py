"""Suite et fin du challenge agent-dipankar (18/09/2026).

verifier_continuite_optimiseur_180926.py vient de montrer que le bug
que je soupçonnais (redémarrage d'Adam tous les 20 pas dans
scratch_trace_naturel.py / verifier_trajectoire_naturelle_mur23.py)
est réel, MAIS que ma première tentative de réplication
(verifier_decouverte_seuil_naturel_180926.py / verifier_aliasing_
cycle2_180926.py, qui appelait continuer_sous_prior AVEC PAS=1 À
CHAQUE PAS) avait le MÊME bug en pire -- ce n'était PAS une réplication
correcte de la découverte du jour. Avec un unique optimiseur Adam
CONTINU (protocole A), le point t=263 rapporté (s3=0,994327,
R4=0,829197) est reproduit À LA DÉCIMALE PRÈS, et la vitesse
pas-à-pas y croise authentiquement zéro (t=264: +3,4e-9, t=266:
-2,4e-9) -- un vrai ralentissement lisse, pas un artefact
d'échantillonnage.

Ici : refit du coefficient `a` sur CETTE trajectoire correcte
(protocole A, optimiseur unique), même fenêtre (pas=100-400), mêmes
strides (2,5,10,20), même centrage x_ref=0,9943 que l'annonce -- pour
vérifier si a≈+2000 tient sur la trajectoire VRAIMENT continue (et pas
seulement sur une reconstruction buguée qui, par coïncidence, donne les
mêmes coordonnées à t=263 mais pourrait avoir un profil de vitesse
différent ailleurs dans la fenêtre)."""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from verifier_prior_asymetrique import objectif_pondere, etat
from verifier_sonde_bassin import poids_delta, fixer_s3
from representable_atteignable_stable import activer, parametres
from verifier_forme_normale_pli import moindres_carres_quadratique

ADAM_EPS = 1e-10
DELTA = 0.013
LR = 0.05
X_REF = 0.9943


def trace_continue(cible_s3, pas_total):
    poids = poids_delta(DELTA)
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    fixer_s3(e, cible_s3)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=LR, eps=ADAM_EPS)
    trace = []
    R, H, m, Hb, S = etat(e, r)
    trace.append((0, S[3], R[4]))
    for t in range(1, pas_total + 1):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        R, H, m, Hb, S = etat(e, r)
        trace.append((t, S[3], R[4]))
    return trace


def fit_stride(trace, pas_min, pas_max, stride, x_ref):
    sous = [(t, s3) for (t, s3, r4) in trace if pas_min <= t <= pas_max]
    xs, vs = [], []
    for k in range(0, len(sous) - stride, stride):
        t0, s0 = sous[k]
        t1, s1 = sous[k + stride]
        v = (s1 - s0) / (t1 - t0)
        x_mid = (s0 + s1) / 2 - x_ref
        xs.append(x_mid)
        vs.append(v)
    A, B, C = moindres_carres_quadratique(xs, vs)
    x0 = -B / (2 * A) + x_ref if A != 0 else None
    mu = C - B**2 / (4 * A) if A != 0 else None
    return A, B, C, x0, mu, len(xs)


if __name__ == "__main__":
    torch.set_printoptions(precision=6)

    print("=== Trace continue (protocole A), s3_init=0.972646, 500 pas ===")
    trace = trace_continue(0.972646, 500)
    for (t, s3, r4) in trace:
        if t % 50 == 0:
            print(f"  t={t:4d}  s3={s3:.6f}  R4={r4:.6f}")

    print("\n=== Fit de a, fenetre pas=100-400, centre x_ref=0.9943, plusieurs strides ===")
    for stride in (2, 5, 10, 20):
        A, B, C, x0, mu, n = fit_stride(trace, 100, 400, stride, X_REF)
        print(f"  stride={stride:2d}  n={n:3d}  x0={x0:.7f}  mu={mu:.3e}  a={A:.4e}")

    print("\n=== Meme fit, mais fenetre etendue pas=100-500 (voir si a converge en ajoutant des pas) ===")
    for stride in (2, 5, 10, 20):
        A, B, C, x0, mu, n = fit_stride(trace, 100, 500, stride, X_REF)
        print(f"  stride={stride:2d}  n={n:3d}  x0={x0:.7f}  mu={mu:.3e}  a={A:.4e}")

    print("\n=== Fit sur sous-fenetres DISJOINTES de la fenetre pas=100-400, stride=2 ===")
    print("    (teste la robustesse -- a_H6direct historique variait x5.6 selon la sous-fenetre)")
    for lo, hi in ((100, 175), (175, 250), (250, 325), (325, 400)):
        A, B, C, x0, mu, n = fit_stride(trace, lo, hi, 2, X_REF)
        print(f"  fenetre=[{lo},{hi}]  n={n:3d}  x0={x0:.7f}  mu={mu:.3e}  a={A:.4e}")

    print("\n=== Rappel pour comparaison directe ===")
    print("  a_delayed (CARNET.md, deja publie) = -9.70 (stable, x7.56-11.76 sur 3 fenetres)")
    print("  a_H6direct (CARNET.md, historique)  ~ -98.5 (instable x5.6 selon fenetre)")
    print("  a rechauffement reel (CARNET.md, n_chauffe)  varie -20.9 (n=5) a +0.31 (n=160), signe change")
