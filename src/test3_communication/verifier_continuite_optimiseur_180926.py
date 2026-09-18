"""Suite du challenge agent-dipankar (18/09/2026), point decisif.

verifier_prior_asymetrique.continuer_sous_prior(e, r, beta, pas, lr,
adam_eps, poids) construit un `torch.optim.Adam` NEUF a CHAQUE APPEL
(ligne `opt = torch.optim.Adam(parametres(e, r), lr=lr, eps=adam_eps)`
executee au debut de la fonction). Or scratch_trace_naturel.py et
verifier_trajectoire_naturelle_mur23.tracer_trajectoire_naturelle()
appellent cette fonction en BOUCLE avec pas=PAS_BLOC=20 -- ce qui
recree un optimiseur Adam A MOMENTS REMIS A ZERO toutes les 20 pas,
PAS un unique Adam continu sur toute la trajectoire.

Or ce meme mecanisme (etat d'Adam remis a zero, "cold restart") est
EXACTEMENT celui deja diagnostique et refute dans CARNET.md (Essai 3,
"rechauffement reel", n_chauffe) comme produisant des pas de magnitude
lr quasi-complete a chaque redemarrage -- un artefact d'optimiseur, pas
la vraie dynamique locale.

Ce script compare, depuis LE MEME point de depart (s3_init=0,972646,
delta=0,013), TROIS protocoles :
  A. Optimiseur Adam UNIQUE, continu sur 400 pas (la vraie dynamique).
  B. Optimiseur recree tous les 20 pas (exactement scratch_trace_naturel.py
     / le protocole qui a produit "t=263, vitesse minimale").
  C. Optimiseur recree a CHAQUE pas (le pire cas, celui utilise par
     erreur dans verifier_decouverte_seuil_naturel_180926.py /
     verifier_aliasing_cycle2_180926.py de ce meme lot de verification --
     erreur reconnue et corrigee ici).

Si A differe qualitativement de B et C, alors TOUTE la "decouverte du
jour" (seuil, plateau a t=263, coefficient a) mesure un artefact de
redemarrage d'optimiseur, pas une propriete du systeme.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from verifier_prior_asymetrique import objectif_pondere, etat
from verifier_sonde_bassin import poids_delta, fixer_s3
from representable_atteignable_stable import activer, parametres

ADAM_EPS = 1e-10
DELTA = 0.013
LR = 0.05


def construire_depart():
    poids = poids_delta(DELTA)
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    fixer_s3(e, 0.972646)
    return e, r, poids


def trace_continue(pas_total):
    """Protocole A : UN SEUL optimiseur Adam, jamais recree."""
    e, r, poids = construire_depart()
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


def trace_redemarrage(pas_total, pas_bloc):
    """Protocole B/C : optimiseur RECREE tous les pas_bloc pas
    (continuer_sous_prior standard, appele en boucle -- exactement le
    protocole de scratch_trace_naturel.py quand pas_bloc=20)."""
    from verifier_prior_asymetrique import continuer_sous_prior
    e, r, poids = construire_depart()
    trace = []
    R, H, m, Hb, S = etat(e, r)
    trace.append((0, S[3], R[4]))
    for bloc in range(pas_total // pas_bloc):
        continuer_sous_prior(e, r, BETA, pas_bloc, LR, ADAM_EPS, poids)
        t = (bloc + 1) * pas_bloc
        R, H, m, Hb, S = etat(e, r)
        trace.append((t, S[3], R[4]))
    return trace


if __name__ == "__main__":
    torch.set_printoptions(precision=6)

    print("=== A. Optimiseur UNIQUE et continu, 400 pas (la vraie dynamique) ===")
    trace_A = trace_continue(400)
    for (t, s3, r4) in trace_A:
        if t <= 40 or t % 20 == 0:
            print(f"  t={t:4d}  s3={s3:.6f}  R4={r4:.6f}")

    print("\n=== B. Optimiseur recree tous les 20 pas (protocole scratch_trace_naturel.py) ===")
    trace_B = trace_redemarrage(400, 20)
    for (t, s3, r4) in trace_B:
        print(f"  t={t:4d}  s3={s3:.6f}  R4={r4:.6f}")

    print("\n=== Comparaison A vs B a t=260/280 (pres du 't=263' rapporte) ===")
    dictA = {t: (s3, r4) for (t, s3, r4) in trace_A}
    dictB = {t: (s3, r4) for (t, s3, r4) in trace_B}
    for t in (0, 20, 40, 100, 200, 260, 280, 300, 400):
        sA, rA = dictA[t]
        sB, rB = dictB[t]
        print(f"  t={t:4d}  A: s3={sA:.6f} R4={rA:.6f}   B: s3={sB:.6f} R4={rB:.6f}   "
              f"diff_s3={sA-sB:+.3e}")

    print("\n=== A. Vitesse pas-a-pas (protocole continu) autour de t=250-280 ===")
    for k in range(250, 281):
        t0, s0, r0 = trace_A[k]
        t1, s1, r1 = trace_A[k + 1] if k + 1 < len(trace_A) else trace_A[k]
        d = s1 - s0
        print(f"  t={t1:4d}  s3={s1:.6f}  R4={r1:.6f}  delta1={d:+.3e}")

    print("\n=== A. Seuil : bisection reprise avec optimiseur CONTINU, 4000 pas ===")
    for cible in (0.972646, 0.972661):
        poids = poids_delta(DELTA)
        e, r = construire_mur23(adam_eps=ADAM_EPS)
        fixer_s3(e, cible)
        activer(e, r)
        opt = torch.optim.Adam(parametres(e, r), lr=LR, eps=ADAM_EPS)
        for _ in range(4000):
            j, _ = objectif_pondere(e, r, BETA, poids)
            opt.zero_grad()
            (-j).backward()
            opt.step()
        R, H, m, Hb, S = etat(e, r)
        issue = "GRADUEE" if S[3] > 0.5 else "EFFONDRE"
        print(f"  cible_s3={cible:.6f}  ->  s3_final={S[3]:.6f}  R[10,4]={R[4]:.6f}  [{issue}]")
