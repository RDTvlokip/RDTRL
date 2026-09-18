"""Challenge agent-dipankar (18/09/2026, tour non-commite) sur la
"decouverte du jour" : seuil naturel s3_init=0,972646/0,972661
(delta=0,013, pas=4000, lr=0,05), vitesse minimale a t=263 pres de H6
(0,994327 / 0,829197), et fit a≈+2000 (signe oppose a a_delayed=-9,70).

Points testes ici, dans l'ordre :
  1. Reproduction brute du seuil a pas=4000.
  2. QUAND : le seuil tient-il a un horizon plus long (pas=40000, comme
     le ROUND 1 historique de verifier_sonde_bassin.py) ? La session
     precedente (CARNET.md ~L10150) a deja trouve qu'a pas=40000,
     "tout reste gradue y compris a s3_init=0,994" -- contredit
     frontalement le ROUND 1 historique (effondrement net a 0,994 a
     pas=40000). Si le nouveau seuil 0,9726 est lui aussi un artefact
     d'horizon, prolonger doit le faire "gradue" aussi.
  3. Vrai creux de vitesse ou point isole bruite : trace pas par pas
     (check_tous=1) autour de t=263, verifie que |Δs3| est bien MINIMAL
     localement (fenetre t=200-350), pas juste petit a un point.
  4. Detection d'oscillation cycle-2 d'Adam (deja documentee dans
     verifier_trajectoire_naturelle_mur23.py pour d'autres deltas) :
     Δs3 alterne-t-il de signe pas a pas dans la fenetre du fit ?
  5. Refit du coefficient a, MEME convention que a_H6direct/a_delayed
     (verifier_forme_normale_pli.analyser_xv : v=Δs3/Δpas vs x=s3, PAS
     centre) pour verifier que le centrage ne change pas le signe/la
     magnitude de A (il ne devrait pas, translation pure).
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from verifier_prior_asymetrique import continuer_sous_prior, etat
from verifier_sonde_bassin import poids_delta, fixer_s3, fixer_r4
from verifier_forme_normale_pli import moindres_carres_quadratique, analyser_xv

ADAM_EPS = 1e-10
DELTA = 0.013


def run_jusqua(cible_s3, pas, lr=0.05, verbose=True):
    poids = poids_delta(DELTA)
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    fixer_s3(e, cible_s3)
    continuer_sous_prior(e, r, BETA, pas, lr, ADAM_EPS, poids)
    R, H, m, Hb, S = etat(e, r)
    issue = "GRADUEE" if S[3] > 0.5 else "EFFONDRE"
    if verbose:
        print(f"  cible_s3={cible_s3:.6f}  pas={pas:6d}  ->  "
              f"s3_final={S[3]:.6f}  R[10,4]={R[4]:.6f}  [{issue}]")
    return S[3], R[4]


def trace_fine(cible_s3, pas_total, lr=0.05):
    """Trace s3(t), R4(t) pas par pas (granularite 1, pas 20)."""
    poids = poids_delta(DELTA)
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    fixer_s3(e, cible_s3)
    trace = []
    R, H, m, Hb, S = etat(e, r)
    trace.append((0, S[3], R[4]))
    for t in range(1, pas_total + 1):
        continuer_sous_prior(e, r, BETA, 1, lr, ADAM_EPS, poids)
        R, H, m, Hb, S = etat(e, r)
        trace.append((t, S[3], R[4]))
    return trace


if __name__ == "__main__":
    torch.set_printoptions(precision=6)

    print("=== 1. Reproduction du seuil a pas=4000 (comme annonce) ===")
    s3_bas, r4_bas = run_jusqua(0.972646, 4000)
    s3_haut, r4_haut = run_jusqua(0.972661, 4000)

    print("\n=== 2. QUAND : le seuil tient-il a pas=40000 (horizon ROUND 1 historique) ? ===")
    s3_bas40, r4_bas40 = run_jusqua(0.972646, 40000)
    s3_haut40, r4_haut40 = run_jusqua(0.972661, 40000)
    print("  (si les deux redeviennent GRADUEE a pas=40000 : le 'seuil' pas=4000")
    print("   est un artefact d'horizon, pas une vraie bifurcation asymptotique)")

    print("\n=== 2bis. Test intermediaire de securite : pas=4000 vs 12000 vs 40000 sur 0,972646 seul ===")
    for pas in (4000, 8000, 12000, 20000, 40000):
        run_jusqua(0.972646, pas)

    print("\n=== 3. Trace fine (pas par pas) de la trajectoire SOUS le seuil (0,972646), 400 pas ===")
    trace = trace_fine(0.972646, 400)
    # chercher le minimum de |delta s3| pas a pas
    deltas = []
    for k in range(1, len(trace)):
        t0, s0, r0 = trace[k - 1]
        t1, s1, r1 = trace[k]
        deltas.append((t1, s1 - s0, s1, r1))
    deltas.sort(key=lambda x: abs(x[1]))
    print("  10 plus petits |Δs3| (t, Δs3, s3, R4) :")
    for t, d, s, rv in deltas[:10]:
        print(f"    t={t:4d}  Δs3={d:+.3e}  s3={s:.6f}  R4={rv:.6f}")

    print("\n=== 4. Fenetre t=230-295 pas par pas : creux ou point isole ? oscillation cycle-2 ? ===")
    fenetre = [x for x in trace if 225 <= x[0] <= 300]
    signes = []
    for k in range(1, len(fenetre)):
        t0, s0, r0 = fenetre[k - 1]
        t1, s1, r1 = fenetre[k]
        d = s1 - s0
        signes.append((t1, d))
    for t, d in signes:
        print(f"    t={t:4d}  Δs3={d:+.3e}")
    changements_signe = sum(
        1 for i in range(1, len(signes))
        if signes[i][1] != 0 and signes[i - 1][1] != 0
        and (signes[i][1] > 0) != (signes[i - 1][1] > 0)
    )
    print(f"  changements de signe de Δs3 dans cette fenetre : {changements_signe}/{len(signes)-1}")

    print("\n=== 5. Refit de a, MEME convention que verifier_forme_normale_pli (v=Δs3/Δpas, x=s3 brut) ===")
    print("    fenetre pas=100-400, stride=2/5/10/20 (comme annonce)")
    for stride in (2, 5, 10, 20):
        sous = [(t, s) for (t, s, r) in trace if 100 <= t <= 400]
        xs, vs = [], []
        for k in range(0, len(sous) - stride, stride):
            t0, s0 = sous[k]
            t1, s1 = sous[k + stride]
            v = (s1 - s0) / (t1 - t0)
            xs.append((s0 + s1) / 2)
            vs.append(v)
        if len(xs) < 3:
            print(f"  stride={stride:2d}  n={len(xs)}  -- pas assez de points")
            continue
        A, B, C, x0, mu = analyser_xv(f"stride={stride}", xs, vs)

    print("\n=== 6. Rappel de la convention a_delayed / a_H6direct (verifier_forme_normale_pli.py) ===")
    print("    (ne re-derive pas leurs chiffres publies -- juste confirme la fonction de fit")
    print("    utilisee est identique : moindres_carres_quadratique sur (x=s3, v=Δs3/Δpas))")
