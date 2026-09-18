"""Verification finale (agent-dipankar, 18/09/2026) : la session
precedente (CARNET.md ~L10150-10168) a rapporte que rejouer le seuil
ROUND 1 historique (verifier_sonde_bassin.py, s3_init=0,994300,
DELTA=0,013, PAS=40000) via continuer_sous_prior "ne reproduit pas"
l'effondrement attendu -- "tout reste gradue y compris a s3_init=0,994".

Mais verifier_continuite_optimiseur_180926.py vient de montrer que
scratch_trace_naturel.py / verifier_trajectoire_naturelle_mur23.py
(qui appellent continuer_sous_prior EN BOUCLE, PAS_BLOC=20) recreent
un optimiseur Adam a chaque bloc -- un bug distinct de continuer_sous_
prior appele UNE SEULE FOIS avec pas=40000 (round 1 style), qui LUI
est correct (un seul optimiseur, cree une fois, boucle interne).

Donc le "seuil ROUND 1 introuvable" de la session precedente n'est
PAS explique par ce bug specifique SI c'etait deja un appel unique --
a verifier ici : la bissection proche de 0,9943 est-elle reproductible
avec un optimiseur unique et continu (protocole correct), a pas=4000
ET pas=40000 ?"""

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


def run_continu(cible_s3, pas):
    poids = poids_delta(DELTA)
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    fixer_s3(e, cible_s3)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=LR, eps=ADAM_EPS)
    for _ in range(pas):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
    R, H, m, Hb, S = etat(e, r)
    return S[3], R[4]


if __name__ == "__main__":
    print("=== dtype des parametres ===")
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    print(f"  e.p dtype={e.p[0].dtype}  r.p dtype={r.p[0].dtype}")

    print("\n=== Bissection pres de 0,9943, optimiseur CONTINU, pas=4000 ===")
    for cible in (0.99000, 0.99400, 0.99420, 0.99425, 0.99428, 0.99429,
                  0.99430, 0.99431, 0.99432, 0.99435, 0.99440, 0.99450, 0.99500):
        s3f, r4f = run_continu(cible, 4000)
        issue = "GRADUEE" if s3f > 0.5 else "EFFONDRE"
        print(f"  cible_s3={cible:.5f}  ->  s3_final={s3f:.6f}  R4={r4f:.6f}  [{issue}]")

    print("\n=== Meme bissection, pas=40000 (horizon ROUND 1 historique) ===")
    for cible in (0.99400, 0.99420, 0.99428, 0.99430, 0.99432, 0.99440, 0.99450):
        s3f, r4f = run_continu(cible, 40000)
        issue = "GRADUEE" if s3f > 0.5 else "EFFONDRE"
        print(f"  cible_s3={cible:.5f}  ->  s3_final={s3f:.6f}  R4={r4f:.6f}  [{issue}]")
