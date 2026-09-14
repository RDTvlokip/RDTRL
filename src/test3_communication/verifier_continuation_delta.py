"""Tour 50 de dipankarsarkar : contre H6 (bifurcation noeud-col), il
propose l'experience decisive -- continuer l'entrainement a delta=0,014
en PARTANT DE L'ETAT DEJA CONVERGE a delta=0,013 (proche de R=0,794,
dans le bassin gradue) plutot que de repartir de l'egalite initiale a
0,5. Version pointue de H10 :

  - si le point gradue existe encore a delta=0,014 mais que le point de
    depart initial (l'egalite a 0,5) est tombe hors de son bassin, la
    continuation depuis 0,013 doit RESTER pres de 0,80 (le point n'a pas
    disparu, seul le chemin standard ne l'atteint plus) ;
  - si le point a vraiment ete annihile (vraie bifurcation), aucune
    initialisation ne peut plus l'atteindre et on retombe a 1,000000
    meme en partant de 0,013.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import N
from verifier_prior_asymetrique import continuer_sous_prior, etat

ADAM_EPS = 1e-10
PAS = 40000

if __name__ == "__main__":
    torch.set_printoptions(precision=15)

    print("=== etape 1 : converger a delta=0,013 (le point gradue, R attendu ~0,794) ===")
    poids_013 = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids_013[4] = 1.013 / N
    poids_013[3] = 0.987 / N
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    continuer_sous_prior(e, r, BETA, PAS, 0.05, ADAM_EPS, poids_013)
    R, H, m, Hb, S = etat(e, r)
    print(f"  R[10,4]={R[4]:.6f}  s[3,10]={S[3]:.9f}  H={H:.6f}")

    print("\n=== etape 2 : CONTINUER (memes parametres, meme optimiseur non reinitialise) a delta=0,014 ===")
    poids_014 = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids_014[4] = 1.014 / N
    poids_014[3] = 0.986 / N
    continuer_sous_prior(e, r, BETA, PAS, 0.05, ADAM_EPS, poids_014)
    R2, H2, m2, Hb2, S2 = etat(e, r)
    print(f"  R[10,4]={R2[4]:.6f}  s[3,10]={S2[3]:.9f}  H={H2:.6f}")
    print(f"\n  si le point gradue existe encore : R proche de 0,80 (le point n'a pas disparu)")
    print(f"  si vraiment annihile : R=1,000000, s[3,10] proche de 1/27={1/27:.6f}")

    print("\n=== controle : depuis LA MEME graine mais reinitialisee (optimiseur neuf) directement a 0,014 ===")
    e3, r3 = construire_mur23(adam_eps=ADAM_EPS)
    continuer_sous_prior(e3, r3, BETA, PAS, 0.05, ADAM_EPS, poids_014)
    R3, H3, m3, Hb3, S3 = etat(e3, r3)
    print(f"  R[10,4]={R3[4]:.6f}  s[3,10]={S3[3]:.9f}  H={H3:.6f}")
