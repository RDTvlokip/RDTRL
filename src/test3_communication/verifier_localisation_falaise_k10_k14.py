"""Suite de la retractation "plat est generique, K=1 exception"
(CARNET.md, section jouet a K variable) : localise plus precisement
la falaise entre le plateau (K=1..8 a l'ecart 0.03%) et le regime plat
(K=20,26), qu'on savait seulement situee "quelque part entre K=8 et
K=20" apres la correction du defaut methodologique (ecart teste plus
petit que la tolerance de bissection).

Etape 1 -- CONTROLE IMPORTANT, pas evident au depart : la largeur (en
delta) de la zone de ralentissement critique RETRECIT avec K. Teste a
un ecart fixe de 1% (largement au-dessus de toute tolerance de
bissection, donc fiable) : K=1 montre encore un signal fort
(premier=260) mais K=8 est DEJA plat (premier=60) a ce meme ecart,
alors qu'il faisait partie du "plateau" a 0.03%. Un test a ecart trop
grossier (1%) pour K=10/14/18 aurait donc ete non informatif -- il
fallait retester au meme ecart fin (0.03%) qui avait revele le
plateau K=1-8.

Etape 2 -- localisation a 0.03% (tol=1e-6 pour la bissection de
delta_c, marge ~4.4x au-dessus de la tolerance, acceptable mais pas la
marge x100 ideale) :
  K=10 : premier=300  (plateau, comme K=1..8)
  K=14 : premier=60   (plat, comme K=20,26)
  K=18 : premier=60   (plat, confirme)

Falaise localisee entre K=10 et K=14 -- pas encore plus fin (K=11,12,13
non testes).
"""

import sys
import time
sys.path.insert(0, '.')
import torch

from verifier_jouet_k_emetteur_variable import construire_toy, objectif_toy, BETA, N, bissecter_delta_c


def trace(K, delta, check_tous=20, pas_max=20000):
    poids3 = torch.tensor((1.0 - delta) / N, dtype=torch.float64)
    poids4 = torch.tensor((1.0 + delta) / N, dtype=torch.float64)
    p3, p4, q = construire_toy(K, s3_init=0.999, s4_init=0.999, r_init=0.5)
    opt = torch.optim.Adam([p3, p4, q], lr=0.05, eps=1e-10)
    trajectoire = []
    for pas in range(pas_max):
        j = objectif_toy(p3, p4, q, poids3, poids4, K)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        if pas % check_tous == 0:
            with torch.no_grad():
                r4 = torch.sigmoid(q).item()
            trajectoire.append((pas, r4))
    with torch.no_grad():
        r4_final = torch.sigmoid(q).item()
    return trajectoire, r4_final


def premier_pas_1pct(traj, r4_final):
    cible = 0.99 * r4_final if r4_final < 1.0 else 0.99
    for pas, r4 in traj:
        if r4 >= cible:
            return pas
    return None


def main():
    print("=== controle : largeur de la zone critique retrecit avec K (ecart 1%) ===")
    for K, dc in [(1, 0.018699), (8, 0.015098)]:
        delta = dc * (1 - 0.01)
        traj, r4_final = trace(K, delta)
        p = premier_pas_1pct(traj, r4_final)
        print(f"  K={K}  premier(1%)={p}")

    print("\n=== localisation a 0.03% (l'ecart qui revele le plateau) ===")
    for K in (10, 14, 18):
        t0 = time.time()
        dc = bissecter_delta_c(K, r_init=0.5, lo=0.0130, hi=0.0200, tol=1e-6, pas=40000)
        delta = dc * (1 - 0.0003)
        traj, r4_final = trace(K, delta)
        p = premier_pas_1pct(traj, r4_final)
        print(f"  K={K}  delta_c={dc:.6f}  premier(0.03%)={p}  time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
