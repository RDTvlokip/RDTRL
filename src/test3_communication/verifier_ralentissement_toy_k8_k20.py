"""Tour du jouet a K variable, 20/09/2026 : test precommis par un
agent-dipankar (voir CARNET.md, section "jouet a K variable"), pas
execute au moment ou il l'a propose. K=26 montre un ralentissement
critique PLAT (verifier_ralentissement_toy_k26.py, 40/60/60 pas) et
K=1 un vrai ralentissement (verifier_ralentissement_toy_k1_controle.py,
80/300/340 pas, x4.25) malgre un pli d'equilibre 12x plus loin de son
propre delta_c. Question posee : a K=8 ou K=20 (strictement entre les
deux), le temps de convergence sort-il plat, ralenti, ou intermediaire ?

Resultat :
  K=20 (delta_c~0.01378)  : 40, 60, 60 pas   -- PLAT, meme signature que K=26
  K=8  (delta_c~0.015098) : 60, 60, 320 pas  -- signal FAIBLE, seulement
                                                 au point le plus proche
                                                 du seuil (0.03%)

Tranche la question : "plat" est le cas GENERIQUE (vrai systeme, K=20,
K=26 tous plats) -- K=1 est l'exception isolee, pas une transition
cachee qui encadrerait K=26. Le signal faible a K=8 suggere une
decroissance progressive de l'effet entre K=1 et K=8, deja eteinte a
K=20 -- K=2 a K=7 non testes, l'endroit precis ou ca s'eteint reste
ouvert.
"""

import sys
import time
sys.path.insert(0, '.')
import torch

from verifier_jouet_k_emetteur_variable import construire_toy, objectif_toy, BETA, N

CONFIGS = [(8, 0.015098), (20, 0.013780)]
DISTANCES_RELATIVES = (0.03, 0.003, 0.0003)
PAS_MAX = 20000
CHECK_TOUS = 20


def trace(K, delta, pas_max=PAS_MAX, check_tous=CHECK_TOUS):
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


def main():
    for K, delta_c in CONFIGS:
        print(f"=== K={K}, delta_c~{delta_c} ===")
        for frac in DISTANCES_RELATIVES:
            delta = delta_c * (1 - frac)
            t0 = time.time()
            traj, r4_final = trace(K, delta)
            cible = 0.99 * r4_final if r4_final < 1.0 else 0.99
            premier = None
            for pas, r4 in traj:
                if r4 >= cible:
                    premier = pas
                    break
            print(f"  delta={delta:.7f} ({frac*100:.3f}% sous delta_c)  "
                  f"r4_final={r4_final:.6f}  premier_pas_1pct={premier}  "
                  f"time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
