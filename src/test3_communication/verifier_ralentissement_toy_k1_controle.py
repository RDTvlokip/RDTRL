"""Controle pour ralentissement_toy_K.py : le meme test de
ralentissement critique, mais a K=1, ou l'analyse d'equilibre
(meanfield_fold.py) n'a trouve AUCUN pli jusqu'a delta=0.229 -- loin
au-dela du delta_c mesure par Adam (0.018699). Si K=1 est plat (pas de
ralentissement) c'est attendu (pas de pli proche). Si K=26 est AUSSI
plat malgre un pli quasi CONFONDU avec son delta_c, le contraste est le
point interessant.
"""
import sys
sys.path.insert(0, 'src/test3_communication')
import torch
from verifier_jouet_k_emetteur_variable import construire_toy, objectif_toy, BETA, N

K = 1
DELTA_C = 0.018699
PAS_MAX = 20000
CHECK_TOUS = 20


def trace(delta, pas_max=PAS_MAX):
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
        if pas % CHECK_TOUS == 0:
            with torch.no_grad():
                r4 = torch.sigmoid(q).item()
            trajectoire.append((pas, r4))
    with torch.no_grad():
        r4_final = torch.sigmoid(q).item()
    return trajectoire, r4_final


if __name__ == "__main__":
    distances_relatives = (0.03, 0.003, 0.0003)
    print(f"=== controle: ralentissement critique, toy K=1, delta_c ~ {DELTA_C} ===", flush=True)
    for frac in distances_relatives:
        delta = DELTA_C * (1 - frac)
        traj, r4_final = trace(delta)
        cible = 0.99 * r4_final if r4_final < 1.0 else 0.99
        premier_pas = None
        for pas, r4 in traj:
            if r4 >= cible:
                premier_pas = pas
                break
        print(f"  delta={delta:.7f}  ({frac*100:.3f}% sous delta_c)  "
              f"r4_final={r4_final:.6f}  premier pas a 1% de la cible = {premier_pas}", flush=True)
