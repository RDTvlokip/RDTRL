"""Agent (role dipankarsarkar, tour 59 simule) : poids[4] n'entre pas dans
d g_e3 / d gap (separabilite, tour 58 -- identite de construction). Entre-t-il
dans l'AUTRE moitie, la valeur propre du mode couple (donc l'instant de la
salve), a etat fixe ? Via H44 = d2(-J)/d r10[4]^2 et S_gap = 2 lr H44 / P_g.

Meme etat (delta reel, pas 59500 ; delta=0, pas 59500), quatre ponderations
(W0, Wr, W4, W3 comme dans verifier_reponse_dipankar_tour58_separabilite_et_eps0.py),
v d'Adam inchange : S_gap, S du mode couple, dS(ligne 3), kappa = H(e3[10],r4)/H44.
Et, a delta=0 : le vecteur propre de la plus petite valeur propre de M
(negative, -5,7e-6) -- ou vit la direction montante de J ?
"""
import sys
sys.path.insert(0, '.')
import torch

from representable_atteignable_stable import N
from agent_dk59_hessien_explicite import (hessien, v_plat, spectre, mode_gap, pas_adam,
                                          I_R3, I_R4, E310, L3)
from sauver_checkpoints_mur23_tour58 import reprendre

DR = 0.013026615


def ponderation(p3, p4):
    w = torch.full((N,), 1.0 / N, dtype=torch.float64)
    w[3], w[4] = p3, p4
    return w


if __name__ == "__main__":
    torch.set_num_threads(1)
    for delta in (DR, 0.0):
        e, r, opt, poids = reprendre(delta)
        for _ in range(59500 - 54000):
            pas_adam(e, r, opt, poids)
        v = v_plat(e, r, opt)
        print(f"=== etat delta={delta}, pas 59500 ===")
        W = {"W0": ponderation(1 / N, 1 / N), "Wr": ponderation((1 - DR) / N, (1 + DR) / N),
             "W4": ponderation(1 / N, (1 + DR) / N), "W3": ponderation((1 - DR) / N, 1 / N)}
        for nom, w in W.items():
            x, H = hessien(e, r, w)
            M, D, lam, Wv = spectre(H, v.clone())
            k, _ = mode_gap(lam, Wv)
            dvec = torch.zeros_like(x)
            dvec[I_R4], dvec[I_R3] = 1.0, -1.0
            dvec = dvec / D
            dvec = dvec / dvec.norm()
            s_gap = (dvec @ M @ dvec).item()
            kappa = (H[E310, I_R4] / H[I_R4, I_R4]).item()
            print(f"  {nom}: H44={H[I_R4, I_R4].item():.6e}  H(e310,r4)={H[E310, I_R4].item():+.6e}  "
                  f"kappa={kappa:.6e}  S_gap={s_gap:.4f}  S_mode={lam[k].item():.4f}  "
                  f"dS={lam[k].item() - s_gap:+.4f}")
            if nom == "W0" and delta == 0.0 or nom == "Wr" and delta == DR:
                wmin = Wv[:, -1]
                xr = D * wmin
                top = xr.abs().topk(6)
                noms = [("e" if i < 729 else "r") + str(divmod(i % 729, 27)) for i in top.indices.tolist()]
                print(f"    plus petite valeur propre de M = {lam[-1].item():.4e} ; direction brute, "
                      f"6 plus grosses coordonnees : {list(zip(noms, [round(t, 4) for t in (xr[top.indices] / xr.norm()).tolist()]))}")
