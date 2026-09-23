"""Agent (role dipankarsarkar, tour 59 simule) : PREDICTION, par le spectre
exact de M = lr P^-1/2 H P^-1/2, de l'instant ou le mode du gap franchit 38
selon le reglage de la ligne 3 de l'emetteur :
  - eps de la ligne 3 (eps3), comme dans l'ablation du tour 58 ;
  - facteur k sur le lr de la ligne 3 (k=0 : ligne 3 gelee ; k=2, k=4).
La trajectoire est la trajectoire de base (eps3=1e-10, k=1) : on suppose
que H et v ne dependent pas du reglage avant le franchissement (la ligne
3 ne pese que 5 % du mode). Le decalage predit est
  t_c(reglage) - t_c(base), t_c = premier pas ou S_mode >= 38,
interpole lineairement entre les pas mesures.

Usage : python agent_dk59_predire_decalage.py <delta> <pas1> <pas2> ...
"""
import sys
sys.path.insert(0, '.')
import torch

from agent_dk59_hessien_explicite import (hessien, v_plat, spectre, mode_gap,
                                          pas_adam, L3)
from sauver_checkpoints_mur23_tour58 import reprendre

REGLAGES = [("base", None, 1.0), ("eps3=1e-9", 1e-9, 1.0), ("eps3=1e-8", 1e-8, 1.0),
            ("eps3=1e-7", 1e-7, 1.0), ("eps3=1e-5", 1e-5, 1.0),
            ("k=0", None, 1e-30), ("k=2", None, 2.0), ("k=4", None, 4.0)]
E310 = 3 * 27 + 10
AUTRES3 = [3 * 27 + j for j in range(27) if j != 10]
I_R3, I_R4 = 729 + 10 * 27 + 3, 729 + 10 * 27 + 4


if __name__ == "__main__":
    torch.set_num_threads(1)
    delta = float(sys.argv[1])
    cibles = sorted(int(a) for a in sys.argv[2:])
    e, r, opt, poids = reprendre(delta)
    pas = 54000
    table = {nom: [] for nom, _, _ in REGLAGES}
    for c in cibles:
        while pas < c:
            pas_adam(e, r, opt, poids)
            pas += 1
        x, H = hessien(e, r, poids)
        v = v_plat(e, r, opt)
        ligne = f"pas={pas}"
        for nom, eps3, k in REGLAGES:
            M, D, lam, W = spectre(H, v.clone(), eps3, k)
            kk, _ = mode_gap(lam, W)
            xr = D * W[:, kk]
            X = xr[E310] - xr[AUTRES3].mean()
            gp = xr[I_R4] - xr[I_R3]
            table[nom].append((pas, lam[kk].item()))
            ligne += f"  {nom}: S={lam[kk].item():.4f} X/gap={(X/gp).item():+.4f}"
        print(ligne)
        sys.stdout.flush()
    print("--- franchissement de 38 (interpolation lineaire) ---")
    tc = {}
    for nom, pts in table.items():
        t = None
        for (p0, s0), (p1, s1) in zip(pts, pts[1:]):
            if s0 < 38.0 <= s1:
                t = p0 + (38.0 - s0) * (p1 - p0) / (s1 - s0)
                break
        if t is None and pts[-1][1] < 38.0 and len(pts) >= 2:
            (p0, s0), (p1, s1) = pts[-2], pts[-1]
            t = p1 + (38.0 - s1) * (p1 - p0) / (s1 - s0)   # extrapolation
        if t is None and pts[0][1] >= 38.0:
            (p0, s0), (p1, s1) = pts[0], pts[1]
            t = p0 - (s0 - 38.0) * (p1 - p0) / (s1 - s0)   # extrapolation arriere
        tc[nom] = t
    for nom in tc:
        print(f"  {nom:10s} t_c={tc[nom]:.1f}   decalage vs base = {tc[nom]-tc['base']:+.1f} pas")
