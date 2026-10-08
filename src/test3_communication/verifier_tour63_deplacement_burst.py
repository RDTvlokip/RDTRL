"""Tour 63 (08/10/2026, vraie critique de dipankarsarkar) : l'inclinaison de la direction du
deplacement des medianes (D_r/D_d) sur la pente du mode mou rho, limite L = (D_r/D_d)/rho =
0,96804 a eps 1e-10 / lr 0,05, depend-elle de l'amplitude des salves ?

Il predit : L plus pres de 1 quand les salves sont plus petites (eps 1e-8 : sd ~ 4e-6 contre 1,3e-5),
D_d proportionnel a sd (3x plus petit a eps 1e-8) plutot qu'a sd^2 (9x). Je mesure sd SATURE,
D_d, D_r, D_r/D_d et L pour une reduction (eps, lr) donnee.

Usage : python verifier_tour63_deplacement_burst.py <eps> <lr> <offsets...>
  lr = 0.05 -> reduction_numba_tour59 ; lr = 0.02 -> reduction_numba_tour63_lr002 (copie avec LR = 0,02).
Protocole : etat de phase 1 (reseau complet, lr 0,05, eps 1e-10, 20 000 pas) ; 20 000 + k pas a
pli-1e-6 sous (eps, lr) ; perturbation relative 1e-9 ; 10 000 pas au delta voulu ; 2 000 000 pas enregistres.
16 phases ; phases echappees (R_b > 0,9) ecartees. Noeud/pente en mpmath (40 chiffres).
"""

import importlib
import sys
import numpy as np
from mpmath import mp, mpf

sys.path.insert(0, ".")
from hasard_reduction_tour59 import base_state
from verifier_tour61_decomposition_d3 import PLI, d3_noeud, r4_noeud, pente_fn

mp.dps = 60  # 50 chiffres ne suffisent pas pour la racine quasi double a x = 1e-30 (tolerance 1e-30)

if __name__ == "__main__":
    # eps : un flottant, ou six valeurs separees par des virgules (e3,o3,e4,o4,l3,l4) pour un eps par coordonnee
    eps = float(sys.argv[1]) if "," not in sys.argv[1] else np.array([float(a) for a in sys.argv[1].split(",")])
    lr = sys.argv[2]
    offs = sys.argv[3:]
    redlib = importlib.import_module("reduction_numba_tour59" if lr == "0.05" else "reduction_numba_tour63_lr002")
    ks = [int(a) for a in np.random.default_rng(7).integers(0, 4444, 16)]
    s1 = base_state()
    d3_pli = float(d3_noeud(PLI - mpf("1e-30"))); r4_pli = float(r4_noeud(PLI - mpf("1e-30")))
    print(f"eps = {eps if np.isscalar(eps) else list(eps)}, lr = {lr}, 16 phases x 2 000 000 pas")
    print("offset      sd_d3 (sature?)   mean-med   D_d (med - pli)    D_r (med - pli)    D_d/sd    D_r/D_d    rho(x)      L = (D_r/D_d)/rho   rapport mesure   echappees")
    for off in offs:
        delta = float(PLI + mpf(off))
        dn, rn = float(d3_noeud(PLI + mpf(off))), float(r4_noeud(PLI + mpf(off)))
        sl3, sl4 = pente_fn(d3_noeud, PLI + mpf(off)), pente_fn(r4_noeud, PLI + mpf(off))
        rho = sl4 / sl3
        rng = np.random.default_rng(11)
        sd, mm, Dd, Dr, rap, esc = [], [], [], [], [], 0
        for k in ks:
            w = redlib.copie(s1)
            redlib.avancer(w, float(PLI - mpf("1e-6")), 20000 + k, eps=eps, cross=2.0, stop=False)
            w["x"] = w["x"] * (1.0 + 1e-9 * rng.standard_normal(6))
            redlib.avancer_obs(w, delta, 10000, np.zeros((1, 7)), eps=eps)
            rec = np.zeros((2_000_000, 7))
            first, maxrb, rb, done = redlib.avancer_obs(w, delta, 2_000_000, rec, eps=eps)
            if maxrb > 0.9:
                esc += 1
                continue
            ex = 26.0 * np.exp(-rec[:, 1]); d3 = ex / (1.0 + ex)
            r4 = 1.0 / (1.0 + np.exp(rec[:, 2]) + w["Se"])
            sd.append(d3.std()); mm.append(d3.mean() - np.median(d3))
            md3, mr4 = np.median(d3), np.median(r4)
            Dd.append(md3 - d3_pli); Dr.append(mr4 - r4_pli)
            rap.append(((mr4 - rn) / sl4) / ((md3 - dn) / sl3))
        if len(sd) == 0:
            print(f"{off:>8s}   toutes les phases echappees ({esc}/16)", flush=True)
            continue
        sd, mm, Dd, Dr, rap = map(np.array, (sd, mm, Dd, Dr, rap))
        L = (Dr.mean() / Dd.mean()) / rho
        # mediane - noeud (d3) : le deplacement PAR RAPPORT AU NOEUD, Dd contient aussi le deplacement du noeud lui-meme
        mn = Dd.mean() - (dn - d3_pli)
        print(f"{off:>8s}   {sd.mean():.4e}    {mm.mean():+.3e}   {Dd.mean():+.5e}    {Dr.mean():+.5e}    {abs(Dd.mean()) / sd.mean():.3f}    {Dr.mean() / Dd.mean():.4f}    {rho:.5f}     {L:.5f}            {rap.mean():.5f}          {esc}"
              f"    mediane-noeud d3 = {mn:+.3e}", flush=True)
