"""Tour 67 : zero de D_d (mediane d3 - noeud au delta mesure) en eps/lr, a x fixe, par module de reduction.

Usage : python verifier_tour67_zero_dd.py <module> <x> <diviseur_eps_o3> <eps/lr> [<eps/lr> ...]
  module  : reduction_numba_tour59 (lr 0,05), reduction_numba_tour63_lr002 (0,02), reduction_numba_tour67_lr010 (0,1),
            reduction_numba_tour67_lr001 (0,01)
  x       : pli - delta (ex. 1e-8)
  diviseur_eps_o3 : eps de la coordonnee o3 = eps / diviseur (1 = eps scalaire ; 26 = test de dipankarsarkar :
            le pas de o3 devient exactement l'oppose de celui de e3)
Etat de base : etat_base_reduction_mur23.json (verifier_tour67_exporter_etat_base.py), pas le .pt hors depot.
16 phases (ks = default_rng(7)), chauffe 20 000 + k pas a pli-1e-6 sous le meme eps, perturbation 1e-9 (default_rng(11)),
10 000 pas de mise en place, 2 000 000 pas enregistres. Sortie par eps/lr : D_d moyen +- ES sur les phases, sd(d3),
sqrt(v_l3) median, S = lr 2 |J_uu| / (sqrt(v_l3) + eps), phases effondrees ; puis le zero de D_d (interpolation lineaire
entre les deux points qui l'encadrent) avec un intervalle bootstrap sur les phases (2000 tirages).
"""

import importlib
import json
import sys

import numpy as np
from mpmath import mp, mpf

sys.path.insert(0, ".")
from verifier_tour61_decomposition_d3 import PLI, d3_noeud

mp.dps = 50
J_UU = 1.1204e-4


def etat_base():
    d = json.load(open("etat_base_reduction_mur23.json", encoding="utf-8"))
    return dict(x=np.array(d["x"]), m=np.array(d["m"]), v=np.array(d["v"]), Se=d["Se"], Sed=d["Sed"], t=d["t"])


def zero(qs, D):
    """premier changement de signe de D (liste de moyennes) : interpolation lineaire en q ; None s'il n'y en a pas."""
    for i in range(len(qs) - 1):
        if D[i] < 0 <= D[i + 1] or D[i] > 0 >= D[i + 1]:
            return qs[i] + (qs[i + 1] - qs[i]) * (0 - D[i]) / (D[i + 1] - D[i])
    return None


if __name__ == "__main__":
    mod, x, div = sys.argv[1], sys.argv[2], float(sys.argv[3])
    qs = [float(a) for a in sys.argv[4:]]
    redlib = importlib.import_module(mod)
    lr = redlib.LR
    s1 = etat_base()
    ks = [int(a) for a in np.random.default_rng(7).integers(0, 4444, 16)]
    delta = float(PLI - mpf(x))
    d3n = float(d3_noeud(PLI - mpf(x)))
    print(f"module {mod} (lr {lr}) x {x} eps_o3 = eps/{div:g}")
    print("eps/lr     eps         D_d(noeud) +- ES       sd_d3       med sqrt(v_l3)   S       effondrees")
    Dphases, Dmean = [], []
    for q in qs:
        eps = q * lr
        e = np.full(6, eps); e[1] = eps / div
        rng = np.random.default_rng(11)
        Dd, sd, vl, esc = [], [], [], 0
        for k in ks:
            w = redlib.copie(s1)
            redlib.avancer(w, float(PLI - mpf("1e-6")), 20000 + k, eps=e, cross=2.0, stop=False)
            w["x"] = w["x"] * (1.0 + 1e-9 * rng.standard_normal(6))
            redlib.avancer_obs(w, delta, 10000, np.zeros((1, 7)), eps=e)
            rec = np.zeros((2_000_000, 7))
            first, maxrb, rb, done = redlib.avancer_obs(w, delta, 2_000_000, rec, eps=e)
            if maxrb > 0.9:
                esc += 1
                continue
            ex = 26.0 * np.exp(-rec[:, 1]); d3 = ex / (1.0 + ex)
            Dd.append(np.median(d3) - d3n); sd.append(d3.std()); vl.append(np.median(rec[:, 4]))
        if len(Dd) < 8:
            print(f"{q:.3e}  {eps:.3e}  effondre ({esc}/16)", flush=True)
            Dphases.append(None); Dmean.append(None)
            continue
        Dd = np.array(Dd)
        S = lr * 2 * J_UU / (np.mean(vl) + eps)
        print(f"{q:.3e}  {eps:.3e}  {Dd.mean():+.4e} +- {Dd.std(ddof=1) / np.sqrt(len(Dd)):.1e}   {np.mean(sd):.4e}   {np.mean(vl):.4e}   {S:6.2f}   {esc}", flush=True)
        Dphases.append(Dd); Dmean.append(Dd.mean())
    ok = [i for i, d in enumerate(Dphases) if d is not None]
    qo = [qs[i] for i in ok]; Do = [Dmean[i] for i in ok]
    z = zero(qo, Do)
    if z is None:
        print("pas de changement de signe de D_d sur la grille")
    else:
        rng = np.random.default_rng(5)
        zs = []
        for _ in range(2000):
            mm = [rng.choice(Dphases[i], len(Dphases[i])).mean() for i in ok]
            zz = zero(qo, mm)
            if zz is not None:
                zs.append(zz)
        lo, md, hi = np.percentile(zs, [16, 50, 84]) if zs else (np.nan,) * 3
        print(f"zero de D_d : eps/lr = {z:.4e} ; bootstrap sur les phases (16 / 50 / 84 %) : {lo:.4e} {md:.4e} {hi:.4e}  ({len(zs)}/2000 tirages avec un zero)")
