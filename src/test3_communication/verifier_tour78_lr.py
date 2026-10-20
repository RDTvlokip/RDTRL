"""Tour 78 (vraie critique de dipankarsarkar, 23/10/2026) : le bloc 2 x 2 (gap, sender) change-t-il avec lr, a eps/lr fixe ?

Il dit : eps ne rencontre que sqrt(v_e3) dans p_z = 1/(sqrt(v_e3) + eps) + 1/(sqrt(v_e3) + 26 eps), lr n'y entre pas, le niveau 38 non plus ; eps/lr fixe ne tient le bloc que si sqrt(v_e3) au franchissement
double quand lr double ; sinon eps 2e-10 est un eps plus gros : p_z baisse de 3 %, partenaire/niveau pres de 0,056 et non 0,0585. Mon modele : sqrt(v) est la variable qui s'ajuste pour que S_c = 38 :
S_c(lr, sqrt(v), eps) est invariant par (lr, sqrt(v), eps) -> (c lr, c sqrt(v), c eps), donc sqrt(v_e3) double et le bloc normalise est le meme.

Modes :
  arithmetique
  reduction <module> <eps> <x> [<nrec>] : reduction (module a lr fixe, b1 = 0,9) au franchissement S_c = 38 de trois salves : bloc (gap, sender), partenaire/niveau, poids, sqrt(v_e3), sqrt(v_l3), p_z
  Pour le reseau complet a un autre lr : verifier_tour77_bloc_eps.py reseau <eps> <debut> <fin> <delta> <lr>
"""

import importlib
import math
import os
import sys

import numpy as np
from mpmath import mp, mpf

sys.path.insert(0, ".")
from verifier_tour75_salves_depart import etat_base, courbe_Sc, PLI
from verifier_tour74_valeur_propre import jac6, valeurs_propres_3x3
from verifier_tour76_bloc_2x2 import bloc_sym

mp.dps = 50


def pz(s, e):
    return 1 / (s + e) + 1 / (s + 26 * e)


def mode_arithmetique():
    s = 4e-8
    print(f"p_z a sqrt(v_e3) = 4e-8 : eps 1e-10 {pz(s, 1e-10):.4e} ; 2e-10 {pz(s, 2e-10):.4e} ; 1e-9 {pz(s, 1e-9):.4e}")
    print(f"baisse 1e-10 -> 2e-10 : {100 * (1 - pz(s, 2e-10) / pz(s, 1e-10)):.1f} % ; 1e-10 -> 1e-9 : {100 * (1 - pz(s, 1e-9) / pz(s, 1e-10)):.1f} % ; baisse du partenaire {100 * (1 - 0.0441 / 0.0585):.1f} %")
    pente = math.log(0.0441 / 0.0585) / math.log(pz(s, 1e-9) / pz(s, 1e-10))
    print(f"pente log = {pente:.3f} ; extrapolation a eps 2e-10 : {0.0585 * (pz(s, 2e-10) / pz(s, 1e-10)) ** pente:.4f}")


def mode_reduction(nom, eps, xs, nrec):
    redlib = importlib.import_module(nom)
    lr = redlib.LR
    b1 = redlib.B1
    thr = 2 * (1 + b1) / (1 - b1)
    S1 = etat_base()
    delta = float(PLI - mpf(xs))
    ks = [int(a) for a in np.random.default_rng(7).integers(0, 4444, 16)]
    rng = np.random.default_rng(11)
    w = redlib.copie(S1)
    if os.environ.get("TOUR78_ECHELLE"):
        # tour 71 : l'etat de base a ses moments de lr 0,05 ; m x lr/0,05 et v x (lr/0,05)^2 evitent le coup de depart
        w["m"] = w["m"] * (lr / 0.05)
        w["v"] = w["v"] * (lr / 0.05) ** 2
    redlib.avancer(w, float(PLI - mpf("1e-6")), 20000 + ks[0], eps=eps, cross=2.0, stop=False)
    w["x"] = w["x"] * (1.0 + 1e-9 * rng.standard_normal(6))
    redlib.avancer_obs(w, delta, 10000, np.zeros((1, 7)), eps=eps)
    x6 = w["x"].copy()
    w0 = redlib.copie(w)
    rec = np.zeros((nrec, 7))
    first, maxrb, rb, done = redlib.avancer_obs(w, delta, nrec, rec, eps=eps)
    if maxrb > 0.9:
        print(f"{nom} lr = {lr} eps = {eps:.1e} x = {xs} : effondre")
        return
    u = rec[:, 2]; s_e3 = rec[:, 3]; s_l3 = rec[:, 4]
    grille, sc_g, J = courbe_Sc(x6, delta, lr, eps, float(np.median(s_e3 / s_l3)), float(np.median(s_l3)), b1)
    Sc = np.interp(np.log(s_l3), np.log(grille), sc_g)
    D2 = u[2:] - 2 * u[1:-1] + u[:-2]
    A = np.abs(D2) / 4.0
    E = np.maximum(A[:-1], A[1:])
    idx = np.nonzero(E > 3e-4)[0]
    if len(idx) == 0:
        print(f"{nom} lr = {lr} eps = {eps:.1e} x = {xs} : pas de salve (S_c max {Sc.max():.2f}, seuil {thr:.0f})")
        return
    coupe = np.nonzero(np.diff(idx) > 30)[0]
    debuts = np.concatenate([[0], coupe + 1]); fins = np.concatenate([coupe, [len(idx) - 1]])
    groupes = [(idx[a], idx[b]) for a, b in zip(debuts, fins)]
    res = []
    for g in range(1, min(len(groupes), 4)):
        prev_end = groupes[g - 1][1]; a = groupes[g][0]
        cand = np.nonzero(Sc[prev_end + 5:a] >= thr)[0]
        if len(cand) == 0:
            continue
        t_c = prev_end + 5 + int(cand[0])
        w1 = redlib.copie(w0)
        niveau_exact = bool(os.environ.get("TOUR80_NIVEAU"))
        if not niveau_exact:
            redlib.avancer(w1, delta, t_c + 2, eps=eps, cross=2.0, stop=False)
            J6 = jac6([mpf(float(q)) for q in w1["x"]], mpf(delta))
            Jn = np.array([[float(J6[i, j]) for j in range(6)] for i in range(6)])
            bl = bloc_sym(Jn, w1["v"], eps, lr)
            bl["sv_e3"] = math.sqrt(w1["v"][0]); bl["sv_l3"] = math.sqrt(w1["v"][4])
            bl["pz"] = pz(bl["sv_e3"], eps)
            res.append(bl)
        else:
            # tour 80 : balayage de +-14 pas autour du franchissement interpole, bloc a chaque pas, interpolation lineaire des grandeurs au niveau S_max = thr
            redlib.avancer(w1, delta, max(t_c - 14, 1), eps=eps, cross=2.0, stop=False)
            suite = []
            for _ in range(29):
                redlib.avancer(w1, delta, 1, eps=eps, cross=2.0, stop=False)
                J6 = jac6([mpf(float(q)) for q in w1["x"]], mpf(delta))
                Jn = np.array([[float(J6[i, j]) for j in range(6)] for i in range(6)])
                bl = bloc_sym(Jn, w1["v"], eps, lr)
                bl["sv_e3"] = math.sqrt(w1["v"][0]); bl["sv_l3"] = math.sqrt(w1["v"][4])
                bl["pz"] = pz(bl["sv_e3"], eps)
                bl["niv0"] = float(bl["niveaux"][0])
                suite.append(bl)
            ok = [i for i in range(len(suite) - 1) if suite[i]["S_max"] < thr <= suite[i + 1]["S_max"]]
            if not ok:
                continue
            i = ok[0]
            fr = (thr - suite[i]["S_max"]) / (suite[i + 1]["S_max"] - suite[i]["S_max"])
            bl = {}
            for k in ("S_max", "d_gap", "d_s", "c", "part_gap", "part_s", "part_autre", "ferm", "sv_e3", "sv_l3", "pz", "niv0"):
                bl[k] = (1 - fr) * suite[i][k] + fr * suite[i + 1][k]
            bl["niveaux"] = np.array([bl["niv0"], bl["S_max"]])
            res.append(bl)
    if not res:
        print(f"{nom} lr = {lr} eps = {eps:.1e} x = {xs} : aucun franchissement")
        return
    f = lambda key: np.mean([r_[key] for r_ in res])
    nv = np.mean([r_["niveaux"] for r_ in res], axis=0)
    print(f"{nom} lr = {lr} eps = {eps:.1e} (eps/lr = {eps / lr:.1e}) x = {xs} ({len(res)} salves) : niveau {f('S_max'):.3f} ; partenaire/niveau = {nv[0] / f('S_max'):.5f} ; poids du sender = {f('part_s'):.5f} ; d_s/niveau = {f('d_s') / f('S_max'):.5f} ; d_gap/niveau = {f('d_gap') / f('S_max'):.5f} ; fermeture {f('ferm'):.4f}")
    print(f"   sqrt(v_e3) = {f('sv_e3'):.4e} ; sqrt(v_l3) = {f('sv_l3'):.4e} ; rapport {f('sv_e3') / f('sv_l3'):.4f} ; p_z = {f('pz'):.4e} ; lr p_z = {lr * f('pz'):.4e}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "arithmetique":
        mode_arithmetique()
    else:
        mode_reduction(sys.argv[2], float(sys.argv[3]), sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 300000)
