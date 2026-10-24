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
    if os.environ.get("TOUR81_LR"):
        # tour 81 : module a lr runtime (reduction_numba_tour71_lrvar)
        lr = float(os.environ["TOUR81_LR"])
        redlib.LRV = lr
    kidx = int(os.environ.get("TOUR81_K", "0"))   # indice du depart (graine d'echauffement et de perturbation)
    b1 = redlib.B1
    thr = 2 * (1 + b1) / (1 - b1)
    S1 = etat_base()
    delta = float(PLI - mpf(xs))
    ks = [int(a) for a in np.random.default_rng(7).integers(0, 4444, 16)]
    rng = np.random.default_rng(11 + kidx)
    w = redlib.copie(S1)
    if os.environ.get("TOUR78_ECHELLE"):
        # tour 71 : l'etat de base a ses moments de lr 0,05 ; m x lr/0,05 et v x (lr/0,05)^2 evitent le coup de depart
        w["m"] = w["m"] * (lr / 0.05)
        w["v"] = w["v"] * (lr / 0.05) ** 2
    redlib.avancer(w, float(PLI - mpf("1e-6")), 20000 + ks[kidx], eps=eps, cross=2.0, stop=False)
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
    if os.environ.get("TOUR83_DETAIL"):
        # tour 83 : degre de fusion des salves dans l'enregistrement (groupes d'amplitude E > 3e-4 coupes a 30 pas de calme)
        dur_g = np.array([b_ - a_ + 1 for a_, b_ in groupes])
        calme_g = np.array([groupes[i][0] - groupes[i - 1][1] for i in range(1, len(groupes))]) if len(groupes) > 1 else np.array([0])
        print(f"   groupes : N = {len(groupes)} ; duree mediane {np.median(dur_g):.0f} (p10 {np.percentile(dur_g, 10):.0f}, p90 {np.percentile(dur_g, 90):.0f}) ; calme median {np.median(calme_g):.0f} (p10 {np.percentile(calme_g, 10):.0f}, p90 {np.percentile(calme_g, 90):.0f}) ; fraction en oscillation {dur_g.sum() / len(E):.3f}")
    if os.environ.get("TOUR82_CALIB") and len(groupes) > 2:
        # tour 82 : l'interpolation S_c est construite avec le jacobien d'un seul etat (apres la mise en regime), parfois mal place (max 36,9 ou 38,1 contre un seuil 38) ;
        # on la recale par un facteur multiplicatif pour que S_c(t_cal) soit le niveau exact du bloc complet, t_cal = fin de la deuxieme salve + 150 pas
        t_cal = groupes[1][1] + 150
        w1c = redlib.copie(w0)
        redlib.avancer(w1c, delta, t_cal + 2, eps=eps, cross=2.0, stop=False)
        J6c = jac6([mpf(float(q)) for q in w1c["x"]], mpf(delta))
        Jc = np.array([[float(J6c[i, j]) for j in range(6)] for i in range(6)])
        facteur = bloc_sym(Jc, w1c["v"], eps, lr)["S_max"] / Sc[t_cal + 1]
        print(f"   calibration S_c : facteur {facteur:.5f}")
        Sc = Sc * facteur
    res = []
    robuste = bool(os.environ.get("TOUR84_ROBUSTE"))   # tour 84 : lecture directe par balayage exact avant le debut de chaque groupe, sans interpolation S_c
    pas_bal = int(os.environ.get("TOUR84_PAS", "1"))
    nsal = int(os.environ.get("TOUR85_NSAL", "3"))   # tour 85 : nombre de salves lues par depart (pour mesurer l'ecart de salve a salve)
    g0 = int(os.environ.get("TOUR86_G0", "1"))   # tour 86 : premier groupe lu (pour relire des salves tardives de l'enregistrement)
    gl = os.environ.get("TOUR86_GLIST")   # tour 86 : liste explicite de groupes a lire (echantillon aleatoire de salves)
    liste_g = [int(q) for q in gl.split(",")] if gl else range(g0, min(len(groupes), g0 - 1 + (max(12, 3 * nsal + 3) if robuste else 4)))
    for g in liste_g:
        if g >= len(groupes):
            continue
        if robuste and len(res) >= nsal:
            break
        prev_end = groupes[g - 1][1]; a = groupes[g][0]
        if robuste:
            if a - 140 < 1:
                continue
            t_c = a - 70
        else:
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
            fen = int(os.environ.get("TOUR81_FEN", "14"))   # demi-fenetre du balayage (tour 81 : 30 pour retrouver le franchissement des departs ks[1], ks[2])
            if robuste:
                fen = 70
            redlib.avancer(w1, delta, max(t_c - fen, 1), eps=eps, cross=2.0, stop=False)
            suite = []
            for _ in range((2 * fen) // pas_bal + 1):
                redlib.avancer(w1, delta, pas_bal, eps=eps, cross=2.0, stop=False)
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
            i = ok[-1] if os.environ.get("TOUR87_DERNIER") else ok[0]   # tour 87 : dernier franchissement montant dans la fenetre (le plus proche du debut du groupe) au lieu du premier
            fr = (thr - suite[i]["S_max"]) / (suite[i + 1]["S_max"] - suite[i]["S_max"])
            bl = {}
            for k in ("S_max", "d_gap", "d_s", "c", "part_gap", "part_s", "part_autre", "ferm", "sv_e3", "sv_l3", "pz", "niv0"):
                bl[k] = (1 - fr) * suite[i][k] + fr * suite[i + 1][k]
            bl["niveaux"] = np.array([bl["niv0"], bl["S_max"]])
            res.append(bl)
            if os.environ.get("TOUR83_DETAIL"):
                print(f"   salve {g} : calme avant {a - prev_end} pas ; duree {groupes[g][1] - a + 1} pas ; lr p_z {lr * bl['pz']:.5e} ; sqrt(v_e3) {bl['sv_e3']:.4e} ; avance {a - (max(t_c - fen, 1) + (i + 1 + fr) * pas_bal):.1f} pas")   # tour 87 : avance = debut du groupe moins instant (interpole) du franchissement du niveau
    if not res:
        print(f"{nom} lr = {lr} eps = {eps:.1e} x = {xs} : aucun franchissement (S_c interpole : max {Sc.max():.3f}, seuil {thr:.0f}, {len(groupes)} groupes d'amplitude)")
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
