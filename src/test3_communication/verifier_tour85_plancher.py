"""Tour 85 : plancher de lecture mesure (ecart de salve a salve a l'interieur d'un depart) a x = 1e-4, et son effet sur le test de segments.

Il dit : le plancher peut etre mesure au lieu d'etre choisi ; pari : l'ecart de salve a salve a lr 0,1 et 0,15 est sous 0,05 %, donc le test (bas 0,1-0,2, haut 0,2-0,3) reste a 3 sigma ou plus.

Donnees : journaux D:/tmp/t85_x1e-4_lr<lr>_k<k>.log (huit lectures par depart, lecture directe, pas 3), + t85_..._pas1.log pour la methode, + les moyennes des tours 82 et 84 pour comparaison.
Deux incertitudes par lr : ET entre departs (sd des moyennes de depart / sqrt(5)) et ET de lecture sigma_w / sqrt(n_lectures x n_etats_distincts) (sigma_w = ecart-type poole a l'interieur des departs).

Usage : python verifier_tour85_plancher.py
"""

import glob
import math
import re
import sys

import numpy as np

sys.path.insert(0, ".")
from verifier_tour84_coupures import signif_segments, droite, sel, lire_pz  # noqa: E402

D = "D:/tmp"
LRS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


def lectures(chemin):
    t = open(chemin, encoding="utf-8", errors="replace").read()
    return [float(a) for a in re.findall(r"salve \d+ : calme avant \d+ pas ; duree \d+ pas ; lr p_z ([0-9.e+]+)", t)]


if __name__ == "__main__":
    print("x = 1e-4, huit lectures par depart : etendue (max/min - 1) par depart, ecart-type poole sigma_w, moyennes de depart, etats distincts (0,02 %)")
    res = {}
    for lr in LRS:
        starts = {}
        for k in range(5):
            try:
                L = lectures(f"{D}/t85_x1e-4_lr{lr}_k{k}.log")
            except FileNotFoundError:
                continue
            if len(L) >= 2:
                starts[k] = np.array(L)
        if not starts:
            continue
        etendues = [100 * (v.max() / v.min() - 1) for v in starts.values()]
        sds = [v.std(ddof=1) / v.mean() for v in starts.values()]
        sw = math.sqrt(np.mean(np.square(sds)))
        mu = np.array([v.mean() for v in starts.values()])
        mm = mu.mean()
        sorted_mu = sorted(mu)
        distinct = 1 + sum(1 for a_, b_ in zip(sorted_mu[:-1], sorted_mu[1:]) if abs(b_ / a_ - 1) > 2e-4)
        nw = int(np.median([len(v) for v in starts.values()]))
        se_between = mu.std(ddof=1) / math.sqrt(len(mu)) / mm if len(mu) > 1 else 0.0
        se_w = sw / math.sqrt(nw * distinct)
        se_tot = math.hypot(se_between, se_w)
        res[lr] = (mm, se_between, se_w, se_tot)
        print(f"  lr {lr:5}: {len(starts)} departs, {nw} lectures ; etendue par depart (%) : " + " ".join(f"{e:.3f}" for e in etendues)
              + f" ; sigma_w {100 * sw:.3f} % ; moyenne {mm:.5e} ; ET entre departs {100 * se_between:.3f} % ; etats distincts {distinct} ; ET de lecture {100 * se_w:.3f} % ; ET totale {100 * se_tot:.3f} %")
    # methode
    for nom, f3, f1 in (("lr 0,15 ks[0]", f"{D}/t85_x1e-4_lr0.15_k0.log", f"{D}/t85_x1e-4_lr0.15_k0_pas1.log"),):
        try:
            a, b = lectures(f3), lectures(f1)
            print(f"methode {nom} : pas 3 : {np.mean(a):.5e} ({len(a)} lectures) ; pas 1 : {np.mean(b):.5e} ({len(b)} lectures) ; ecart {100 * (np.mean(b) / np.mean(a) - 1):+.3f} %")
        except FileNotFoundError:
            pass
    print("comparaison avec les moyennes des tours 82/84 (3 lectures, S_c interpole ou lecture directe) :")
    for lr in LRS:
        anc = []
        for modele in (f"{D}/t82_x1e-4_lr{lr}_k{{k}}.log", f"{D}/t84_x1e-4_lr{lr}_k{{k}}.log"):
            vals = []
            for k in range(5):
                try:
                    v = lire_pz(modele.format(k=k))
                except FileNotFoundError:
                    v = None
                if v is not None:
                    vals.append(v)
            if vals:
                anc.append(np.mean(vals))
        if lr in res and anc:
            print(f"  lr {lr:5}: tour 85 {res[lr][0]:.5e} ; anciennes " + " ; ".join(f"{a:.5e} ({100 * (res[lr][0] / a - 1):+.3f} %)" for a in anc))
    # test de segments
    if all(l in res for l in (0.1, 0.15, 0.2, 0.25, 0.3)):
        lrs = np.array(sorted(res))
        m = np.array([res[l][0] for l in lrs])
        for nom, idx in (("ET entre departs seule", 1), ("ET de lecture seule", 2), ("ET totale (quadrature)", 3)):
            se = np.array([res[l][idx] for l in lrs]) * m
            r = signif_segments(lrs, m, se, bas=(0.1, 0.2), haut=(0.2, 0.3))
            print(f"test de segments (bas 0,1-0,2 / haut 0,2-0,3), {nom} : bas {r[0]:+.4f} +- {r[1]:.4f}, haut {r[2]:+.4f} +- {r[3]:.4f} ; {r[4]:.1f} sigma ; haut au-dessus de plat : {r[2] / r[3]:.1f} ET")
        # plancher uniforme des tours precedents, pour comparaison
        for fl in (0.0, 0.1, 0.2):
            se = np.array([math.hypot(res[l][1], fl / 100) for l in lrs]) * m
            r = signif_segments(lrs, m, se, bas=(0.1, 0.2), haut=(0.2, 0.3))
            print(f"   (plancher uniforme {fl:.1f} % ajoute a l'ET entre departs : {r[4]:.1f} sigma)")
        # sensibilite (post hoc, signalee comme telle) : sans lr 0,3 (segment haut 0,2-0,25, deux points) ; et en ecartant les lectures dont le groupe dure plus de 300 pas
        se = np.array([res[l][3] for l in lrs]) * m
        r = signif_segments(lrs, m, se, bas=(0.1, 0.2), haut=(0.2, 0.25))
        print(f"sensibilite sans lr 0,3 (haut 0,2-0,25, deux points), ET totale : bas {r[0]:+.4f}, haut {r[2]:+.4f} +- {r[3]:.4f} ; {r[4]:.1f} sigma ; haut au-dessus de plat {r[2] / r[3]:.1f} ET")
        res2 = {}
        retire = 0
        for lr in LRS:
            mus = []
            for k in range(5):
                try:
                    t = open(f"{D}/t85_x1e-4_lr{lr}_k{k}.log", encoding="utf-8", errors="replace").read()
                except FileNotFoundError:
                    continue
                L = [(float(a), int(b)) for b, a in re.findall(r"duree (\d+) pas ; lr p_z ([0-9.e+]+)", t)]
                keep = [p for p, dur in L if dur < 300]
                retire += len(L) - len(keep)
                if len(keep) >= 2:
                    mus.append(np.mean(keep))
            mus = np.array(mus)
            res2[lr] = (mus.mean(), mus.std(ddof=1) / math.sqrt(len(mus)) / mus.mean())
        lrs2 = np.array(sorted(res2)); m2 = np.array([res2[l][0] for l in lrs2]); se2 = np.array([res2[l][1] for l in lrs2]) * m2
        r = signif_segments(lrs2, m2, se2, bas=(0.1, 0.2), haut=(0.2, 0.3))
        print(f"sensibilite sans les lectures de groupes de plus de 300 pas ({retire} lectures retirees) : lr 0,3 moyenne {res2[0.3][0]:.5e}, ET entre departs {100 * res2[0.3][1]:.3f} % ; "
              f"bas {r[0]:+.4f}, haut {r[2]:+.4f} +- {r[3]:.4f} ; {r[4]:.1f} sigma ; haut au-dessus de plat {r[2] / r[3]:.1f} ET")
