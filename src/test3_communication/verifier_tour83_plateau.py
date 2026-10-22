"""Tour 83 : (1) verification de ses chiffres sur les neuf moyennes (plateau 0,175 / 0,2 / 0,25, ecart de 0,3, loi unique avec et sans 0,3 sur sd arrondis) ;
(2) lecture des journaux t83_x<x>_lr<lr>_k<k>.log : nombre de groupes, calme, duree, lectures par salve, lr p_z par depart, correlation entre lr p_z et nombre de groupes.

Usage : python verifier_tour83_plateau.py plateau
        python verifier_tour83_plateau.py groupes <x> <lr> [<lr> ...]
"""

import glob
import math
import re
import sys

import numpy as np


def ajuste(x, y, s):
    w = 1 / s ** 2
    W = w.sum(); xb = (w * x).sum() / W; yb = (w * y).sum() / W
    sxx = (w * (x - xb) ** 2).sum()
    e = (w * (x - xb) * (y - yb)).sum() / sxx
    a = yb - e * xb
    return e, 1 / math.sqrt(sxx), float((w * (y - a - e * x) ** 2).sum())


def plateau():
    lrs = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3]
    m = np.array([2.27240, 2.28798, 2.30458, 2.31778, 2.33220, 2.34110, 2.34002, 2.33630, 2.35068]) * 1e6
    sd = np.array([0.03, 0.04, 0.09, 0.19, 0.29, 0.32, 0.41, 0.19, 0.31]) / 100   # sd arrondis, relatifs
    se = sd * m / math.sqrt(5)
    i = [5, 6, 7]
    w = 1 / se[i] ** 2
    niv = (w * m[i]).sum() / w.sum(); chi2 = float((w * (m[i] - niv) ** 2).sum()); sn = 1 / math.sqrt(w.sum())
    print(f"plateau 0,175 / 0,2 / 0,25 : moyenne ponderee {niv:.5e} ; chi2 {chi2:.2f} sur 2 dof ; erreur-type {100 * sn / niv:.3f} %")
    ecart = m[8] / niv - 1
    print(f"lr 0,3 : {100 * ecart:+.2f} % au-dessus du plateau ; {(m[8] - niv) / math.hypot(se[8], sn):.1f} erreurs-types")
    x = np.log(lrs); y = np.log(m); s = se / m
    for nom, sel in (("avec lr 0,3 (9 points)", slice(0, 9)), ("sans lr 0,3 (8 points)", slice(0, 8))):
        e, de, chi = ajuste(x[sel], y[sel], s[sel])
        n = len(x[sel])
        print(f"loi unique {nom} : exposant {e:.4f} +- {de:.4f} ; chi2 {chi:.1f} sur {n - 2} dof")
    # moyennes 1e-4
    print(f"x = 1e-4 : 0,25 -> 0,3 : {100 * (3.30380 / 3.29653 - 1):+.2f} % ; x = 1e-6 : {100 * (2.35068 / 2.33630 - 1):+.2f} %")


def groupes(xs, lrs):
    print("lr      depart  lectures  lr p_z       groupes  duree med  calme med  fraction en osc.")
    for lr in lrs:
        tab = []
        for k in range(5):
            fs = glob.glob(f"D:/tmp/t83_x{xs}_lr{lr}_k{k}.log")
            if not fs:
                continue
            t = open(fs[0], encoding="utf-8", errors="replace").read()
            g = re.search(r"groupes : N = (\d+) ; duree mediane (\d+).*?calme median (\d+).*?fraction en oscillation ([0-9.]+)", t)
            sal = re.findall(r"salve (\d+) : calme avant (\d+) pas ; duree (\d+) pas ; lr p_z ([0-9.e+]+)", t)
            pzs = [float(a[3]) for a in sal]
            m = re.search(r"\((\d) salves\)", t)
            tab.append((k, len(pzs), np.mean(pzs) if pzs else float("nan"), int(g.group(1)) if g else -1, int(g.group(2)) if g else -1, int(g.group(3)) if g else -1, float(g.group(4)) if g else float("nan"), pzs))
            print(f"{lr:6} k{k}      {len(pzs)}         {tab[-1][2]:.5e}  {tab[-1][3]:6d}   {tab[-1][4]:6d}    {tab[-1][5]:6d}     {tab[-1][6]:.3f}   lectures: " + " ".join(f"{p:.5e}" for p in pzs))
        ok = [t_ for t_ in tab if not math.isnan(t_[2])]
        if len(ok) >= 3:
            c = np.corrcoef([t_[2] for t_ in ok], [t_[3] for t_ in ok])[0, 1]
            print(f"   lr {lr} : moyenne lr p_z {np.mean([t_[2] for t_ in ok]):.5e} ; groupes moyens {np.mean([t_[3] for t_ in ok]):.0f} ; correlation lr p_z / nombre de groupes {c:+.2f} (n = {len(ok)})")


if __name__ == "__main__":
    if sys.argv[1] == "plateau":
        plateau()
    else:
        groupes(sys.argv[2], sys.argv[3:])
