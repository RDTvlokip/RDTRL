"""Tour 82 : ajustements du gain lr p_z en fonction de lr a partir de departs multiples (journaux de verifier_tour78_lr.py reduction).

Il propose : sur les six moyennes (28 departs, x = 1e-6) une seule loi de puissance (exposant 0,0206, chi2 21,3 sur 4 dof) est rejetee ; segments disjoints 0,0291 +- 0,0029 (0,1 a 0,15) et 0,0082 +- 0,0036 (0,175 a 0,3) ;
loi brisee continue a coude balaye : chi2 0,8 (0,16 a 0,175), 1,9 (0,15), 2,5 (0,19), 4,3 (0,2).
Ici : moyennes, erreurs-types sd/sqrt(n) (ddof = 1), ajustements ponderes dans l'espace log.

Usage : python verifier_tour82_ajustements.py <gabarit1[,gabarit2,...]> <lr1> <lr2> ...
  gabarit : chemin avec {lr} et * pour les departs, ex. "D:/tmp/t81_lr{lr}_k*.log,D:/tmp/t82_x1e-6_lr{lr}_k*.log"
"""

import glob
import math
import re
import sys

import numpy as np


def lire(gabarits, lr):
    vals = {}
    for g in gabarits:
        for f in sorted(glob.glob(g.format(lr=lr))):
            t = open(f, encoding="utf-8", errors="replace").read()
            m = re.search(r"lr p_z = ([0-9.e+\-]+)", t)
            d = re.search(r"d_s/niveau = ([0-9.]+)", t)
            r = re.search(r"rapport ([0-9.]+)", t)
            if m and d and r:
                vals[f] = (float(m.group(1)), float(d.group(1)), float(r.group(1)))
    return list(vals.values())


def ajuste_droite(x, y, s):
    w = 1 / s ** 2
    W = w.sum(); xb = (w * x).sum() / W; yb = (w * y).sum() / W
    sxx = (w * (x - xb) ** 2).sum()
    e = (w * (x - xb) * (y - yb)).sum() / sxx
    a = yb - e * xb
    chi2 = (w * (y - a - e * x) ** 2).sum()
    return e, 1 / math.sqrt(sxx), a, chi2


def brisee(x, y, s, xk):
    """loi brisee continue en xk : y = a + e1 (x - xk) pour x < xk, a + e2 (x - xk) sinon ; moindres carres ponderes lineaires en (a, e1, e2)."""
    w = 1 / s
    A = np.column_stack([np.ones_like(x), np.where(x < xk, x - xk, 0.0), np.where(x >= xk, x - xk, 0.0)])
    c, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
    chi2 = float((((y - A @ c) * w) ** 2).sum())
    return c, chi2


if __name__ == "__main__":
    gabarits = sys.argv[1].split(",")
    lrs = [float(a) for a in sys.argv[2:]]
    xs, ms, ses, ns = [], [], [], []
    print("lr      n   moyenne lr p_z   sd entre departs   erreur-type (sd/sqrt n)")
    for lr in lrs:
        v = lire(gabarits, lr)
        if len(v) < 2:
            print(f"{lr:6} {len(v)}  (insuffisant)")
            continue
        pz = np.array([a[0] for a in v])
        xs.append(math.log(lr)); ms.append(pz.mean()); ses.append(pz.std(ddof=1) / math.sqrt(len(pz))); ns.append(len(pz))
        print(f"{lr:6} {len(pz)}   {pz.mean():.5e}   {100 * pz.std(ddof=1) / pz.mean():.2f} %          {100 * ses[-1] / pz.mean():.3f} %")
    x = np.array(xs); y = np.log(ms); s = np.array(ses) / np.array(ms)
    e, de, a, chi2 = ajuste_droite(x, y, s)
    print(f"une seule loi de puissance : exposant {e:.4f} +- {de:.4f} ; chi2 {chi2:.1f} sur {len(x) - 2} dof")
    if len(x) >= 6:
        lo = slice(0, 3); hi = slice(3, 6)
        e1, d1, _, c1 = ajuste_droite(x[lo], y[lo], s[lo]); e2, d2, _, c2 = ajuste_droite(x[hi], y[hi], s[hi])
        print(f"segments disjoints (3 premiers / 3 derniers points) : {e1:.4f} +- {d1:.4f} et {e2:.4f} +- {d2:.4f} ; ecart {abs(e1 - e2) / math.hypot(d1, d2):.1f} erreurs-types")
    print("loi brisee continue, coude balaye (chi2 sur n-4 dof) :")
    best = None
    for k in np.arange(0.11, max(lrs) * 0.97, 0.005):
        c, ch = brisee(x, y, s, math.log(k))
        if best is None or ch < best[0]:
            best = (ch, k, c)
    for k in (0.125, 0.15, 0.16, 0.175, 0.19, 0.2, 0.25):
        if min(lrs) < k < max(lrs):
            c, ch = brisee(x, y, s, math.log(k))
            print(f"   coude {k:.3f} : chi2 {ch:.2f} ; exposants {c[1]:.4f} puis {c[2]:.4f}")
    print(f"   minimum : coude {best[1]:.3f}, chi2 {best[0]:.2f}, exposants {best[2][1]:.4f} puis {best[2][2]:.4f}")
    # predictions a lr 0,25 de la loi brisee optimale et d'une seule loi a partir de lr 0,175 (si ces lr existent)
    if 0.175 in lrs and 0.25 not in lrs:
        i = lrs.index(0.175)
        print(f"lecture a lr 0,25 : une seule loi (exposant {e:.4f}) ancree a lr 0,175 : {ms[i] * (0.25 / 0.175) ** e:.4e} ; loi brisee optimale : {math.exp(best[2][0] + best[2][2] * (math.log(0.25) - math.log(best[1]))):.4e}")
