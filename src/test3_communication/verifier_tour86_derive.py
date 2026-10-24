"""Tour 86 : la derive des lectures le long des salves, et le test de segments a k fixe.

Il dit : la derive est un biais, pas un plancher ; elle ne bouge l'exposant que si elle differe selon lr. Par depart, ajuster lecture = a + b k sur les huit salves, prendre la valeur a k fixe (meme k a chaque lr), ET d'apres la dispersion sur les cinq departs, puis le test de segments a k = 1 et k = 8.

Donnees : journaux D:/tmp/t85_x1e-4_lr<lr>_k<k>.log (huit lectures par depart, avec duree, calme, sqrt(v_e3)).
Sorties : (1) chiffres arithmetiques de sa lettre, dont le poids du point 0,3 dans mon ajustement pondere ; (2) pentes b par depart et par lr (MCO, Theil-Sen, sans groupes longs) ; (3) lectures a k = 1, 4,5 (moyenne), 8 ; (4) test de segments a ces k, ET de lui (dispersion sur cinq departs) et avec plancher ; (5) pente de ln(lecture) contre ln(sqrt(v_e3)) a l'interieur des departs ; (6) pente par 1000 pas et par lr x 1000 pas.

Usage : python verifier_tour86_derive.py
"""

import math
import re
import sys

import numpy as np

sys.path.insert(0, ".")
from verifier_tour84_coupures import signif_segments, droite  # noqa: E402

D = "D:/tmp"
LRS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
MOTIF = re.compile(r"salve (\d+) : calme avant (\d+) pas ; duree (\d+) pas ; lr p_z ([0-9.e+]+) ; sqrt\(v_e3\) ([0-9.e+\-]+)")


def charger(lr, k):
    """Liste de (indice de salve, calme, duree, lecture, sqrt(v_e3)) dans l'ordre du journal, ou None."""
    try:
        t = open(f"{D}/t85_x1e-4_lr{lr}_k{k}.log", encoding="utf-8", errors="replace").read()
    except FileNotFoundError:
        return None
    return [(int(a), int(b), int(c), float(d), float(e)) for a, b, c, d, e in MOTIF.findall(t)]


def theil_sen(x, y):
    p = [(y[j] - y[i]) / (x[j] - x[i]) for i in range(len(x)) for j in range(i + 1, len(x))]
    return float(np.median(p))


def mco(x, y):
    xb, yb = x.mean(), y.mean()
    b = ((x - xb) * (y - yb)).sum() / ((x - xb) ** 2).sum()
    return b, yb - b * xb


def arithmetique():
    x = np.log([0.2, 0.25, 0.3])
    w = (x - x.mean()) / ((x - x.mean()) ** 2).sum()
    print(f"poids de chaque point dans la pente MCO non ponderee (0,2 ; 0,25 ; 0,3) : {np.round(w, 3)} (lui : 2,38 pour le point 0,3)")
    print(f"poids dans la pente ponderee avec les ET totales du tour 85 (0,071 ; 0,088 ; 0,240 %) : {np.round(poids_ponderes([0.00071, 0.00088, 0.0024]), 3)}")
    print(f"  0,3 a 0,5 % sur 3 departs sur 5 : moyenne {3 / 5 * 0.3:.2f} a {3 / 5 * 0.5:.2f} % ; deplacement de l'exposant {w[2] * 0.0018:.4f} a {w[2] * 0.0030:.4f} ; 0,0172 - 0,0107 = {0.0172 - 0.0107:.4f}")


def poids_ponderes(se_rel):
    """Poids du point 0,3 dans la pente ponderee (0,2 ; 0,25 ; 0,3) pour des ET relatifs donnes."""
    x = np.log([0.2, 0.25, 0.3])
    wt = 1 / np.array(se_rel) ** 2
    xb = (wt * x).sum() / wt.sum()
    sxx = (wt * (x - xb) ** 2).sum()
    return wt * (x - xb) / sxx


if __name__ == "__main__":
    arithmetique()
    data = {}
    for lr in LRS:
        data[lr] = {k: charger(lr, k) for k in range(5)}
        data[lr] = {k: v for k, v in data[lr].items() if v}
    print()
    print("pente b de lecture = a + b k, par depart (% de la moyenne du depart, par salve) : MCO | Theil-Sen | MCO sans lectures de groupes > 300 pas ; puis lecture brute de la salve 1 et de la salve 8 (ecart relatif de 8 a 1, %)")
    sommaire = {}
    for lr in LRS:
        lignes = []
        bs, bts, bsg = [], [], []
        for k, L in data[lr].items():
            R = np.array([e[3] for e in L]); n = len(R); kk = np.arange(1, n + 1, dtype=float)
            dur = np.array([e[2] for e in L])
            b, a = mco(kk, R)
            bt = theil_sen(kk, R)
            ok = dur < 300
            bg = mco(kk[ok], R[ok])[0] if ok.sum() >= 4 else float("nan")
            m = R.mean()
            bs.append(100 * b / m); bts.append(100 * bt / m); bsg.append(100 * bg / m)
            lignes.append(f"    k{k}: n {n} b {100 * b / m:+.4f} | TS {100 * bt / m:+.4f} | sans longs {100 * bg / m:+.4f} (retire {int((~ok).sum())}) ; salve 1 {R[0]:.5e}, salve 8 {R[-1]:.5e} ({100 * (R[-1] / R[0] - 1):+.3f} %) ; durees {' '.join(str(d) for d in dur)}")
        bs = np.array(bs); bts = np.array(bts); bsg = np.array(bsg)
        sommaire[lr] = (bs, bts, bsg)
        print(f"lr {lr}: b MCO moyenne {bs.mean():+.4f} %/salve (sd entre departs {bs.std(ddof=1):.4f}, {int((bs > 0).sum())} departs > 0 sur {len(bs)}) ; Theil-Sen {bts.mean():+.4f} ; sans longs {np.nanmean(bsg):+.4f}")
        for s in lignes:
            print(s)
    print()
    # valeurs a k fixe
    valeurs = {}
    for lr in LRS:
        v1, vm, v8, r1, r8 = [], [], [], [], []
        for k, L in data[lr].items():
            R = np.array([e[3] for e in L]); n = len(R); kk = np.arange(1, n + 1, dtype=float)
            b, a = mco(kk, R)
            v1.append(a + b * 1); vm.append(R.mean()); v8.append(a + b * n)
            r1.append(R[0]); r8.append(R[-1])
        valeurs[lr] = {"k=1 ajuste": np.array(v1), "moyenne": np.array(vm), "k=8 ajuste": np.array(v8), "salve 1 brute": np.array(r1), "salve 8 brute": np.array(r8)}
    lrs = np.array(LRS[1:])
    for nom in ("salve 1 brute", "k=1 ajuste", "moyenne", "k=8 ajuste", "salve 8 brute"):
        m = np.array([valeurs[l][nom].mean() for l in lrs])
        se_b = np.array([valeurs[l][nom].std(ddof=1) / math.sqrt(len(valeurs[l][nom])) for l in lrs])
        print(f"== lecture : {nom} ; moyenne par lr {' '.join(f'{v:.5e}' for v in m)} ; ET entre departs (%) {' '.join(f'{100 * s / v:.3f}' for s, v in zip(se_b, m))}")
        for etiq, se in (("ET entre departs seule (la sienne)", se_b), ("ET entre departs + plancher 0,1 %", np.hypot(se_b, 0.001 * m)), ("ET entre departs + plancher 0,2 %", np.hypot(se_b, 0.002 * m))):
            r = signif_segments(lrs, m, se, bas=(0.1, 0.2), haut=(0.2, 0.3))
            print(f"   {etiq} : bas {r[0]:+.4f} +- {r[1]:.4f}, haut {r[2]:+.4f} +- {r[3]:.4f} ; difference {r[0] - r[2]:+.4f} ; {r[4]:.1f} sigma")
        # poids du point 0,3 dans l'ajustement pondere
        wts = poids_ponderes(se_b[2:] / m[2:])
        print(f"   poids des points 0,2 / 0,25 / 0,3 dans la pente haute ponderee : {np.round(wts, 3)}")
    print()
    # pente de ln(lecture) contre ln(sqrt(v_e3)) a l'interieur des departs
    print("pente de ln(lecture) contre ln(sqrt(v_e3)) a l'interieur des departs (moyenne centree par depart), par lr :")
    for lr in LRS:
        xs, ys = [], []
        for k, L in data[lr].items():
            R = np.log([e[3] for e in L]); S = np.log([e[4] for e in L])
            xs.extend(S - S.mean()); ys.extend(R - R.mean())
        xs, ys = np.array(xs), np.array(ys)
        pente = (xs * ys).sum() / (xs ** 2).sum()
        res = ys - pente * xs
        et = math.sqrt((res ** 2).sum() / (len(xs) - 1) / (xs ** 2).sum())
        print(f"  lr {lr}: pente {pente:+.3f} +- {et:.3f} (n {len(xs)}) ; amplitude de ln(sqrt(v_e3)) : sd {xs.std() * 100:.4f} %")
    print()
    # temps
    print("derive par 1000 pas et par lr x 1000 pas (lecture contre pas cumules depuis la premiere salve lue), MCO, moyenne sur les departs (% de la moyenne du depart) :")
    for lr in LRS:
        p1, p2 = [], []
        for k, L in data[lr].items():
            R = np.array([e[3] for e in L])
            t = np.cumsum([e[1] + e[2] for e in L]).astype(float)
            b = mco(t, R)[0]
            p1.append(100 * b * 1000 / R.mean()); p2.append(100 * b * 1000 / R.mean() / lr)
        print(f"  lr {lr}: {np.mean(p1):+.4f} % par 1000 pas ; {np.mean(p2):+.4f} % par (lr x 1000 pas) ; sd entre departs {np.std(p1, ddof=1):.4f}")
