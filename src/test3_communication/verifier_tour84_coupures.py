"""Tour 84 : le ralentissement du gain lr p_z entre lr 0,15 et 0,25 survit-il a la coupure, aux departs, au retrait d'un lr et a un plancher d'erreur systematique ?

Il dit : garder 0,25 (ecarter seulement 0,3) ; chi2 de la loi unique 64,8 sur 5 dof (0,05-0,2) et 48,6 sur 3 (0,05-0,15) ; a x = 1e-4 la pente 0,15 -> 0,2 prolongee a 0,25 tombe 0,35 % au-dessus
de la mesure (3,0 ET), a x = 1e-6 0,42 % (2,1 ET) ; 1e-4 a 0,25 est n = 3.

Donnees : journaux D:/tmp de verifier_tour78_lr.py reduction (t81_lr*, t82_x*, t83_x*, t84_x*). Moyenne par lr sur les departs, ET = sd/sqrt(n), ajustements dans l'espace log.
Un plancher systematique f (en pourcent de la moyenne) est ajoute en quadrature a chaque erreur-type.

Usage : python verifier_tour84_coupures.py
"""

import math
import re

import numpy as np

D = "D:/tmp"


def lire_pz(chemin):
    t = open(chemin, encoding="utf-8", errors="replace").read()
    m = re.search(r"lr p_z = ([0-9.e+\-]+)", t)
    return float(m.group(1)) if m and "aucun franchissement" not in t else None


def charger(x):
    """dict lr -> {depart: lr p_z} ; x en '1e-6' ou '1e-4'."""
    d = {}
    if x == "1e-6":
        modeles = [f"{D}/t82_x1e-6_lr{{lr}}_k{{k}}.log", f"{D}/t83_x1e-6_lr{{lr}}_k{{k}}.log", f"{D}/t81_lr{{lr}}_k{{k}}.log"]
        lrs = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.35]
        pref = {0.05: 0, 0.075: 0, 0.25: 0, 0.225: 1, 0.275: 1, 0.35: 1}
        defaut = 2
    else:
        modeles = [f"{D}/t82_x1e-4_lr{{lr}}_k{{k}}.log", f"{D}/t84_x1e-4_lr{{lr}}_k{{k}}.log"]
        lrs = [0.05, 0.1, 0.15, 0.2, 0.225, 0.25, 0.275, 0.3, 0.35, 0.4]
        pref = {0.225: 1, 0.25: 1, 0.275: 1, 0.35: 1, 0.4: 1}
        defaut = 0
    for lr in lrs:
        modele = modeles[pref.get(lr, defaut)]
        vals = {}
        for k in range(5):
            try:
                v = lire_pz(modele.format(lr=lr, k=k))
            except FileNotFoundError:
                v = None
            if v is not None:
                vals[k] = v
        if vals:
            d[lr] = vals
    return d


def stats(d, retire_depart=None, floor=0.0):
    lrs, m, se, n = [], [], [], []
    for lr in sorted(d):
        v = [p for k, p in d[lr].items() if k != retire_depart]
        if len(v) < 2:
            continue
        v = np.array(v)
        lrs.append(lr); m.append(v.mean()); n.append(len(v))
        s = v.std(ddof=1) / math.sqrt(len(v))
        se.append(math.hypot(s, floor * v.mean() / 100))
    return np.array(lrs), np.array(m), np.array(se), np.array(n)


def droite(x, y, s):
    w = 1 / s ** 2
    W = w.sum(); xb = (w * x).sum() / W; yb = (w * y).sum() / W
    sxx = (w * (x - xb) ** 2).sum()
    e = (w * (x - xb) * (y - yb)).sum() / sxx
    a = yb - e * xb
    return e, 1 / math.sqrt(sxx), float((w * (y - a - e * x) ** 2).sum())


def sel(lrs, lo, hi):
    return [i for i, l in enumerate(lrs) if lo - 1e-9 <= l <= hi + 1e-9]


def signif_segments(lrs, m, se, bas=(0.1, 0.15), haut=(0.175, 0.25), retire_lr=None):
    ib = sel(lrs, *bas); ih = [i for i in sel(lrs, *haut) if lrs[i] != retire_lr]
    if len(ib) < 2 or len(ih) < 2:
        return None
    x = np.log(lrs); y = np.log(m); s = se / m
    eb, db, _ = droite(x[ib], y[ib], s[ib]); eh, dh, _ = droite(x[ih], y[ih], s[ih])
    return eb, db, eh, dh, abs(eb - eh) / math.hypot(db, dh)


def chute_deux_pas(lrs, m, se, a, b, c):
    """e(a,b) - e(b,c) et son ET (trois moyennes independantes), plus l'ecart de la valeur en c a la prolongation de la pente (a,b)."""
    lst = list(lrs)
    ia, ib, ic = [lst.index(v) for v in (a, b, c)]
    la, lb, lc = math.log(a), math.log(b), math.log(c)
    ya, yb, yc = [math.log(m[i]) for i in (ia, ib, ic)]
    sa, sb, sc = [se[i] / m[i] for i in (ia, ib, ic)]
    e1 = (yb - ya) / (lb - la); e2 = (yc - yb) / (lc - lb)
    w_a = -1 / (lb - la); w_b = 1 / (lb - la) + 1 / (lc - lb); w_c = -1 / (lc - lb)
    et = math.sqrt((w_a * sa) ** 2 + (w_b * sb) ** 2 + (w_c * sc) ** 2)
    s = (lc - lb) / (lb - la)
    pred = (1 + s) * yb - s * ya
    et_pred = math.sqrt(((1 + s) * sb) ** 2 + (s * sa) ** 2 + sc ** 2)
    return e1, e2, e1 - e2, et, 100 * (math.exp(yc - pred) - 1), et_pred * 100, (pred - yc) / et_pred


if __name__ == "__main__":
    for x in ("1e-6", "1e-4"):
        d = charger(x)
        print(f"===== x = {x} : lr disponibles et nombre de departs : " + ", ".join(f"{lr}:{len(d[lr])}" for lr in sorted(d)))
        lrs, m, se, n = stats(d)
        print("   lr      n   moyenne lr p_z   ET (%)  valeurs distinctes (a 0,02 %)")
        for lr, mm, ss, nn in zip(lrs, m, se, n):
            v = sorted(d[lr].values())
            distinct = 1 + sum(1 for a_, b_ in zip(v[:-1], v[1:]) if abs(b_ / a_ - 1) > 2e-4)
            print(f"   {lr:6} {nn}   {mm:.5e}   {100 * ss / mm:.3f}   {distinct}")
        if x == "1e-6":
            xs = np.log(lrs); ys = np.log(m); ss_ = se / m
            for lo, hi in ((0.05, 0.2), (0.05, 0.15)):
                i = sel(lrs, lo, hi)
                e, de, chi = droite(xs[i], ys[i], ss_[i])
                print(f"   loi unique sur {lo} a {hi} ({len(i)} lr) : exposant {e:.4f} +- {de:.4f} ; chi2 {chi:.1f} sur {len(i) - 2} dof")
        a, b, c = 0.15, 0.2, 0.25
        e1, e2, de, et, ecart, et_pred, nsig = chute_deux_pas(lrs, m, se, a, b, c)
        print(f"   pas {a} -> {b} : exposant {e1:.4f} ; pas {b} -> {c} : {e2:.4f} ; chute {de:.4f} +- {et:.4f} ({de / et:.1f} ET) ; valeur a {c} sous la prolongation de {ecart:+.2f} % ({nsig:.1f} ET en propageant les trois moyennes ; ET de la prediction {et_pred:.3f} %)")
        print("   chute sur deux pas (0,15 -> 0,2 contre 0,2 -> 0,25), significance selon le plancher systematique (en % de la moyenne) :")
        for fl in (0.0, 0.03, 0.05, 0.1, 0.2):
            lrs_f, m_f, se_f, _ = stats(d, floor=fl)
            r = chute_deux_pas(lrs_f, m_f, se_f, a, b, c)
            print(f"      plancher {fl:.2f} % : chute {r[2]:.4f} +- {r[3]:.4f} ({r[2] / r[3]:.1f} ET) ; ecart a la prolongation {r[6]:.1f} ET")
        if x == "1e-6":
            print("   segments : bas 0,1-0,15, haut 0,175 jusqu'a la coupure ; significance de la difference d'exposant, selon le plancher (plancher 0 / 0,05 / 0,1 / 0,2 %) :")
            for cut in (0.2, 0.225, 0.25, 0.275, 0.3, 0.35):
                lig = []
                for fl in (0.0, 0.05, 0.1, 0.2):
                    lrs_f, m_f, se_f, _ = stats(d, floor=fl)
                    r = signif_segments(lrs_f, m_f, se_f, haut=(0.175, cut))
                    lig.append(f"{r[4]:5.1f}" if r else "  -  ")
                lrs_f, m_f, se_f, _ = stats(d)
                r0 = signif_segments(lrs_f, m_f, se_f, haut=(0.175, cut))
                print(f"      coupure {cut:5} : exposants bas {r0[0]:+.4f}, haut {r0[2]:+.4f} ; sigma : " + " / ".join(lig))
            print("   retrait d'un seul depart (le meme indice a tous les lr), coupure 0,25, plancher 0 :")
            sig = []
            for kk in range(5):
                lrs_j, m_j, se_j, _ = stats(d, retire_depart=kk)
                r = signif_segments(lrs_j, m_j, se_j, haut=(0.175, 0.25))
                sig.append(r[4]); print(f"      sans ks[{kk}] : bas {r[0]:+.4f}, haut {r[2]:+.4f}, {r[4]:.1f} sigma")
            print(f"      etendue : {min(sig):.1f} a {max(sig):.1f} sigma")
            print("   retrait d'un seul lr du segment haut (coupure 0,25, plancher 0) :")
            lrs_f, m_f, se_f, _ = stats(d)
            for lr_out in (0.175, 0.2, 0.225, 0.25):
                r = signif_segments(lrs_f, m_f, se_f, haut=(0.175, 0.25), retire_lr=lr_out)
                print(f"      sans lr {lr_out} : " + (f"haut {r[2]:+.4f} +- {r[3]:.4f}, {r[4]:.1f} sigma" if r else "segment haut < 2 points"))
        else:
            # x = 1e-4 : lr jusqu'a 0,3 sans fusion (575 a 713 groupes a 0,3) ; segments bas 0,1-0,2 et haut 0,2-0,3, plancher, retrait d'un depart ou d'un lr
            print("   segments bas 0,1-0,2 / haut 0,2-0,3 (lr sans fusion) : significance de la difference d'exposant selon le plancher (0 / 0,05 / 0,1 / 0,2 %) :")
            lig = []
            for fl in (0.0, 0.05, 0.1, 0.2):
                lrs_f, m_f, se_f, _ = stats(d, floor=fl)
                r = signif_segments(lrs_f, m_f, se_f, bas=(0.1, 0.2), haut=(0.2, 0.3))
                lig.append(f"{r[4]:.1f}")
            lrs_f, m_f, se_f, _ = stats(d)
            r0 = signif_segments(lrs_f, m_f, se_f, bas=(0.1, 0.2), haut=(0.2, 0.3))
            print(f"      exposants bas {r0[0]:+.4f} +- {r0[1]:.4f}, haut {r0[2]:+.4f} +- {r0[3]:.4f} ; sigma : " + " / ".join(lig))
            sig = []
            for kk in range(5):
                lrs_j, m_j, se_j, _ = stats(d, retire_depart=kk)
                r = signif_segments(lrs_j, m_j, se_j, bas=(0.1, 0.2), haut=(0.2, 0.3)); sig.append(r[4])
            print(f"      retrait d'un depart : {min(sig):.1f} a {max(sig):.1f} sigma")
            for lr_out in (0.225, 0.25, 0.275, 0.3):
                r = signif_segments(lrs_f, m_f, se_f, bas=(0.1, 0.2), haut=(0.2, 0.3), retire_lr=lr_out)
                print(f"      sans lr {lr_out} : haut {r[2]:+.4f} +- {r[3]:.4f}, {r[4]:.1f} sigma")
            for nom, mod in (("t82 (S_c interpole)", f"{D}/t82_x1e-4_lr0.25_k{{k}}.log"), ("t84 (lecture directe)", f"{D}/t84_x1e-4_lr0.25_k{{k}}.log")):
                vals = []
                for k in range(5):
                    try:
                        v = lire_pz(mod.format(k=k))
                    except FileNotFoundError:
                        v = None
                    vals.append(v)
                print(f"   lr 0,25 {nom} : " + " ".join("-" if v is None else f"k{k}:{v / 1e6:.4f}" for k, v in enumerate(vals)))
