"""Tour 87 : les salves en retard ont-elles une definition avant la lecture, et la population a +0,4 % ?

Il dit : oui au niveau de l'enregistrement, non a "toutes les salves comptees" ; rapporter p_late(lr) et l'exposant sur les lectures non en retard, "en retard" defini par l'instant de franchissement avant le groupe (jamais par la lecture) ; la population a +0,4 % a-t-elle une definition avant le groupe ?

Modes :
  arithmetique : ses chiffres a partir de mes chiffres du tour 86 (aucune donnee lue)
"""

import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, ".")
from verifier_tour86_derive import poids_ponderes  # noqa: E402


def arithmetique():
    d = math.log(3.29510e6 / 3.30752e6)
    print(f"ln(3,29510e6 / 3,30752e6) = {100 * d:+.4f} % ; 31/300 = {100 * 31 / 300:.2f} % ; deficit moyen des salves en retard si tout l'ecart vient d'elles : {-100 * d / (31 / 300):.2f} %")
    print(f"exposant haut : 0,0127 - 0,0053 = {0.0127 - 0.0053:.4f} ; divise par {-d:.5f} = poids {(0.0127 - 0.0053) / -d:.2f} ; part de la difference bas - haut : {0.0074 / 0.0131 * 100:.1f} %")
    # poids du point 0,3 dans les deux ajustements (ET entre departs + plancher 0,1 %), pour verifier que 1,97 est un poids
    for nom, se03 in (("toutes les lectures", 0.00102), ("sans les salves en retard", 0.00024)):
        se = [math.hypot(0.00009, 0.001), math.hypot(0.00018, 0.001), math.hypot(se03, 0.001)]
        w = poids_ponderes(se)
        print(f"poids des points 0,2 / 0,25 / 0,3 ({nom}, ET entre departs 0,009 / 0,018 / {100 * se03:.3f} % + plancher 0,1 %) : {np.round(w, 3)}")


D = "D:/tmp"
LRS = [0.1, 0.15, 0.2, 0.25, 0.3]
MOTIF = re.compile(r"salve (\d+) : calme avant (\d+) pas ; duree (\d+) pas ; lr p_z ([0-9.e+]+) ; sqrt\(v_e3\) ([0-9.e+\-]+) ; avance ([0-9.\-]+) pas")
SEUIL_AVANCE = float(os.environ.get("TOUR87_SEUIL", "100"))   # precommis : 40 pas (echec : l'avance normale est 63 pas a lr 0,1 et 20 pas a lr 0,3) ; 100 pas trouve APRES avoir vu les histogrammes (ecart net entre 63 et 130 pas)


def charger(ks):
    """Lignes (lr, k, g, calme, duree, duree du groupe precedent, lecture, avance, rho, sqrt(v_l3), z3, ecart C/rho - lecture en % ou nan)."""
    lignes = []
    for lr in LRS:
        for k in ks:
            try:
                t = open(f"{D}/t87_alea_lr{lr}_k{k}.log", encoding="utf-8", errors="replace").read()
                z = np.load(f"{D}/t86_lr{lr}_k{k}.npz")
            except FileNotFoundError:
                continue
            for g, calme, dur, r, sv, av in MOTIF.findall(t):
                g = int(g); r = float(r)
                pr = z["PR"][g]
                ec = 100 * (pr / r - 1) if np.isfinite(pr) else float("nan")
                lignes.append((lr, k, g, int(calme), int(dur), int(z["dur"][g - 1]), r, float(av), float(z["RHO"][g, 1]), float(z["SLD"][g, 2]), float(z["Z3"][g]), ec))
    return np.array(lignes)


def centres(A):
    """ln(lecture) en %, centre par depart sur la mediane de ses lectures hors retard (avance <= seuil) ; meme chose pour ln(lecture x rho)."""
    y = np.zeros(len(A)); yr = np.zeros(len(A))
    for lr in LRS:
        for k in np.unique(A[:, 1]):
            m = (A[:, 0] == lr) & (A[:, 1] == k)
            if not m.any():
                continue
            nl = m & (A[:, 7] <= SEUIL_AVANCE)
            ref = nl if nl.sum() >= 5 else m
            y[m] = 100 * (np.log(A[m, 6]) - np.median(np.log(A[ref, 6])))
            yr[m] = 100 * (np.log(A[m, 6] * A[m, 8]) - np.median(np.log(A[ref, 6] * A[ref, 8])))
    return y, yr


def moyennes_depart(A, masque):
    """dict lr -> tableau des moyennes de depart (lectures retenues par le masque)."""
    res = {}
    for lr in LRS:
        v = []
        for k in np.unique(A[:, 1]):
            m = (A[:, 0] == lr) & (A[:, 1] == k) & masque
            if m.sum() >= 5:
                v.append(A[m, 6].mean())
        res[lr] = np.array(v)
    return res


def test_segments(A, masque, etiquette):
    from verifier_tour84_coupures import signif_segments
    mm = moyennes_depart(A, masque)
    lrs = np.array([l for l in LRS if len(mm[l]) >= 2])
    if len(lrs) < 5:
        print(f"   {etiquette} : lr manquants")
        return
    m = np.array([mm[l].mean() for l in lrs]); se = np.array([mm[l].std(ddof=1) / math.sqrt(len(mm[l])) for l in lrs])
    ligne = f"   {etiquette} (departs par lr {len(mm[0.3])}) : moyenne a lr 0,3 {m[-1]:.5e}, ET entre departs {100 * se[-1] / m[-1]:.3f} %"
    for nom, ee in (("ET seule", se), ("+ plancher 0,1 %", np.hypot(se, 0.001 * m))):
        r = signif_segments(lrs, m, ee, bas=(0.1, 0.2), haut=(0.2, 0.3))
        ligne += f" | {nom} : bas {r[0]:+.4f}, haut {r[2]:+.4f} +- {r[3]:.4f}, difference {r[0] - r[2]:+.4f}, {r[4]:.1f} sigma"
    print(ligne)


def meilleur_seuil(x, pos):
    """Seuil unique et sens maximisant la precision equilibree ; retourne (seuil, sens, precision)."""
    ordre = np.sort(np.unique(x))
    mids = (ordre[:-1] + ordre[1:]) / 2
    best = (None, 0, 0.0)
    for s in mids:
        for sens in (1, -1):
            pred = (x > s) if sens == 1 else (x < s)
            tpr = (pred & pos).sum() / max(pos.sum(), 1); tnr = (~pred & ~pos).sum() / max((~pos).sum(), 1)
            acc = (tpr + tnr) / 2
            if acc > best[2]:
                best = (s, sens, acc)
    return best


def precision(x, pos, seuil, sens):
    pred = (x > seuil) if sens == 1 else (x < seuil)
    return ((pred & pos).sum() / max(pos.sum(), 1) + (~pred & ~pos).sum() / max((~pos).sum(), 1)) / 2


def analyse():
    for nom, ks in (("exploration ks[0..4]", range(5)), ("confirmation ks[5..9]", range(5, 10)), ("les dix departs", range(10))):
        A = charger(ks)
        if len(A) == 0:
            print(f"== {nom} : pas de donnees")
            continue
        y, yr = centres(A)
        retard = A[:, 7] > SEUIL_AVANCE
        print(f"===== {nom} : {len(A)} lectures")
        print("avance (pas) par lr : percentiles 1 5 25 50 75 95 99, max ; nombre au-dessus de 40 (p_late) :")
        for lr in LRS:
            m = A[:, 0] == lr
            if m.any():
                a = A[m, 7]
                print(f"   lr {lr}: " + " ".join(f"{v:7.1f}" for v in np.percentile(a, [1, 5, 25, 50, 75, 95, 99])) + f" max {a.max():7.1f} ; en retard {int((a > SEUIL_AVANCE).sum())}/{int(m.sum())} ({100 * (a > SEUIL_AVANCE).mean():.1f} %)")
        for lr in (0.25, 0.3):
            m = A[:, 0] == lr
            if m.any():
                h, bords = np.histogram(A[m, 7], bins=np.arange(0, 150, 10))
                print(f"   histogramme de l'avance a lr {lr} (classes de 10 pas) : " + " ".join(f"{int(b_)}:{n_}" for b_, n_ in zip(bords[:-1], h)))
        if np.isfinite(A[:, 11]).any():
            print("ecart C/rho - lecture (%) des lectures hors retard, par depart (mediane, sd, puis mediane par tiers de l'indice de salve) :")
            for lr in (0.25, 0.3):
                for k in np.unique(A[:, 1]):
                    m = (A[:, 0] == lr) & (A[:, 1] == k) & (A[:, 7] <= SEUIL_AVANCE)
                    if m.sum() >= 9:
                        ec_ = A[m, 11]; g_ = A[m, 2]
                        t3 = np.percentile(g_, [33.3, 66.7])
                        parts = [np.median(ec_[g_ <= t3[0]]), np.median(ec_[(g_ > t3[0]) & (g_ <= t3[1])]), np.median(ec_[g_ > t3[1]])]
                        print(f"   lr {lr} k{int(k)}: {int(m.sum())} lectures ; mediane {np.median(ec_):+.3f}, sd {ec_.std(ddof=1):.3f} ; par tiers {parts[0]:+.3f} {parts[1]:+.3f} {parts[2]:+.3f}")
        m3 = A[:, 0] == 0.3
        if m3.any():
            ec = A[m3, 11]
            if np.isfinite(ec).any():
                ecart_tard = ec > 0.3; av_tard = retard[m3]
                print(f"lr 0,3 : croisement de la definition par la lecture (ecart C/rho - lecture > +0,3 %) et de la definition par l'avance (> 40) : lecture&avance {int((ecart_tard & av_tard).sum())}, lecture seule {int((ecart_tard & ~av_tard).sum())}, avance seule {int((~ecart_tard & av_tard).sum())}, ni l'une ni l'autre {int((~ecart_tard & ~av_tard).sum())} ; deficit moyen des lectures en retard par l'avance (ecart a C/rho) {np.nanmean(ec[av_tard]):+.2f} %")
            # P87-3 : pente de ln(lecture) contre l'avance
            x = A[m3, 7]; yy = y[m3]
            b = ((x - x.mean()) * (yy - yy.mean())).sum() / ((x - x.mean()) ** 2).sum()
            r2 = 1 - ((yy - yy.mean() - b * (x - x.mean())) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
            print(f"lr 0,3 : pente de ln(lecture) contre l'avance {b:+.4f} % par pas, R2 {r2:.3f} (toutes les lectures) ; hors retard : ", end="")
            nl = ~retard[m3]
            xb = x[nl]; yb = yy[nl]
            b2 = ((xb - xb.mean()) * (yb - yb.mean())).sum() / ((xb - xb.mean()) ** 2).sum()
            r22 = 1 - ((yb - yb.mean() - b2 * (xb - xb.mean())) ** 2).sum() / ((yb - yb.mean()) ** 2).sum()
            print(f"{b2:+.4f} % par pas, R2 {r22:.3f}")
            # P87-4 : lecture corrigee de l'avance
            zc = yy + 100 * 0.0005 * x
            print(f"lr 0,3 : ecart-type de ln(lecture) {yy.std(ddof=1):.4f} % ; de ln(lecture) + 0,05 % x avance {zc.std(ddof=1):.4f} % ; hors retard {yb.std(ddof=1):.4f} % contre {(yb + 100 * 0.0005 * xb).std(ddof=1):.4f} %")
            h, bords = np.histogram(zc, bins=np.arange(np.floor(zc.min() * 5) / 5, zc.max() + 0.2, 0.2))
            print("   histogramme de ln(lecture) + 0,05 % x avance (classes de 0,2 %) : " + " ".join(f"{b_:+.1f}:{n_}" for b_, n_ in zip(bords[:-1], h) if n_))
            h, bords = np.histogram(yb, bins=np.arange(np.floor(yb.min() * 5) / 5, yb.max() + 0.2, 0.2))
            print("   histogramme de ln(lecture) hors retard (classes de 0,2 %) : " + " ".join(f"{b_:+.1f}:{n_}" for b_, n_ in zip(bords[:-1], h) if n_))
        print("test de segments (bas 0,1-0,2, haut 0,2-0,3), moyennes de depart :")
        test_segments(A, np.ones(len(A), bool), "toutes les lectures")
        test_segments(A, ~retard, f"hors retard (avance <= {SEUIL_AVANCE:.0f})")
    # P87-7 : classer la population a +0,4 % par une variable d'avant le groupe
    E = charger(range(5)); C = charger(range(5, 10))
    if len(E) and len(C):
        print("===== la population a +0,4 % de lr 0,3 : variables d'avant le groupe, seuil choisi sur ks[0..4], applique tel quel a ks[5..9]")
        for nom, A in (("ks[0..4]", E), ("ks[5..9]", C)):
            m3 = A[:, 0] == 0.3
            y, yr = centres(A)
            nl = m3 & (A[:, 7] <= SEUIL_AVANCE)
            print(f"   {nom} : lr 0,3 hors retard {int(nl.sum())} lectures ; ecart-type de ln(lecture x rho) centre {yr[nl].std(ddof=1):.4f} % ; percentiles 5 25 50 75 95 : " + " ".join(f"{v:+.3f}" for v in np.percentile(yr[nl], [5, 25, 50, 75, 95])))
        mE = (E[:, 0] == 0.3) & (E[:, 7] <= SEUIL_AVANCE)
        yE, yrE = centres(E)
        ecE = E[:, 11]
        # etiquette sur ks[0..4] : la meme que celle du tour 86 (ecart C/rho - lecture < -0,3 %) ; sur ks[5..9] : yr centre au-dessus du seuil qui reproduit cette etiquette sur ks[0..4]
        pos_ref = ecE[mE] < -0.3
        s_lab, sens_lab, acc_lab = meilleur_seuil(yrE[mE], pos_ref)
        print(f"   etiquette de reference (ks[0..4]) : {int(pos_ref.sum())} lectures a +0,4 % sur {int(mE.sum())} ; seuil sur ln(lecture x rho) centre {s_lab:+.3f} % (sens {sens_lab:+d}), precision equilibree {acc_lab:.3f}")
        mC = (C[:, 0] == 0.3) & (C[:, 7] <= SEUIL_AVANCE)
        yC, yrC = centres(C)
        posC = (yrC[mC] > s_lab) if sens_lab == 1 else (yrC[mC] < s_lab)
        print(f"   ks[5..9] : {int(posC.sum())} lectures a +0,4 % sur {int(mC.sum())} ({100 * posC.mean():.1f} %) contre {100 * pos_ref.mean():.1f} % sur ks[0..4]")
        noms = ["avance (pas)", "calme avant (pas)", "duree du groupe precedent (pas)", "rho (centre, %)", "sqrt(v_l3) (centre, %)", "z3 (centre, %)", "indice de salve g"]
        cols = [7, 3, 5, 8, 9, 10, 2]
        def cen(A, mask, col):
            x = A[:, col].astype(float).copy()
            if col in (8, 9):
                x = 100 * np.log(x)
            if col in (8, 9, 10):   # z3 : difference brute a la mediane du depart (le signe de z3 n'est pas fixe)
                for lr in LRS:
                    for k in np.unique(A[:, 1]):
                        mm = (A[:, 0] == lr) & (A[:, 1] == k)
                        if mm.any():
                            x[mm] -= np.median(x[mm])
            return x[mask]
        for nom, col in zip(noms, cols):
            xE = cen(E, mE, col); xC = cen(C, mC, col)
            s, sens, acc = meilleur_seuil(xE, pos_ref)
            accC = precision(xC, posC, s, sens)
            print(f"   {nom:34s}: seuil {s:+.3f} (sens {sens:+d}) precision equilibree ks[0..4] {acc:.3f}, ks[5..9] {accC:.3f}")
        # la meme avance : P87-7 version "plus courte que la mediane des lectures normales"
        normal = mE & (np.abs(ecE) <= 0.3)
        haute = mE & (ecE < -0.3)
        if haute.any() and normal.any():
            med = np.median(E[normal, 7])
            print(f"   avance mediane des lectures normales {med:.1f} pas ; part des lectures a +0,4 % avec une avance plus courte {100 * (E[haute, 7] < med).mean():.1f} % ({int(haute.sum())} lectures)")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "arithmetique":
        arithmetique()
    elif mode == "analyse":
        analyse()
