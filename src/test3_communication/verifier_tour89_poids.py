"""Tour 89 : la marche de M vient-elle des poids f (fraction du temps en oscillation) ou des moyennes conditionnelles ?

Il dit : M est une moyenne temporelle, donc M = f M_osc + (1 - f) M_calm ; l'exposant local de f saute en passant par lr 0,2 (0,96 puis 1,48 puis 1,54) au meme endroit ou celui de M tombe de 0,0227 a 0,0140, et le calme median rejoint la duree mediane de groupe (127 contre 127) a lr 0,225 ; si les poids portent la baisse de 0,0088, la marche est le croisement calme / groupe, et son lieu doit suivre lr* (calme median = duree mediane) quand x et b1 changent ; sinon le croisement est une coincidence. Quelle piece porte 0,0088 : les poids ou les moyennes conditionnelles ?

Ce qu'il faut voir d'abord dans mon code : M (tours 87-88) est la moyenne, sur les groupes g >= 9, de 1/rho_g avec rho_g = sqrt(v_e3)/sqrt(v_l3) pris 30 pas AVANT le debut du groupe ; ce n'est pas une moyenne temporelle. La moyenne temporelle de 1/rho(t) sur tous les pas est un autre objet, T, qui admet la decomposition exacte T = f T_osc + (1 - f) T_calm.

Modes :
  f_exposants : exposants locaux de f sur la grille du tour 88 (enregistrements existants), avec ET, et ses trois nombres
  essai <b1> <x> <lr...> : structure des groupes seule (f, calme median, duree mediane) pour une configuration, un depart
  enregistrer <b1> <x> <lr> <k> : enregistrement, sauvegarde de M, T, T_osc, T_calm, f, calme et duree medians (npz)
  temps : decomposition exacte de l'exposant local de T (poids / moyennes conditionnelles) sur la grille, configuration de base
  config <b1> <x> : exposants locaux de M et de T, coude de M ajuste, lr* = croisement calme median / duree mediane
"""

import importlib
import math
import os
import sys

import numpy as np
from mpmath import mp, mpf

sys.path.insert(0, ".")
from verifier_tour75_salves_depart import etat_base, PLI  # noqa: E402
from verifier_tour86_longue import groupes_de  # noqa: E402

mp.dps = 50
D = "D:/tmp"
GRILLE88 = [0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]
MODULES = {0.9: "reduction_numba_tour71_lrvar", 0.8: "reduction_numba_tour74_b1_08", 0.95: "reduction_numba_tour74_b1_095"}


def etiquette(b1, xs):
    return f"b{b1}_x{xs}"


def enregistrement(lr, kidx, xs, b1, nrec=300000):
    """Meme protocole que verifier_tour86_longue.enregistrer (attente de 10 000 pas, moments a l'echelle lr / 0,05), module et x au choix."""
    redlib = importlib.import_module(MODULES[b1])
    redlib.LRV = lr
    eps = float(f"{2e-9 * lr:.4e}")
    S1 = etat_base()
    delta = float(PLI - mpf(xs))
    ks = [int(a) for a in np.random.default_rng(7).integers(0, 4444, 16)]
    rng = np.random.default_rng(11 + kidx)
    w = redlib.copie(S1)
    w["m"] = w["m"] * (lr / 0.05)
    w["v"] = w["v"] * (lr / 0.05) ** 2
    redlib.avancer(w, float(PLI - mpf("1e-6")), 20000 + ks[kidx], eps=eps, cross=2.0, stop=False)
    w["x"] = w["x"] * (1.0 + 1e-9 * rng.standard_normal(6))
    redlib.avancer_obs(w, delta, 10000, np.zeros((1, 7)), eps=eps)
    rec = np.zeros((nrec, 7))
    first, maxrb, rb, done = redlib.avancer_obs(w, delta, nrec, rec, eps=eps)
    if maxrb > 0.9:
        return None, None
    return rec, groupes_de(rec[:, 2])


def statistiques(rec, groupes):
    s_e3 = rec[:, 3]; s_l3 = rec[:, 4]
    a = np.array([g[0] for g in groupes]); dur = np.array([g[1] - g[0] + 1 for g in groupes])
    G = len(a)
    nrec = len(rec)
    calmes = a[1:] - (a[:-1] + dur[:-1])
    res = {"G": G, "f_all": dur.sum() / nrec, "med_calme": float(np.median(calmes)), "med_dur": float(np.median(dur))}
    if G < 20:
        return res
    inv = s_l3 / s_e3
    osc = np.zeros(nrec, bool)
    for ai, di in zip(a, dur):
        osc[ai:ai + di] = True
    t0 = a[9]
    o = osc[t0:]; v = inv[t0:]
    res.update({"f": float(o.mean()), "T": float(v.mean()), "T_osc": float(v[o].mean()) if o.any() else float("nan"), "T_calme": float(v[~o].mean()) if (~o).any() else float("nan"),
                "M": float(np.mean(inv[a[9:] - 30])), "med_calme9": float(np.median(calmes[9:])), "med_dur9": float(np.median(dur[9:]))})
    return res


def arithmetique_f():
    """f_all par lr a partir des enregistrements du tour 86-88 et exposants locaux de f."""
    f = {}; se = {}
    for lr in GRILLE88:
        v = []
        for k in range(10):
            try:
                z = np.load(f"{D}/t86_lr{lr}_k{k}.npz")
            except FileNotFoundError:
                continue
            v.append(z["dur"].sum() / int(z["nrec"]))
        f[lr] = np.mean(v); se[lr] = np.std(v, ddof=1) / math.sqrt(len(v))
    print("f (fraction du temps en oscillation, tous les groupes) par lr : " + " ".join(f"{lr}:{f[lr]:.4f}(+-{se[lr]:.4f})" for lr in GRILLE88))
    e = []
    for a, b in zip(GRILLE88[:-1], GRILLE88[1:]):
        l = math.log(b / a)
        ex = math.log(f[b] / f[a]) / l
        s = math.hypot(se[a] / f[a], se[b] / f[b]) / l
        e.append((ex, s))
        print(f"   {a} -> {b} : exposant de f {ex:+.3f} +- {s:.3f}")
    print("variation de l'exposant de f d'un intervalle a l'autre (en ET) :")
    for i in range(len(e) - 1):
        d = e[i + 1][0] - e[i][0]
        s = math.hypot(e[i][1], e[i + 1][1])
        print(f"   {GRILLE88[i]}-{GRILLE88[i+1]} -> {GRILLE88[i+1]}-{GRILLE88[i+2]} : {d:+.3f} +- {s:.3f}")
    print("ses trois nombres : 0,175->0,2 :", f"{e[3][0]:.2f}", "; 0,2->0,225 :", f"{e[4][0]:.2f}", "; 0,225->0,25 :", f"{e[5][0]:.2f}")


def essai(b1, xs, lrs):
    """Structure des groupes seule (aucune moyenne de 1/rho n'est imprimee) pour une configuration, un depart."""
    for lr in lrs:
        rec, groupes = enregistrement(lr, 0, xs, b1)
        if rec is None:
            print(f"b1 {b1} x {xs} lr {lr}: effondre")
            continue
        st = statistiques(rec, groupes)
        print(f"b1 {b1} x {xs} lr {lr}: {st['G']} groupes ; f {st['f_all']:.3f} ; calme median {st['med_calme']:.0f} ; duree mediane {st['med_dur']:.0f}")


def enregistrer(b1, xs, lr, k):
    rec, groupes = enregistrement(lr, k, xs, b1)
    chemin = f"{D}/t89_{etiquette(b1, xs)}_lr{lr}_k{k}.npz"
    if rec is None:
        print(f"b1 {b1} x {xs} lr {lr} k{k}: effondre")
        np.savez(chemin, G=0)
        return
    st = statistiques(rec, groupes)
    print(f"b1 {b1} x {xs} lr {lr} k{k}: " + " ".join(f"{a} {b:.6g}" for a, b in st.items()))
    np.savez(chemin, **st)


def charger_config(b1, xs, lrs, ks=range(10)):
    res = {}
    for lr in lrs:
        lignes = []
        for k in ks:
            try:
                z = np.load(f"{D}/t89_{etiquette(b1, xs)}_lr{lr}_k{k}.npz")
            except FileNotFoundError:
                continue
            if int(z["G"]) < 20 or "M" not in z.files:
                continue
            lignes.append({n: float(z[n]) for n in z.files})
        if lignes:
            res[lr] = {n: np.array([l[n] for l in lignes]) for n in lignes[0]}
    return res


def temps():
    res = charger_config(0.9, "1e-4", GRILLE88)
    lrs = [l for l in GRILLE88 if l in res]
    mm = {l: {n: res[l][n].mean() for n in ("f", "T", "T_osc", "T_calme", "M")} for l in lrs}
    print("niveaux moyens (departs) : lr, f, T, T_osc, T_calme, M, departs")
    for l in lrs:
        print(f"   {l}: f {mm[l]['f']:.4f} T {mm[l]['T']:.6g} T_osc {mm[l]['T_osc']:.6g} T_calme {mm[l]['T_calme']:.6g} M {mm[l]['M']:.6g} ({len(res[l]['M'])})")
        ident = mm[l]["f"] * mm[l]["T_osc"] + (1 - mm[l]["f"]) * mm[l]["T_calme"]
    print("identite T = f T_osc + (1-f) T_calme : ecart relatif max " + f"{max(abs(mm[l]['f'] * mm[l]['T_osc'] + (1 - mm[l]['f']) * mm[l]['T_calme']) / mm[l]['T'] - 1 for l in lrs):.2e}")
    print("exposants locaux de T et de M, et decomposition exacte de celui de T : poids = df (T_osc - T_calme) / T, conditionnelles = f dT_osc / T et (1-f) dT_calme / T (moyennes arithmetiques des deux lr) :")
    pieces = []
    for a, b in zip(lrs[:-1], lrs[1:]):
        l = math.log(b / a)
        fa, fb = mm[a]["f"], mm[b]["f"]; fm = (fa + fb) / 2
        To = (mm[a]["T_osc"] + mm[b]["T_osc"]) / 2; Tc = (mm[a]["T_calme"] + mm[b]["T_calme"]) / 2; Tm = (mm[a]["T"] + mm[b]["T"]) / 2
        w = (fb - fa) * (To - Tc) / Tm / l
        co = fm * (mm[b]["T_osc"] - mm[a]["T_osc"]) / Tm / l
        cc = (1 - fm) * (mm[b]["T_calme"] - mm[a]["T_calme"]) / Tm / l
        eT = math.log(mm[b]["T"] / mm[a]["T"]) / l; eM = math.log(mm[b]["M"] / mm[a]["M"]) / l
        eC = math.log(mm[b]["T_calme"] / mm[a]["T_calme"]) / l
        pieces.append((a, b, l, w, co, cc, eT, eM, eC, eM - eC))
        print(f"   {a} -> {b} : M {eM:+.5f} ; T {eT:+.5f} = poids {w:+.5f} + osc {co:+.5f} + calme {cc:+.5f} (somme {w + co + cc:+.5f}) ; T_calme {eC:+.5f} ; M / T_calme {eM - eC:+.5f}")
    def plage(lo, hi):
        sel = [p for p in pieces if p[0] >= lo - 1e-9 and p[1] <= hi + 1e-9]
        L = sum(p[2] for p in sel)
        return [sum(p[i] * p[2] for p in sel) / L for i in (3, 4, 5, 6, 7, 8, 9)]
    bas, haut = plage(0.15, 0.2), plage(0.2, 0.3)
    print("exposant moyen 0,15 -> 0,2 contre 0,2 -> 0,3 (poids, osc, calme, T, M, T_calme, M/T_calme) :")
    print("   bas  : " + " ".join(f"{x:+.5f}" for x in bas)); print("   haut : " + " ".join(f"{x:+.5f}" for x in haut))
    d = [b - h for b, h in zip(bas, haut)]
    print("   marche (bas - haut) : " + " ".join(f"{x:+.5f}" for x in d))
    if abs(d[3]) > 1e-9:
        print(f"   part de la marche de T : poids {100 * d[0] / d[3]:.0f} %, osc {100 * d[1] / d[3]:.0f} %, calme {100 * d[2] / d[3]:.0f} %")


def ajuster_coude(lr, y, ecart_min=0.0):
    """Ajustement continu a deux segments de y = ln M contre x = ln lr ; retourne (lr_coude, pente basse, pente haute, RSS_coude, RSS_droite)."""
    x = np.log(lr); y = np.asarray(y)
    A = np.vstack([x, np.ones_like(x)]).T
    c = np.linalg.lstsq(A, y, rcond=None)[0]
    rss_d = float(((y - A @ c) ** 2).sum())
    best = None
    for xk in np.linspace(x[1], x[-2], 400):
        B = np.vstack([x, np.ones_like(x), np.maximum(x - xk, 0.0)]).T
        c = np.linalg.lstsq(B, y, rcond=None)[0]
        rss = float(((y - B @ c) ** 2).sum())
        if best is None or rss < best[0]:
            best = (rss, math.exp(xk), c[0], c[0] + c[2])
    return best[1], best[2], best[3], best[0], rss_d


def config(b1, xs, lrs):
    res = charger_config(b1, xs, lrs, range(5))
    lrs = [l for l in lrs if l in res]
    if len(lrs) < 5:
        print(f"b1 {b1} x {xs} : trop peu de lr ({len(lrs)})")
        return
    M = np.array([res[l]["M"].mean() for l in lrs]); T = np.array([res[l]["T"].mean() for l in lrs])
    cal = np.array([np.mean(res[l]["med_calme9"]) for l in lrs]); dur = np.array([np.mean(res[l]["med_dur9"]) for l in lrs]); f = np.array([res[l]["f"].mean() for l in lrs])
    print(f"b1 {b1} x {xs} : lr, f, calme median, duree mediane, exposant local de M, de T")
    for i, l in enumerate(lrs):
        if i + 1 < len(lrs):
            ll = math.log(lrs[i + 1] / l)
            print(f"   {l}: f {f[i]:.3f} calme {cal[i]:.0f} duree {dur[i]:.0f} ; {l}->{lrs[i + 1]}: M {math.log(M[i + 1] / M[i]) / ll:+.5f} T {math.log(T[i + 1] / T[i]) / ll:+.5f}")
        else:
            print(f"   {l}: f {f[i]:.3f} calme {cal[i]:.0f} duree {dur[i]:.0f}")
    d = cal - dur
    star = None
    for i in range(len(lrs) - 1):
        if d[i] > 0 >= d[i + 1]:
            u = d[i] / (d[i] - d[i + 1])
            star = math.exp(math.log(lrs[i]) + u * math.log(lrs[i + 1] / lrs[i]))
            break
    print(f"   lr* (calme median = duree mediane, interpolation en ln lr) : {star if star is None else round(star, 4)}")
    for nom, y in (("M", M), ("T", T)):
        lk, e1, e2, rss, rss0 = ajuster_coude(np.array(lrs), np.log(y))
        print(f"   coude de {nom} : lr_coude {lk:.4f} ; pente basse {e1:+.5f}, pente haute {e2:+.5f}, baisse {e1 - e2:+.5f} ; RSS coude {rss:.3e} contre droite {rss0:.3e} (rapport {rss / rss0:.4f})")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "f_exposants":
        arithmetique_f()
    elif mode == "essai":
        essai(float(sys.argv[2]), sys.argv[3], [float(v) for v in sys.argv[4:]])
    elif mode == "enregistrer":
        enregistrer(float(sys.argv[2]), sys.argv[3], float(sys.argv[4]), int(sys.argv[5]))
    elif mode == "temps":
        temps()
    elif mode == "config":
        config(float(sys.argv[2]), sys.argv[3], [float(v) for v in sys.argv[4:]])
