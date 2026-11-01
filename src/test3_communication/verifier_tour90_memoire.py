"""Tour 90 : b1 = 0,95 tombe-t-il sur la meme courbe M(f) que les trois x ? La memoire de rho est-elle b2 ?

Il dit : M est une moyenne par groupe, pas une moyenne temporelle ; verification gratuite sur les enregistrements b1 = 0,95 : tracer M contre f, pas contre lr ; son lr* est 0,0608, donc la plus grande partie de la grille est apres le croisement, ou les courbes en x s'aplatissent deja (pente de ln M sur ln f de 0,0072 a 0,0089) ; si b1 = 0,95 tombe sur la meme courbe, l'absence de marche est une grille qui commence apres la marche ; ensuite b1 = 0,85 et 0,92 testent la collapse ; prediction : la pente de ln M sur ln f chute entre f 0,45 et 0,5, ou que ce soit en lr. Question de mecanisme : rho est un rapport de deux moyennes a memoire de 1 000 pas contre des salves de 120 ; la memoire est-elle fixee par b2, peut-on la changer sans toucher b1 ? Si rho suit f sur les 1 000 derniers pas, la chute doit bouger quand la memoire change, a calme = groupe fixe.

Modes :
  libre      : verification gratuite sur les enregistrements existants (b1 = 0,95 et les trois x)
  essai <module> <x> <lr...> : structure des groupes seule (aucune moyenne de 1/rho imprimee), un depart
  enregistrer <module> <x> <lr> <k> : enregistrement, sauvegarde M, T, f, calme et duree medians
  config <module> <x> <lr...> : exposants locaux de M, lr*, plus grandes baisses, pente de ln M sur ln f
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
from verifier_tour89_poids import charger_config, statistiques  # noqa: E402

mp.dps = 50
D = "D:/tmp"
BASE_LRS = [0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]
B95_LRS = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.16, 0.2]
X3_LRS = [0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4]
X5_LRS = [0.05, 0.075, 0.1, 0.1125, 0.125, 0.1375, 0.15, 0.1625, 0.175, 0.1875, 0.2, 0.2125, 0.225, 0.2375, 0.25]


def courbe(b1, xs, lrs):
    res = charger_config(b1, xs, lrs, range(5))
    ls = [l for l in lrs if l in res]
    f = np.array([res[l]["f"].mean() for l in ls]); M = np.array([res[l]["M"].mean() for l in ls]); seM = np.array([res[l]["M"].std(ddof=1) / math.sqrt(len(res[l]["M"])) / res[l]["M"].mean() for l in ls])
    o = np.argsort(f)
    return np.array(ls)[o], f[o], np.log(M[o]), seM[o]


def pentes_f(f, y):
    return [((f[i] + f[i + 1]) / 2, (y[i + 1] - y[i]) / math.log(f[i + 1] / f[i])) for i in range(len(f) - 1)]


def libre():
    cfgs = [("b0.9 x1e-4", 0.9, "1e-4", BASE_LRS), ("b0.9 x1e-3", 0.9, "1e-3", X3_LRS), ("b0.9 x1e-5", 0.9, "1e-5", X5_LRS), ("b0.95 x1e-4", 0.95, "1e-4", B95_LRS)]
    cs = {n: courbe(b1, xs, lrs) for n, b1, xs, lrs in cfgs}
    ls, f, y, se = cs["b0.95 x1e-4"]
    print(f"b1 = 0,95 : {len(ls)} lr de {ls.min()} a {ls.max()} ; f de {f.min():.3f} a {f.max():.3f} ; points avec f >= 0,50 : {int((f >= 0.5).sum())} sur {len(f)} ; avec f < 0,47 : {int((f < 0.47).sum())}")
    print("pentes de ln M sur ln f, b1 = 0,95 : " + " ".join(f"{fm:.3f}:{p:+.5f}" for fm, p in pentes_f(f, y)))
    for n in ("b0.9 x1e-4", "b0.9 x1e-3", "b0.9 x1e-5"):
        lsx, fx, yx, sex = cs[n]
        print(f"pentes de ln M sur ln f, {n} : " + " ".join(f"{fm:.3f}:{p:+.4f}" for fm, p in pentes_f(fx, yx)))
    print("amplitude 100 x (ln M(f2) - ln M(f1)), interpolee en f :")
    plages = [(0.32, 0.50), (0.50, 0.79), (0.35, 0.45), (0.50, 0.60)]
    for n, (lsx, fx, yx, sex) in cs.items():
        ligne = []
        for f1, f2 in plages:
            if fx.min() <= f1 and fx.max() >= f2:
                ligne.append(f"[{f1:.2f},{f2:.2f}] {100 * (np.interp(f2, fx, yx) - np.interp(f1, fx, yx)):+.4f}")
            else:
                ligne.append(f"[{f1:.2f},{f2:.2f}] hors plage")
        print(f"   {n:12s}: " + "  ".join(ligne))
    # rapport b1 = 0,95 sur les x, plage f 0,32-0,50 et f 0,50-0,79
    a95 = {p: 100 * (np.interp(p[1], f, y) - np.interp(p[0], f, y)) for p in plages}
    for n in ("b0.9 x1e-4", "b0.9 x1e-3", "b0.9 x1e-5"):
        lsx, fx, yx, sex = cs[n]
        r = []
        for p in plages[:2]:
            if fx.min() <= p[0] and fx.max() >= p[1]:
                ax = 100 * (np.interp(p[1], fx, yx) - np.interp(p[0], fx, yx))
                r.append(f"[{p[0]:.2f},{p[1]:.2f}] b1 0,95 / {n} = {a95[p] / ax:.3f}")
        print("   " + " ; ".join(r))
    print("incertitude de M a b1 = 0,95 (relative, %) : " + " ".join(f"{100 * s:.4f}" for s in se))


def enregistrement(module, lr, kidx, xs, nrec=300000):
    """Meme protocole que verifier_tour89_poids.enregistrement ; module donne par son nom ; TOUR90_ATT : attente avant l'enregistrement (10 000 par defaut)."""
    redlib = importlib.import_module(module)
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
    att = int(os.environ.get("TOUR90_ATT", "10000"))
    redlib.avancer_obs(w, delta, att, np.zeros((1, 7)), eps=eps)
    rec = np.zeros((nrec, 7))
    first, maxrb, rb, done = redlib.avancer_obs(w, delta, nrec, rec, eps=eps)
    if maxrb > 0.9:
        return None, None
    return rec, groupes_de(rec[:, 2])


def etiquette(module, xs):
    return f"{module.replace('reduction_numba_tour', '')}_x{xs}"


def essai(module, xs, lrs):
    for lr in lrs:
        rec, groupes = enregistrement(module, lr, 0, xs)
        if rec is None:
            print(f"{module} x {xs} lr {lr}: effondre")
            continue
        st = statistiques(rec, groupes)
        print(f"{module} x {xs} lr {lr}: {st['G']} groupes ; f {st['f_all']:.3f} ; calme median {st['med_calme']:.0f} ; duree mediane {st['med_dur']:.0f}")


def enregistrer(module, xs, lr, k):
    rec, groupes = enregistrement(module, lr, k, xs)
    chemin = f"{D}/t90_{etiquette(module, xs)}_lr{lr}_k{k}.npz"
    if rec is None:
        print(f"{module} x {xs} lr {lr} k{k}: effondre")
        np.savez(chemin, G=0)
        return
    st = statistiques(rec, groupes)
    print(f"{module} x {xs} lr {lr} k{k}: " + " ".join(f"{a} {b:.6g}" for a, b in st.items()))
    np.savez(chemin, **st)


def charger90(module, xs, lrs, ks=range(5)):
    res = {}
    for lr in lrs:
        lignes = []
        for k in ks:
            try:
                z = np.load(f"{D}/t90_{etiquette(module, xs)}_lr{lr}_k{k}.npz")
            except FileNotFoundError:
                continue
            if int(z["G"]) < 20 or "M" not in z.files:
                continue
            lignes.append({n: float(z[n]) for n in z.files})
        if lignes:
            res[lr] = {n: np.array([l[n] for l in lignes]) for n in lignes[0]}
    return res


def config(module, xs, lrs):
    res = charger90(module, xs, lrs)
    lrs = [l for l in lrs if l in res]
    if len(lrs) < 5:
        print(f"{module} x {xs} : trop peu de lr ({len(lrs)})")
        return
    M = np.array([res[l]["M"].mean() for l in lrs]); f = np.array([res[l]["f"].mean() for l in lrs])
    cal = np.array([np.mean(res[l]["med_calme9"]) for l in lrs]); dur = np.array([np.mean(res[l]["med_dur9"]) for l in lrs])
    seM = np.array([res[l]["M"].std(ddof=1) / math.sqrt(len(res[l]["M"])) / res[l]["M"].mean() for l in lrs])
    print(f"{module} x {xs} : lr, f, calme, duree, exposant local de M (en lr), pente de ln M sur ln f")
    ex = []
    for i, l in enumerate(lrs):
        if i + 1 < len(lrs):
            ll = math.log(lrs[i + 1] / l)
            e = math.log(M[i + 1] / M[i]) / ll; sf = (math.log(M[i + 1]) - math.log(M[i])) / math.log(f[i + 1] / f[i])
            ex.append((e, math.hypot(seM[i], seM[i + 1]) / ll, (f[i] + f[i + 1]) / 2, sf))
            print(f"   {l}: f {f[i]:.3f} calme {cal[i]:.0f} duree {dur[i]:.0f} ; {l}->{lrs[i + 1]}: exposant {e:+.5f} ; pente en f (f = {(f[i] + f[i + 1]) / 2:.3f}) {sf:+.5f}")
        else:
            print(f"   {l}: f {f[i]:.3f} calme {cal[i]:.0f} duree {dur[i]:.0f}")
    d = cal - dur
    star = None
    for i in range(len(lrs) - 1):
        if d[i] > 0 >= d[i + 1]:
            u = d[i] / (d[i] - d[i + 1])
            star = math.exp(math.log(lrs[i]) + u * math.log(lrs[i + 1] / lrs[i]))
            break
    print(f"   lr* (calme median = duree mediane) : {star if star is None else round(star, 4)}")
    chutes = sorted([(ex[i][0] - ex[i + 1][0], math.hypot(ex[i][1], ex[i + 1][1]), lrs[i + 1], (f[i + 1])) for i in range(len(ex) - 1)], reverse=True)
    print("   plus grandes baisses de l'exposant de M en lr (au point lr, f) : " + " ; ".join(f"{c[2]} (f {c[3]:.3f}): {c[0]:+.5f} +- {c[1]:.5f}" for c in chutes[:3]))
    cf = sorted([(ex[i][3] - ex[i + 1][3], ex[i + 1][2]) for i in range(len(ex) - 1)], reverse=True)
    print("   plus grandes baisses de la pente en ln f (f au milieu de l'intervalle commun) : " + " ; ".join(f"f {c[1]:.3f}: {c[0]:+.5f}" for c in cf[:3]))


LAMBDAS = [0.99, 0.995, 0.998, 0.999, 0.9995, 0.9998, 0.9999]


def regression(module, xs, lrs, ks=range(5)):
    """A l'interieur d'un enregistrement : ln(1/rho_g) (rho pris 30 pas avant le groupe g) contre la moyenne mobile exponentielle, de decroissance lambda,
    de l'indicateur d'oscillation (1 pendant un groupe), evaluee au meme instant ; plus deux candidats simples : la duree du calme avant g, la duree du groupe precedent."""
    for lr in lrs:
        r_l = {lam: [] for lam in LAMBDAS}; b_l = {lam: [] for lam in LAMBDAS}; r_c = []; r_d = []
        for k in ks:
            rec, groupes = enregistrement(module, lr, k, xs)
            if rec is None:
                continue
            a = np.array([g[0] for g in groupes]); dur = np.array([g[1] - g[0] + 1 for g in groupes])
            nrec = len(rec)
            ind = np.zeros(nrec)
            for ai, di in zip(a, dur):
                ind[ai:ai + di] = 1.0
            t = a[9:] - 30
            y = np.log(rec[t, 4] / rec[t, 3])
            calme = (a[9:] - (a[8:-1] + dur[8:-1])).astype(float); dprec = dur[8:-1].astype(float)
            for lam in LAMBDAS:
                F = np.empty(nrec); s = ind[0]
                for i in range(nrec):
                    s = lam * s + (1.0 - lam) * ind[i]
                    F[i] = s
                x = F[t]
                r = np.corrcoef(x, y)[0, 1]
                r_l[lam].append(r); b_l[lam].append(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
            r_c.append(np.corrcoef(calme, y)[0, 1]); r_d.append(np.corrcoef(dprec, y)[0, 1])
        if not r_c:
            print(f"lr {lr}: effondre")
            continue
        print(f"lr {lr} ({len(r_c)} departs) : r(ln 1/rho, F_lambda) " + " ".join(f"{lam}:{np.mean(r_l[lam]):+.3f}" for lam in LAMBDAS) + f" ; r(calme avant) {np.mean(r_c):+.3f} ; r(duree du groupe precedent) {np.mean(r_d):+.3f}")
        best = max(LAMBDAS, key=lambda lam: abs(np.mean(r_l[lam])))
        print(f"      pente de ln(1/rho) sur F_lambda (par unite de F) : " + " ".join(f"{lam}:{np.mean(b_l[lam]):+.4f}" for lam in LAMBDAS) + f" ; lambda de plus forte correlation {best}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "regression":
        regression(sys.argv[2], sys.argv[3], [float(v) for v in sys.argv[4:]])
    elif mode == "libre":
        libre()
    elif mode == "essai":
        essai(sys.argv[2], sys.argv[3], [float(v) for v in sys.argv[4:]])
    elif mode == "enregistrer":
        enregistrer(sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5]))
    elif mode == "config":
        config(sys.argv[2], sys.argv[3], [float(v) for v in sys.argv[4:]])
