"""Tour 86 : la derive de la lecture continue-t-elle apres la salve 8 ? Enregistrement long, estimateur sans jacobien.

Il demande : la derive de lr 0,3 continue-t-elle de monter au-dela de la salve 8, ou se stabilise-t-elle ?
La lecture est lr p_z au franchissement du niveau 38, p_z = 1/(sqrt(v_e3) + eps) + 1/(sqrt(v_e3) + 26 eps) ; sur les huit salves lues, sa pente contre ln sqrt(v_e3) est -0,94 a -0,96 (verifier_tour86_derive.py). On prend donc comme estimateur lr p_z calcule avec sqrt(v_e3) a un decalage fixe avant le debut de chaque groupe, pour TOUTES les salves de l'enregistrement (300 000 pas), pas seulement huit. L'estimateur est calibre sur les lectures du tour 85 pour les memes salves (memes graines, memes groupes).

Modes :
  enregistrer <lr> <k> [<nrec>] : reduction (module a lr runtime, x = 1e-4, eps/lr = 2e-9, moments a l'echelle), groupes comme verifier_tour78_lr, estimateur aux decalages 1, 30, 70, 140 pas avant chaque groupe, calibration sur D:/tmp/t85_x1e-4_lr<lr>_k<k>.log, ecrit D:/tmp/t86_lr<lr>_k<k>.npz
  synthese : lit les .npz, tableau par lr (derive par fenetre de salves, valeur par fenetre, test de segments)
"""

import importlib
import math
import os
import re
import sys

import numpy as np
from mpmath import mp, mpf

sys.path.insert(0, ".")
from verifier_tour75_salves_depart import etat_base, PLI  # noqa: E402
from verifier_tour78_lr import pz  # noqa: E402

mp.dps = 50
D = "D:/tmp"
LRS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
DECALAGES = (1, 30, 70, 140)
MOTIF = re.compile(r"salve (\d+) : calme avant (\d+) pas ; duree (\d+) pas ; lr p_z ([0-9.e+]+)")


def groupes_de(u):
    D2 = u[2:] - 2 * u[1:-1] + u[:-2]
    A = np.abs(D2) / 4.0
    E = np.maximum(A[:-1], A[1:])
    idx = np.nonzero(E > 3e-4)[0]
    coupe = np.nonzero(np.diff(idx) > 30)[0]
    debuts = np.concatenate([[0], coupe + 1]); fins = np.concatenate([coupe, [len(idx) - 1]])
    return [(int(idx[a]), int(idx[b])) for a, b in zip(debuts, fins)]


def enregistrer(lr, kidx, nrec):
    redlib = importlib.import_module("reduction_numba_tour71_lrvar")
    redlib.LRV = lr
    eps = float(f"{2e-9 * lr:.4e}")
    S1 = etat_base()
    delta = float(PLI - mpf("1e-4"))
    ks = [int(a) for a in np.random.default_rng(7).integers(0, 4444, 16)]
    rng = np.random.default_rng(11 + kidx)
    w = redlib.copie(S1)
    w["m"] = w["m"] * (lr / 0.05)
    w["v"] = w["v"] * (lr / 0.05) ** 2
    redlib.avancer(w, float(PLI - mpf("1e-6")), 20000 + ks[kidx], eps=eps, cross=2.0, stop=False)
    w["x"] = w["x"] * (1.0 + 1e-9 * rng.standard_normal(6))
    att = int(os.environ.get("TOUR86_ATT", "10000"))   # attente apres le passage de x = 1e-6 a 1e-4, avant l'enregistrement (10 000 : protocole du tour 85)
    redlib.avancer_obs(w, delta, att, np.zeros((1, 7)), eps=eps)
    rec = np.zeros((nrec, 7))
    first, maxrb, rb, done = redlib.avancer_obs(w, delta, nrec, rec, eps=eps)
    if maxrb > 0.9:
        print("effondre")
        return
    groupes = groupes_de(rec[:, 2])
    s_e3 = rec[:, 3]; s_l3 = rec[:, 4]; z3 = rec[:, 1]
    G = len(groupes)
    a = np.array([g[0] for g in groupes]); dur = np.array([g[1] - g[0] + 1 for g in groupes])
    P = np.full((G, len(DECALAGES)), np.nan)
    for j, off in enumerate(DECALAGES):
        for g in range(1, G):
            t = a[g] - off
            if t >= 0:
                P[g, j] = lr * pz(s_e3[t], eps)
    # pendant le calme sqrt(v_e3) et sqrt(v_l3) decroissent au meme rythme (0,05 % par pas) : leur rapport ne depend pas de l'instant de lecture
    RHO = np.full((G, len(DECALAGES)), np.nan)
    SLD = np.full((G, len(DECALAGES)), np.nan)
    for j, off in enumerate(DECALAGES):
        for g in range(1, G):
            t = a[g] - off
            if t >= 0:
                RHO[g, j] = s_e3[t] / s_l3[t]
                SLD[g, j] = s_l3[t]
    SL = SLD[:, 2]
    Z3 = np.array([z3[a[g] - 70] if g >= 1 and a[g] - 70 >= 0 else np.nan for g in range(G)])
    # calibration sur le journal du tour 85 (memes groupes : l'indice de salve du journal est l'indice de groupe)
    cal = []
    try:
        if att != 10000:
            raise FileNotFoundError   # le journal du tour 85 ne decrit que l'attente de 10 000 pas
        t_ = open(f"{D}/t85_x1e-4_lr{lr}_k{kidx}.log", encoding="utf-8", errors="replace").read()
        lect = [(int(g_), float(r_)) for g_, c_, d_, r_ in MOTIF.findall(t_)]
    except FileNotFoundError:
        lect = []
    print(f"lr {lr} k{kidx} : {G} groupes sur {nrec} pas ; {len(lect)} lectures du tour 85 pour la calibration")
    for j, off in enumerate(DECALAGES):
        if lect:
            rat = np.array([math.log(r_ / P[g_, j]) for g_, r_ in lect])
            cal.append((float(rat.mean()), float(rat.std(ddof=1)) if len(rat) > 1 else float("nan")))
            print(f"   decalage {off:4d} : ln(lecture / estimateur) moyenne {100 * rat.mean():+.4f} %, sd {100 * rat.std(ddof=1):.4f} % ; plage {100 * (rat.max() - rat.min()):.4f} %")
            rr = np.array([math.log(r_ * RHO[g_, j]) for g_, r_ in lect])
            rs = np.array([math.log(r_ * RHO[g_, j] * SLD[g_, j]) for g_, r_ in lect])
            print(f"                  ln(lecture x rho) sd {100 * rr.std(ddof=1):.4f} % plage {100 * (rr.max() - rr.min()):.4f} % ; ln(lecture x sqrt(v_e3)) sd {100 * np.array([math.log(r_ * s_e3[a[g_] - off]) for g_, r_ in lect]).std(ddof=1):.4f} %")
    # estimateur retenu : lecture = C / rho avec C calibre sur les lectures du tour 85 (decalage 30 ; rho ne depend pas du decalage)
    PR = np.full(G, np.nan)
    if lect:
        C = math.exp(np.mean([math.log(r_ * RHO[g_, 1]) for g_, r_ in lect]))
        PR = C / RHO[:, 1]
        ecart = np.array([100 * (PR[g_] / r_ - 1) for g_, r_ in lect])
        print(f"   estimateur C/rho : ecart a la lecture, par salve lue (%) : " + " ".join(f"{v:+.4f}" for v in ecart))
    np.savez(f"{D}/t86_lr{lr}_k{kidx}.npz" if att == 10000 else f"{D}/t86_att{att}_lr{lr}_k{kidx}.npz", a=a, dur=dur, P=P, PR=PR, RHO=RHO, SLD=SLD, SL=SL, Z3=Z3, lect=np.array(lect) if lect else np.zeros((0, 2)), nrec=nrec)


def pente(g, y):
    g = np.asarray(g, float); y = np.asarray(y, float)
    ok = np.isfinite(y)
    g, y = g[ok], y[ok]
    if len(g) < 3:
        return float("nan")
    b = ((g - g.mean()) * (y - y.mean())).sum() / ((g - g.mean()) ** 2).sum()
    return 100 * b / y.mean()


def synthese():
    sys.path.insert(0, ".")
    from verifier_tour84_coupures import signif_segments
    fenetres = [(1, 8), (9, 40), (41, 120), (121, 400), (401, 10 ** 6)]
    j = 2   # decalage 70 (celui de la fenetre de balayage du tour 85) ; la calibration dit lequel colle le mieux
    donnees = {}
    for lr in LRS:
        donnees[lr] = {}
        for k in range(5):
            try:
                z = np.load(f"{D}/t86_lr{lr}_k{k}.npz")
            except FileNotFoundError:
                continue
            donnees[lr][k] = z
    print("pente de l'estimateur contre l'indice de salve, par fenetre de salves (% de la moyenne de la fenetre par salve), moyenne sur les departs [min .. max] :")
    for lr in LRS:
        if not donnees[lr]:
            continue
        ligne = []
        for lo, hi in fenetres:
            ps = []
            for k, z in donnees[lr].items():
                P = z["PR"]; G = len(P)
                g = np.arange(G)
                sel = (g >= lo) & (g <= min(hi, G - 1))
                ps.append(pente(g[sel], P[sel]))
            ps = np.array(ps)
            ligne.append(f"[{lo}-{hi if hi < 10 ** 6 else 'fin'}] {np.nanmean(ps):+.4f} [{np.nanmin(ps):+.4f} .. {np.nanmax(ps):+.4f}]")
        print(f"  lr {lr}: " + " ; ".join(ligne))
    print()
    print("niveau de l'estimateur par fenetre de salves, relatif a la moyenne des salves 1-8 du meme depart (%), moyenne sur les departs [min .. max] :")
    for lr in LRS:
        if not donnees[lr]:
            continue
        ligne = []
        for lo, hi in fenetres[1:]:
            rel = []
            for k, z in donnees[lr].items():
                P = z["PR"]; G = len(P)
                base = np.nanmean(P[1:9])
                rel.append(100 * (np.nanmean(P[lo:min(hi, G - 1) + 1]) / base - 1))
            ligne.append(f"[{lo}-{hi if hi < 10 ** 6 else 'fin'}] {np.mean(rel):+.3f} [{np.min(rel):+.3f} .. {np.max(rel):+.3f}]")
        print(f"  lr {lr}: " + " ; ".join(ligne))
    print()
    print("blocs de 50 salves (premiers 400 groupes) : moyenne de l'estimateur par bloc, relative a la moyenne de tout l'enregistrement du depart (%), par depart")
    for lr in LRS:
        for k, z in donnees[lr].items():
            P = z["PR"]; G = len(P); tout = np.nanmean(P[1:])
            blocs = [100 * (np.nanmean(P[b:b + 50]) / tout - 1) for b in range(1, min(G, 401), 50)]
            print(f"  lr {lr} k{k}: " + " ".join(f"{v:+.3f}" for v in blocs))
    print()
    print("niveau de tout l'enregistrement (salves >= 9) et test de segments, ET = dispersion sur les cinq departs :")
    for nom, deb in (("salves >= 9", 9), ("salves >= 41", 41), ("salves 1-8 (estimateur)", 1)):
        lrs = []; m = []; se = []
        for lr in LRS[1:]:
            vals = []
            for k, z in donnees[lr].items():
                P = z["PR"]; G = len(P)
                vals.append(np.nanmean(P[deb:9]) if deb == 1 else np.nanmean(P[deb:]))
            if len(vals) >= 2:
                lrs.append(lr); m.append(np.mean(vals)); se.append(np.std(vals, ddof=1) / math.sqrt(len(vals)))
        lrs = np.array(lrs); m = np.array(m); se = np.array(se)
        print(f"  {nom} : moyenne par lr {' '.join(f'{v:.5e}' for v in m)} ; ET entre departs (%) {' '.join(f'{100 * s / v:.3f}' for s, v in zip(se, m))}")
        for etiq, ee in (("ET entre departs seule", se), ("+ plancher 0,1 %", np.hypot(se, 0.001 * m))):
            r = signif_segments(lrs, m, ee, bas=(0.1, 0.2), haut=(0.2, 0.3))
            print(f"     {etiq} : bas {r[0]:+.4f} +- {r[1]:.4f}, haut {r[2]:+.4f} +- {r[3]:.4f} ; difference {r[0] - r[2]:+.4f} ; {r[4]:.1f} sigma")
    print()
    print("effet d'un groupe long (> 300 pas) sur l'estimateur de la salve suivante, relatif a la moyenne locale (%), lr 0,3 :")
    for lr in (0.25, 0.3):
        for k, z in donnees[lr].items():
            P = z["PR"]; dur = z["dur"]
            apres = [g for g in range(2, len(P)) if dur[g - 1] > 300 and np.isfinite(P[g])]
            autres = [g for g in range(2, len(P)) if dur[g - 1] <= 300 and np.isfinite(P[g])]
            if apres:
                print(f"  lr {lr} k{k}: {len(apres)} salves apres un groupe long, ecart moyen {100 * (np.mean(P[apres]) / np.mean(P[autres]) - 1):+.4f} % ; {len(autres)} autres")


def attente():
    """Pente de ln(1/rho) sur les salves 1-8 (puis 9-40) selon l'attente avant l'enregistrement."""
    for lr in (0.2, 0.3):
        for att in (10000, 30000, 100000):
            sl8, sl40, niv = [], [], []
            for k in range(5):
                nom = f"{D}/t86_lr{lr}_k{k}.npz" if att == 10000 else f"{D}/t86_att{att}_lr{lr}_k{k}.npz"
                try:
                    z = np.load(nom)
                except FileNotFoundError:
                    continue
                inv = 1 / z["RHO"][:, 1]; G = len(inv); g = np.arange(G)
                sl8.append(pente(g[1:9], inv[1:9])); sl40.append(pente(g[9:41], inv[9:41]))
                niv.append(100 * (np.nanmean(inv[1:9]) / np.nanmean(inv[41:]) - 1))
            if sl8:
                print(f"lr {lr} attente {att:6d} : {len(sl8)} departs ; pente salves 1-8 {np.mean(sl8):+.4f} %/salve [{np.min(sl8):+.4f} .. {np.max(sl8):+.4f}] ; pente salves 9-40 {np.mean(sl40):+.4f} [{np.min(sl40):+.4f} .. {np.max(sl40):+.4f}] ; moyenne des salves 1-8 relative aux salves >= 41 : {np.mean(niv):+.3f} % [{np.min(niv):+.3f} .. {np.max(niv):+.3f}]")


def validation():
    """Relectures au jacobien de salves tardives (t86_relecture_*) contre l'estimateur C/rho, meme salve."""
    tous = {}
    for lr in (0.15, 0.3):
        for k in (0, 1):
            z = np.load(f"{D}/t86_lr{lr}_k{k}.npz")
            PR, dur = z["PR"], z["dur"]
            for g0 in (50, 200, 500):
                t = open(f"{D}/t86_relecture_lr{lr}_k{k}_g{g0}.log", encoding="utf-8", errors="replace").read()
                lect = [(int(g_), int(d_), float(r_)) for g_, c_, d_, r_ in MOTIF.findall(t)]
                ec = [(g_, 100 * (PR[g_] / r_ - 1), d_, dur[g_ - 1]) for g_, d_, r_ in lect]
                print(f"lr {lr} k{k} groupe {g0} : " + " ; ".join(f"g{g_} {e:+.4f} % (duree du groupe {d_}, du precedent {dp})" for g_, e, d_, dp in ec))
                tous[(lr, k, g0)] = [e for _, e, _, _ in ec]
    for lr in (0.15, 0.3):
        v = np.array([e for (l, k, g0), es in tous.items() if l == lr for e in es])
        print(f"lr {lr}: {len(v)} salves relues, ecart C/rho - lecture : moyenne {v.mean():+.4f} %, sd {v.std(ddof=1):.4f} %, max |ecart| {np.abs(v).max():.4f} % ; sans les ecarts > 0,3 % : sd {v[np.abs(v) < 0.3].std(ddof=1):.4f} % ({int((np.abs(v) < 0.3).sum())} salves)")


def echantillon():
    """Lectures directes d'un echantillon aleatoire de salves (t86_alea_*) : fraction de salves en retard, moyennes, test de segments."""
    from verifier_tour84_coupures import signif_segments
    res = {}
    for lr in LRS[1:]:
        par_depart = []
        for k in range(5):
            try:
                t = open(f"{D}/t86_alea_lr{lr}_k{k}.log", encoding="utf-8", errors="replace").read()
            except FileNotFoundError:
                continue
            z_ = np.load(f"{D}/t86_lr{lr}_k{k}.npz")
            PR, dur_ = z_["PR"], z_["dur"]
            lect = [(int(g_), float(r_)) for g_, c_, d_, r_ in MOTIF.findall(t)]
            if lect:
                par_depart.append(np.array([(g_, r_, PR[g_], dur_[g_], dur_[g_ - 1]) for g_, r_ in lect]))
        res[lr] = par_depart
    print("echantillon aleatoire de salves, lecture directe : nombre de lectures, salves en retard (C/rho depasse la lecture de plus de 0,3 %), moyennes (direct tout, direct sans retard, C/rho sur les memes salves), ET entre departs (%)")
    stats = {}
    for lr in LRS[1:]:
        pd_ = res[lr]
        if not pd_:
            continue
        n = sum(len(a_) for a_ in pd_)
        ec = np.concatenate([100 * (a_[:, 2] / a_[:, 1] - 1) for a_ in pd_])
        retard = int((ec > 0.3).sum()); autre = int((np.abs(ec) > 0.3).sum()) - retard
        m_all = np.array([a_[:, 1].mean() for a_ in pd_])
        m_ok = np.array([a_[100 * (a_[:, 2] / a_[:, 1] - 1) <= 0.3, 1].mean() for a_ in pd_])
        m_pr = np.array([a_[:, 2].mean() for a_ in pd_])
        stats[lr] = (m_all, m_ok, m_pr)
        print(f"  lr {lr}: {n} lectures dans {len(pd_)} departs ; en retard {retard} ({100 * retard / n:.1f} %), autres ecarts > 0,3 % : {autre} ; ecart sd sans les > 0,3 % {ec[np.abs(ec) <= 0.3].std(ddof=1):.4f} % ; moyennes {m_all.mean():.5e} | {m_ok.mean():.5e} | {m_pr.mean():.5e} ; ET entre departs {100 * m_all.std(ddof=1) / math.sqrt(len(m_all)) / m_all.mean():.3f} | {100 * m_ok.std(ddof=1) / math.sqrt(len(m_ok)) / m_ok.mean():.3f} | {100 * m_pr.std(ddof=1) / math.sqrt(len(m_pr)) / m_pr.mean():.3f}")
        print(f"        par depart : retards " + " ".join(str(int((100 * (a_[:, 2] / a_[:, 1] - 1) > 0.3).sum())) + "/" + str(len(a_)) for a_ in pd_) + " ; moyenne directe " + " ".join(f"{v:.5e}" for v in m_all))
    for lr in (0.25, 0.3):
        A = np.concatenate(res[lr])
        ec = 100 * (A[:, 2] / A[:, 1] - 1)
        print(f"lr {lr}: percentiles de l'ecart C/rho - lecture (%) 1 5 25 50 75 95 99 : " + " ".join(f"{v:+.3f}" for v in np.percentile(ec, [1, 5, 25, 50, 75, 95, 99])))
        for nom, masque in (("groupe precedent long (> 300)", A[:, 4] > 300), ("groupe precedent court", A[:, 4] <= 300), ("groupe propre long (> 300)", A[:, 3] > 300), ("groupe propre court", A[:, 3] <= 300)):
            if masque.sum():
                e = ec[masque]
                print(f"   {nom} : {int(masque.sum())} lectures ; ecart moyen {e.mean():+.3f} %, mediane {np.median(e):+.3f} % ; > +0,3 % : {int((e > 0.3).sum())} ; < -0,3 % : {int((e < -0.3).sum())}")
        # lectures lues apres les huit premieres salves de l'enregistrement : le groupe 9-40 contre le reste
        for lo, hi in ((9, 40), (41, 200), (201, 559)):
            m_ = (A[:, 0] >= lo) & (A[:, 0] <= hi)
            if m_.sum():
                e = ec[m_]
                print(f"   groupes {lo}-{hi} : {int(m_.sum())} lectures ; ecart moyen {e.mean():+.3f} %, mediane {np.median(e):+.3f} %")
    lrs = np.array([l for l in LRS[1:] if l in stats])
    if len(lrs) == 5:
        for nom, i in (("direct, toutes les lectures", 0), ("direct, sans les salves en retard", 1), ("C/rho sur les memes salves", 2)):
            m = np.array([stats[l][i].mean() for l in lrs]); se = np.array([stats[l][i].std(ddof=1) / math.sqrt(len(stats[l][i])) for l in lrs])
            for etiq, ee in (("ET entre departs seule", se), ("+ plancher 0,1 %", np.hypot(se, 0.001 * m))):
                r = signif_segments(lrs, m, ee, bas=(0.1, 0.2), haut=(0.2, 0.3))
                print(f"test de segments, {nom}, {etiq} : bas {r[0]:+.4f} +- {r[1]:.4f}, haut {r[2]:+.4f} +- {r[3]:.4f} ; difference {r[0] - r[2]:+.4f} ; {r[4]:.1f} sigma ; haut au-dessus de plat {r[2] / r[3]:.1f} ET")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "liste":
        # liste aleatoire de groupes pour TOUR86_GLIST : liste <k> <n>
        g_ = np.random.default_rng(860 + int(sys.argv[2])).choice(np.arange(9, 560), size=int(sys.argv[3]), replace=False)
        print(",".join(str(int(v)) for v in sorted(g_)))
    elif mode == "echantillon":
        echantillon()
    elif mode == "enregistrer":
        enregistrer(float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 300000)
    elif mode == "validation":
        validation()
    elif mode == "attente":
        attente()
    else:
        synthese()
