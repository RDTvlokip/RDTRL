"""Tour 91 : la carte de retour de rho, burst par burst.

Il dit : dans un enregistrement, ln(1/rho) baisse avec F (pente -0,030 a -0,058 par unite de F), entre lr M monte avec f (+0,049 par unite de f entre f 0,35 et 0,45) ; une formule a amplitudes A et a fixes donne le meme signe aux deux ; si elle s'ajuste, A ou a bouge avec lr, et ce mouvement serait la marche. |S_F| est plat de lr 0,15 a 0,2 puis tombe. rho est constant dans un calme, donc tout changement de rho a lieu dans une salve : d_g = ln rho_(g+1) - ln rho_g est l'increment de la salve g, lisible sur le rho_g sauvegarde ; en regime permanent sa moyenne est nulle, l'information est dans sa dependance de rho_g et de la duree de la salve, et le point fixe de cette carte fixe M. d_g contre rho_g donne-t-il une droite propre par lr, et sa pente bouge-t-elle a 0,2 ?

Modes :
  calme : sur un enregistrement regenere (lr 0,1 / 0,2 / 0,3, depart 0), variation de ln rho cumulee en valeur absolue pendant les calmes et pendant les salves
  carte : regression de d_g sur ln rho_g (et sur la duree de la salve, ln sqrt(v_l3), les calmes avant et apres), par lr, a partir des .npz du tour 86-88 (t86_lr<lr>_k<k>.npz)
"""

import math
import sys

import numpy as np

sys.path.insert(0, ".")
from verifier_tour90_memoire import enregistrement  # noqa: E402

D = "D:/tmp"
GRILLE = [0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]
B2 = 0.999


def calme():
    for lr in (0.1, 0.2, 0.3):
        rec, groupes = enregistrement("reduction_numba_tour71_lrvar", lr, 0, "1e-4")
        a = np.array([g[0] for g in groupes]); fin = np.array([g[1] for g in groupes])
        lrho = np.log(rec[:, 3] / rec[:, 4])
        d_calme = []; d_salve = []
        for g in range(10, len(a) - 1):
            i0, i1 = fin[g - 1] + 1, a[g] - 1
            j0, j1 = a[g], fin[g]
            d_calme.append(lrho[i1] - lrho[i0] if i1 > i0 else 0.0)
            d_salve.append(lrho[j1 + 1] - lrho[j0 - 1])
        d_calme = np.array(d_calme); d_salve = np.array(d_salve)
        print(f"lr {lr}: {len(d_salve)} salves ; somme |variation de ln rho| : calmes {np.abs(d_calme).sum():.5f}, salves {np.abs(d_salve).sum():.5f}, rapport calmes / salves {np.abs(d_calme).sum() / np.abs(d_salve).sum():.4f} ; "
              f"moyenne signee par salve {d_salve.mean():+.6f}, par calme {d_calme.mean():+.6f} ; sd d'une salve {d_salve.std():.5f}, d'un calme {d_calme.std():.5f}")


def charger(lr, k):
    try:
        z = np.load(f"{D}/t86_lr{lr}_k{k}.npz")
    except FileNotFoundError:
        return None
    return z


def regresser(X, y):
    """MCO avec constante ; retourne coefficients (constante en premier), R2."""
    A = np.column_stack([np.ones(len(y))] + [x - x.mean() for x in X])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ c
    r2 = 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return c, r2


def carte():
    resume = {}
    for lr in GRILLE:
        recs = []
        for k in range(10):
            z = charger(lr, k)
            if z is None:
                continue
            a = z["a"]; dur = z["dur"].astype(float); rho = z["RHO"][:, 1]; sl = z["SLD"][:, 1]
            G = len(a)
            g = np.arange(9, G - 1)
            x = np.log(rho[g]); d = np.log(rho[g + 1]) - x
            cavant = (a[g] - (a[g - 1] + dur[g - 1])).astype(float); capres = (a[g + 1] - (a[g] + dur[g])).astype(float)
            ok = np.isfinite(x) & np.isfinite(d)
            x, d, dd, ls = x[ok], d[ok], dur[g][ok], np.log(sl[g][ok])
            cavant, capres = cavant[ok], capres[ok]
            cyc = 300000 / G
            c1, r2_1 = regresser([x], d)
            c2, r2_2 = regresser([x, dd, ls, cavant], d)
            r = np.corrcoef(x, d)[0, 1]
            # linearite : moyennes de d par quintile de x contre la droite
            q = np.quantile(x, [0.2, 0.4, 0.6, 0.8])
            idx = np.digitize(x, q)
            binned = np.array([d[idx == b].mean() for b in range(5)]); xb = np.array([x[idx == b].mean() for b in range(5)])
            pred = c1[0] + c1[1] * (xb - x.mean())
            r2_bin = 1 - ((binned - pred) ** 2).sum() / ((binned - binned.mean()) ** 2).sum()
            phi = np.corrcoef(x[:-1], x[1:])[0, 1]
            recs.append({"beta": c1[1], "r": r, "r2_lin": r2_1, "r2_multi": r2_2, "cyc": cyc, "sd_x": x.std(), "sd_d": d.std(), "moy_d": d.mean(), "phi": phi, "r2_bin": r2_bin,
                         "b_dur": c2[2], "b_ls": c2[3], "b_cav": c2[4], "b_x_multi": c2[1], "n": len(x)})
        if recs:
            resume[lr] = {n: np.array([r[n] for r in recs]) for n in recs[0]}
    print("carte d_g = ln rho_(g+1) - ln rho_g contre ln rho_g, par lr (moyenne sur les departs [min .. max]) :")
    print("   lr : pente beta ; r ; R2 lineaire ; cycle moyen (pas) ; beta du modele -(1 - b2^cycle) ; rapport ; autocorrelation de ln rho a un salve ; moyenne de d (identite : 0)")
    modele = {}
    for lr in GRILLE:
        s = resume[lr]
        bm = -(1 - B2 ** s["cyc"].mean())
        modele[lr] = bm
        print(f"   {lr}: beta {s['beta'].mean():+.4f} [{s['beta'].min():+.4f} .. {s['beta'].max():+.4f}] ; r {s['r'].mean():+.3f} ; R2 {s['r2_lin'].mean():.3f} ; cycle {s['cyc'].mean():.0f} ; modele {bm:+.4f} ; rapport {s['beta'].mean() / bm:.3f} ; phi {s['phi'].mean():.3f} (1 - phi = {1 - s['phi'].mean():.3f}) ; moyenne de d {s['moy_d'].mean():+.2e} ; sd de ln rho {100 * s['sd_x'].mean():.3f} % ; n {int(s['n'].mean())} ({len(s['beta'])} departs)")
    print("linearite (R2 des moyennes par quintile autour de la droite) et regression multiple (d_g sur ln rho_g, duree de la salve, ln sqrt(v_l3), calme avant) :")
    for lr in GRILLE:
        s = resume[lr]
        print(f"   {lr}: R2 des quintiles {s['r2_bin'].mean():.3f} ; R2 lineaire {s['r2_lin'].mean():.3f} -> R2 multiple {s['r2_multi'].mean():.3f} ; coefficients : ln rho {s['b_x_multi'].mean():+.4f}, duree {s['b_dur'].mean():+.3e} par pas, ln sqrt(v_l3) {s['b_ls'].mean():+.4f}, calme avant {s['b_cav'].mean():+.3e} par pas")
    # coude de |beta|
    lrs = np.array(GRILLE); y = np.log(np.abs([resume[l]["beta"].mean() for l in GRILLE]))
    x = np.log(lrs)
    A = np.vstack([x, np.ones_like(x)]).T
    c = np.linalg.lstsq(A, y, rcond=None)[0]; rss0 = float(((y - A @ c) ** 2).sum())
    best = None
    for xk in np.linspace(x[1], x[-2], 400):
        B = np.vstack([x, np.ones_like(x), np.maximum(x - xk, 0.0)]).T
        cc = np.linalg.lstsq(B, y, rcond=None)[0]
        rss = float(((y - B @ cc) ** 2).sum())
        if best is None or rss < best[0]:
            best = (rss, math.exp(xk), cc[0], cc[0] + cc[2])
    print(f"coude de ln|beta| contre ln lr (9 lr) : lr_coude {best[1]:.4f} ; pente basse {best[2]:+.3f}, haute {best[3]:+.3f} ; RSS coude {best[0]:.3e} contre droite {rss0:.3e} (rapport {best[0] / rss0:.3f})")
    print("beta par intervalle, et exposants locaux de |beta| (en ln lr) : " + " ".join(f"{a}->{b}: {math.log(abs(resume[b]['beta'].mean()) / abs(resume[a]['beta'].mean())) / math.log(b / a):+.3f}" for a, b in zip(GRILLE[:-1], GRILLE[1:])))


def injection_depart(lr, k):
    """Injections nettes I_e, I_l par cycle (de a_g - 1 a a_(g+1) - 1), poids omega et cible rho* ; retourne un dictionnaire de moyennes, ou None."""
    rec, groupes = enregistrement("reduction_numba_tour71_lrvar", lr, k, "1e-4")
    if rec is None:
        return None
    a = np.array([g[0] for g in groupes])
    ve = rec[:, 3] ** 2; vl = rec[:, 4] ** 2
    g = np.arange(10, len(a) - 1)
    i0 = a[g] - 1; i2 = a[g + 1] - 1; T = i2 - i0
    th = B2 ** T
    Ie = ve[i2] - th * ve[i0]; Il = vl[i2] - th * vl[i0]
    rho0 = np.sqrt(ve[i0] / vl[i0]); rho2 = np.sqrt(ve[i2] / vl[i2])
    omega = th * vl[i0] / vl[i2]
    cible2 = Ie / Il
    ident = np.abs(rho2 ** 2 / (omega * rho0 ** 2 + (1 - omega) * cible2) - 1).max()
    ok = (Ie > 0) & (Il > 0)
    w1 = 1 - omega
    lnrho = np.log(rho0); lncib = 0.5 * np.log(cible2)
    dur_g = np.array([groupes[j][1] - groupes[j][0] + 1 for j in g], float)
    gam, alpha0 = np.polyfit(dur_g[ok], lncib[ok], 1)   # post hoc (tour 91) : dependance de la cible de la salve vis-a-vis de sa duree
    gam_e = np.polyfit(dur_g[ok], np.log(Ie[ok]), 1)[0]; gam_l = np.polyfit(dur_g[ok], np.log(Il[ok]), 1)[0]   # ... et de chacune des deux injections
    return {"gamma_e": gam_e, "gamma_l": gam_l, "ln_Ie": np.log(Ie[ok]).mean(), "ln_Il": np.log(Il[ok]).mean(), "gamma": gam, "alpha0": alpha0, "dur_moy": dur_g[ok].mean(), "identite": ident, "positives": float(ok.mean()), "omega": omega.mean(), "sd_omega": omega.std(), "lnrho": lnrho[ok].mean(), "lncible": lncib[ok].mean(),
            "lncible_w": (w1[ok] * lncib[ok]).sum() / w1[ok].sum(), "lnrho_w": (w1[ok] * lnrho[ok]).sum() / w1[ok].sum(),
            "rho2_w": (w1[ok] * rho0[ok] ** 2).sum() / w1[ok].sum(), "cible2_w": (w1[ok] * cible2[ok]).sum() / w1[ok].sum(),
            "sd_cible": lncib[ok].std(), "sd_rho": lnrho[ok].std(), "cyc": 300000 / len(a), "n": int(ok.sum()),
            "corr_cible_dur": float(np.corrcoef(lncib[ok], np.array([groupes[j][1] - groupes[j][0] + 1 for j in g])[ok])[0, 1])}


def injection(lr):
    for k in range(5):
        r = injection_depart(lr, k)
        if r is None:
            print(f"lr {lr} k{k}: effondre")
            continue
        print(f"INJ lr {lr} k {k} " + " ".join(f"{n} {v:.10g}" for n, v in r.items()))


def injection_synthese():
    import re
    res = {}
    for lr in GRILLE:
        try:
            t = open(f"{D}/t91_inj_lr{lr}.log", encoding="utf-8", errors="replace").read()
        except FileNotFoundError:
            continue
        L = []
        for ligne in t.split("\n"):
            if ligne.startswith("INJ"):
                parts = ligne.split()[5:]
                L.append({parts[i]: float(parts[i + 1]) for i in range(0, len(parts), 2)})
        if L:
            res[lr] = {n: np.array([x[n] for x in L]) for n in L[0]}
    lrs = [l for l in GRILLE if l in res]
    print("lr : identite (max |ecart relatif|), part des cycles a injections positives, <omega> (b2^C), sd omega, <ln rho>, <ln cible>, <ln cible>_(1-omega), <ln rho>_(1-omega), correlation (cible, duree)")
    for lr in lrs:
        s = res[lr]
        print(f"   {lr}: identite {s['identite'].max():.1e} ; positives {s['positives'].min():.4f} ; omega {s['omega'].mean():.4f} (b2^C {B2 ** s['cyc'].mean():.4f}) sd {s['sd_omega'].mean():.4f} ; ln rho {s['lnrho'].mean():+.6f} ; ln cible {s['lncible'].mean():+.6f} ; ln cible_w {s['lncible_w'].mean():+.6f} ; ln rho_w {s['lnrho_w'].mean():+.6f} ; sd ln cible {100 * s['sd_cible'].mean():.3f} % ; sd ln rho {100 * s['sd_rho'].mean():.3f} % ; corr(cible, duree) {s['corr_cible_dur'].mean():+.3f}")
    def exposants(nom, vals):
        return [(math.log(vals[i + 1] / vals[i]) / math.log(lrs[i + 1] / lrs[i])) for i in range(len(lrs) - 1)]
    def marche(ex):
        L = [math.log(lrs[i + 1] / lrs[i]) for i in range(len(lrs) - 1)]
        def moy(lo, hi):
            sel = [i for i in range(len(lrs) - 1) if lrs[i] >= lo - 1e-9 and lrs[i + 1] <= hi + 1e-9]
            return sum(ex[i] * L[i] for i in sel) / sum(L[i] for i in sel)
        return moy(0.15, 0.2), moy(0.2, 0.3)
    print("post hoc : pente de ln I_e et de ln I_l sur la duree de la salve (par pas), ln I moyens, par lr :")
    for lr in lrs:
        s = res[lr]
        print(f"   {lr}: gamma_e {s['gamma_e'].mean():+.3e} [{s['gamma_e'].min():+.2e} .. {s['gamma_e'].max():+.2e}] ; gamma_l {s['gamma_l'].mean():+.3e} [{s['gamma_l'].min():+.2e} .. {s['gamma_l'].max():+.2e}] ; ln I_e {s['ln_Ie'].mean():+.4f} ; ln I_l {s['ln_Il'].mean():+.4f}")
    print("post hoc : ln cible = alpha0 + gamma x duree, par lr (moyenne des departs) :")
    for lr in lrs:
        s = res[lr]
        print(f"   {lr}: gamma {s['gamma'].mean():+.3e} par pas [{s['gamma'].min():+.2e} .. {s['gamma'].max():+.2e}] ; duree moyenne {s['dur_moy'].mean():.1f} ; terme gamma x duree {100 * (s['gamma'] * s['dur_moy']).mean():+.4f} % ; alpha0 {s['alpha0'].mean():+.6f}")
    Y_dur = -np.array([(res[l]["gamma"] * res[l]["dur_moy"]).mean() for l in lrs]); Y_a0 = -np.array([res[l]["alpha0"].mean() for l in lrs])
    for nom, v in (("-alpha0 (hors duree)", Y_a0), ("-gamma x duree", Y_dur)):
        ex = [(v[i + 1] - v[i]) / math.log(lrs[i + 1] / lrs[i]) for i in range(len(lrs) - 1)]
        Ls = [math.log(lrs[i + 1] / lrs[i]) for i in range(len(lrs) - 1)]
        def moyenne(lo, hi):
            sel = [i for i in range(len(lrs) - 1) if lrs[i] >= lo - 1e-9 and lrs[i + 1] <= hi + 1e-9]
            return sum(ex[i] * Ls[i] for i in sel) / sum(Ls[i] for i in sel)
        print(f"   contribution a l'exposant de 1/cible : {nom}: " + " ".join(f"{x:+.5f}" for x in ex) + f" ; 0,15->0,2 {moyenne(0.15, 0.2):+.5f} ; 0,2->0,3 {moyenne(0.2, 0.3):+.5f} ; marche {moyenne(0.15, 0.2) - moyenne(0.2, 0.3):+.5f}")
    print("post hoc : exposant local de 1/cible a duree de reference fixee (L = alpha0 + gamma x d_ref, lr <= 0,25) ; la reference 120 est dans la plage de duree moyenne (107 a 131), 100 et 140 en sont aux bords :")
    l7 = [l for l in lrs if l <= 0.25 + 1e-9]
    for dref in (100, 120, 140):
        L = np.array([-(res[l]["alpha0"] + res[l]["gamma"] * dref).mean() for l in l7])
        ex = [(L[i + 1] - L[i]) / math.log(l7[i + 1] / l7[i]) for i in range(len(l7) - 1)]
        print(f"   d_ref {dref}: " + " ".join(f"{x:+.5f}" for x in ex))
    ex_a = [(-(res[l7[i + 1]]["gamma"] * res[l7[i + 1]]["dur_moy"]).mean() + (res[l7[i]]["gamma"] * res[l7[i]]["dur_moy"]).mean()) / math.log(l7[i + 1] / l7[i]) for i in range(len(l7) - 1)]
    print("   exposant actuel de 1/cible (lr <= 0,25) : " + " ".join(f"{(math.log(np.exp(-np.array([res[l]['lncible'].mean() for l in l7]))[i + 1] / np.exp(-np.array([res[l]['lncible'].mean() for l in l7]))[i]) / math.log(l7[i + 1] / l7[i])):+.5f}" for i in range(len(l7) - 1)))
    series = {"1/rho = exp(-<ln rho>)": np.exp(-np.array([res[l]["lnrho"].mean() for l in lrs])), "1/cible = exp(-<ln cible>)": np.exp(-np.array([res[l]["lncible"].mean() for l in lrs])),
              "1/cible_w": np.exp(-np.array([res[l]["lncible_w"].mean() for l in lrs])), "1/rho_w": np.exp(-np.array([res[l]["lnrho_w"].mean() for l in lrs])), "omega": np.array([res[l]["omega"].mean() for l in lrs])}
    for nom, v in series.items():
        ex = exposants(nom, v)
        b, h = marche(ex)
        print(f"   {nom}: exposants locaux " + " ".join(f"{x:+.5f}" for x in ex) + f" ; 0,15->0,2 {b:+.5f} ; 0,2->0,3 {h:+.5f} ; marche {b - h:+.5f}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "injection":
        injection(float(sys.argv[2]))
    elif mode == "injection_synthese":
        injection_synthese()
    elif mode == "calme":
        calme()
    elif mode == "carte":
        carte()
