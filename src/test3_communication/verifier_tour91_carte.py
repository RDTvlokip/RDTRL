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


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "calme":
        calme()
    elif mode == "carte":
        carte()
