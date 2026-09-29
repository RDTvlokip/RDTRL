"""Tour 61 (29/09/2026, vraie critique de dipankarsarkar) : le decalage de d3
se decompose en (mean - median), le "biais de salves", et (median - noeud),
l'etat calme qui quitte le noeud. Verification et prolongement.

Partie A, reseau complet (traces E60-1/E60-3, eps 1e-10, 6 phases, 30 000
derniers pas de 40 000) : mean - median, median - noeud, mean - noeud en unites
de d3, sd(d3), et le rapport (mean - median)/sd -- sa table et son "0,63 sd".
Partie B, reduction numba (16 phases, 2 000 000 pas enregistres), jusqu'a
pli-1e-12 : les memes, plus la meme decomposition pour r4, et le delta'
equivalent de la MEDIANE lue dans d3 et dans r4 separement :
    delta'_x = (mediane_x - noeud_x(delta)) / (d noeud_x / d delta).
Si l'etat calme etait le noeud d'un delta decale, delta'_d3 = delta'_r4.
Partie C : ajustement B - c x^q du biais moyen de d3 sur trois lignes
(1e-10, 1e-11, 1e-12), prediction des autres lignes.
"""

import glob
import re
import sys

import numpy as np
from mpmath import mp, mpf, exp

sys.path.insert(0, ".")
from verifier_tour60_exposant_reduction import PLI, d3_noeud, findroot, G, U_STAR, msqrt, G2, GD  # noqa: F401
import reduction_numba_tour59 as redlib
from hasard_reduction_tour59 import base_state, warmed

mp.dps = 40


def u_noeud(delta):
    delta = mpf(delta)
    u0 = U_STAR + msqrt(-2 * GD * (delta - PLI) / G2) if delta < PLI else U_STAR
    return findroot(lambda x: G(x, delta), u0, tol=mpf(10) ** -30, maxsteps=200)


def r4_noeud(delta):
    u = u_noeud(delta)
    return 1 / (1 + exp(u))


def pente_fn(fn, delta):
    h = abs(mpf(delta) - PLI) * mpf("1e-3")
    return float((fn(mpf(delta) + h) - fn(mpf(delta) - h)) / (2 * h))


def partie_A():
    print("=== A. reseau complet (6 phases), en unites de d3 ; sa table entre parentheses")
    print("offset       mean-median          median-noeud         mean-noeud          sd(d3)      (mean-median)/sd")
    sa = {"-1e-05": (-0.80e-6, 0, -0.80e-6), "-1e-06": (-2.28e-6, 0, -2.28e-6), "-1e-07": (-4.82e-6, 0, -4.82e-6),
          "-1e-08": (-7.59e-6, -0.45e-6, -8.04e-6), "-1e-09": (-7.98e-6, -3.17e-6, -11.14e-6),
          "-3e-10": (-7.99e-6, -4.32e-6, -12.31e-6), "-1e-10": (-7.99e-6, -5.00e-6, -12.99e-6)}
    groupes = {}
    for f in glob.glob("D:/tmp/rdtrl_tour59_trace_mur23_a3_b4_delta*_pas40000_eps1e-10_chauffe20000_*.txt"):
        m = re.search(r"delta([0-9.e-]+)_pas40000_eps1e-10_chauffe20000_(\d+)\.txt$", f)
        if not m or int(m.group(2)) == 20000:
            continue
        off = float(mpf(m.group(1)) - PLI)
        groupes.setdefault(round(off, 12), []).append(f)
    for off in sorted(groupes, key=lambda o: o):
        pass
    for off in sorted(groupes, key=lambda o: -abs(o)):
        dn = float(d3_noeud(PLI + mpf(off)))
        mm, mn, mnn, sds = [], [], [], []
        for f in groupes[off]:
            x = np.loadtxt(f)[-30000:, 1]
            mm.append(x.mean() - np.median(x)); mn.append(np.median(x) - dn); mnn.append(x.mean() - dn); sds.append(x.std())
        lab = f"{off:.0e}".replace("e-0", "e-") if abs(off) >= 1e-9 else f"{off:.0e}"
        cle = min(sa, key=lambda k: abs(float(k) - off)) if any(abs(float(k) - off) < 0.3 * abs(off) for k in sa) else None
        s = f"{off:+.0e}   {np.mean(mm):+.3e}   {np.mean(mn):+.3e}   {np.mean(mnn):+.3e}   {np.mean(sds):.3e}   {np.mean(mm) / np.mean(sds):+.3f}"
        if cle:
            s += f"    (sa table : {sa[cle][0]:+.2e} {sa[cle][1]:+.2e} {sa[cle][2]:+.2e})"
        print(s)


def d4_noeud(delta):
    """d4 = 1 - s4 au noeud : X4 = (1+delta) r4/beta, d4 = 26 e^-X4 / (1 + 26 e^-X4)."""
    delta = mpf(delta)
    x4 = (1 + delta) * r4_noeud(delta) / BETA_
    return C_ * exp(-x4) / (1 + C_ * exp(-x4))


def partie_B(nrec=2_000_000, offs=None):
    ks = [int(a) for a in np.random.default_rng(7).integers(0, 4444, 16)]
    offs = offs or ["-1e-5", "-1e-6", "-1e-7", "-1e-8", "-1e-9", "-3e-10", "-1e-10", "-3e-11", "-1e-11", "-3e-12", "-1e-12"]
    s1 = base_state()
    rng = np.random.default_rng(1)
    print("\n=== B. reduction (16 phases x %d pas), d3 puis r4" % nrec)
    print("offset     mean-median_d3  median-noeud_d3  mean-noeud_d3   sd_d3      (mm)/sd   | delta'_med(d3)   delta'_med(r4)   rapport r4/d3 | delta'_moy(d3)  delta'_moy(r4)")
    lignes = []
    for off in offs:
        delta = float(PLI + mpf(off))
        dn = float(d3_noeud(PLI + mpf(off))); rn = float(r4_noeud(PLI + mpf(off)))
        sl3 = pente_fn(d3_noeud, PLI + mpf(off)); sl4 = pente_fn(r4_noeud, PLI + mpf(off))
        dn4 = float(d4_noeud(PLI + mpf(off))); sl_d4 = pente_fn(d4_noeud, PLI + mpf(off))
        acc = []
        for k in ks:
            w = warmed(s1, k, rng, 1e-9)
            redlib.avancer_obs(w, delta, 10000, np.zeros((1, 7)))
            rec = np.zeros((nrec, 7))
            first, maxrb, rb, done = redlib.avancer_obs(w, delta, nrec, rec)
            if maxrb > 0.9:
                continue
            ex = 26.0 * np.exp(-rec[:, 1]); d3 = ex / (1.0 + ex)
            r4 = 1.0 / (1.0 + np.exp(rec[:, 2]) + w["Se"])
            d4 = 1.0 - rec[:, 0] / r4  # rb = (1 - d4) r4 ; d4 ~ 3e-12, erreur relative ~ 3e-5
            acc.append((d3.mean() - np.median(d3), np.median(d3) - dn, d3.mean() - dn, d3.std(),
                        (np.median(d3) - dn) / sl3, (np.median(r4) - rn) / sl4,
                        (d3.mean() - dn) / sl3, (r4.mean() - rn) / sl4, d3.mean(),
                        (np.median(d4) - dn4) / sl_d4))
        a = np.array(acc).mean(axis=0)
        lignes.append((off, a, sl3))
        rap = a[5] / a[4] if a[4] != 0 else float("nan")
        rap4 = a[9] / a[4] if a[4] != 0 else float("nan")
        print(f"{off:8s}   {a[0]:+.3e}      {a[1]:+.3e}      {a[2]:+.3e}    {a[3]:.3e}   {a[0] / a[3]:+.3f}   |  {a[4]:+.3e}      {a[5]:+.3e}      {rap:+7.3f}     | {a[6]:+.3e}    {a[7]:+.3e}"
              f"   | delta'_med(d4) {a[9]:+.3e}  rapport d4/d3 {rap4:+7.3f}", flush=True)
    return lignes


def partie_C(lignes):
    print("\n=== C. biais moyen de d3 (mean - noeud) : ajustement B - c x^q sur 1e-10, 1e-11, 1e-12, puis prediction des autres lignes")
    x = np.array([abs(float(mpf(o))) for o, _, _ in lignes])
    b = np.array([-a[2] for _, a, _ in lignes])  # biais positif = |mean - noeud|
    idx = [list(map(lambda t: t[0], lignes)).index(o) for o in ("-1e-10", "-1e-11", "-1e-12")]
    xs, bs = x[idx], b[idx]
    from scipy.optimize import fsolve
    def f(p):
        B, lc, q = p
        return [B - np.exp(lc) * xs[i] ** q - bs[i] for i in range(3)]
    B, lc, q = fsolve(f, [1.4e-5, np.log(1e-5 / 1e-5 ** 0.5), 0.47], xtol=1e-14)
    print(f"   B = {B:.4e}   c = {np.exp(lc):.4e}   q = {q:.4f}   (lui : B = 1,41e-5, q = 0,47)")
    for (o, a, _), xi, bi in zip(lignes, x, b):
        pred = B - np.exp(lc) * xi ** q
        print(f"   {o:8s}  mesure {bi:.4e}   ajuste {pred:.4e}   ecart {100 * (pred / bi - 1):+6.2f} %")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "d4":
        # E61-1 seul : la troisieme coordonnee (d4) aux offsets proches du pli
        partie_B(offs=["-1e-8", "-1e-9", "-1e-10", "-1e-11", "-1e-12"])
    else:
        partie_A()
        lignes = partie_B()
        partie_C(lignes)
