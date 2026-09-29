"""Tour 60 (29/09/2026) : ma question a dipankarsarkar. L'exposant local p du
decalage converti s(delta) = (moyenne d3 - noeud ferme)/pente, a eps 1e-10,
atteint-il 1/2 (s ~ sqrt(delta_c - delta)) tout pres du pli ?

Instrument : la reduction a deux lignes (port numba, reduction_numba_tour59.py),
verifiee contre le reseau complet jusqu'a pli-1e-10 dans ce script meme (colonne
"reseau complet"). Protocole de hasard_reduction_tour59.py : etat de phase 1 du
reseau complet, 20 000 + k pas a pli-1e-6 (k aleatoire), puis 10 000 pas de
mise en regime au delta voulu, puis NREC pas enregistres (z3 = e3 - o3, donc
d3 = C e^-z3 / (1 + C e^-z3)). Les phases qui s'echappent (R_b > 0,9) sont
comptees et ecartees.

Noeud ferme et pente calcules en mpmath (40 chiffres) par Newton depuis le
developpement local au pli, car la grille de racines de
verifier_tour59_branches_fermees.py ne resout plus l'ecart noeud-jumeau a
moins de ~1e-11 du pli.
"""

import sys
import numpy as np
from mpmath import mp, mpf, exp, findroot, sqrt as msqrt

sys.path.insert(0, ".")
import reduction_numba_tour59 as redlib
from hasard_reduction_tour59 import base_state, warmed

mp.dps = 40
BETA_ = mpf("0.02")
C_ = 26
PLI = mpf("0.0134372100660973")
U_STAR = mpf("-1.47814609082")
G2, GD = mpf("-6.7902"), mpf("-101.109")


def G(u, d):
    r3 = 1 / (1 + exp(-u)); r4 = 1 / (1 + exp(u))
    s3 = 1 / (1 + C_ * exp(-(1 - d) * r3 / BETA_)); s4 = 1 / (1 + C_ * exp(-(1 + d) * r4 / BETA_))
    return -u - ((1 + d) * s4 - (1 - d) * s3) / BETA_


def d3_noeud(delta, signe=+1):
    """d3 du noeud stable (racine de plus grand u, signe=+1) ou du jumeau instable (signe=-1) au delta donne (mpf)."""
    delta = mpf(delta)
    # (u - u*)^2 = -2 G_d (delta - delta_c) / G'' = 29,8 (delta_c - delta) pour G'' = -6,79 et G_d = -101,1
    u0 = U_STAR + signe * msqrt(-2 * GD * (delta - PLI) / G2) if delta < PLI else U_STAR
    u = findroot(lambda x: G(x, delta), u0, tol=mpf(10) ** -30, maxsteps=200)
    r3 = 1 / (1 + exp(-u))
    return C_ * exp(-(1 - delta) * r3 / BETA_) / (1 + C_ * exp(-(1 - delta) * r3 / BETA_))


def pente(delta):
    h = abs(mpf(delta) - PLI) * mpf("1e-3")
    return float((d3_noeud(mpf(delta) + h) - d3_noeud(mpf(delta) - h)) / (2 * h))


def mesurer(off, ks, nrec, burn=10000, seed=1):
    s1 = base_state()
    rng = np.random.default_rng(seed)
    delta = float(PLI + mpf(off))
    dn = float(d3_noeud(PLI + mpf(off)))
    sl = pente(PLI + mpf(off))
    sn, sm, esc = [], [], 0
    for k in ks:
        w = warmed(s1, int(k), rng, 1e-9)
        rec0 = np.zeros((1, 7))
        redlib.avancer_obs(w, delta, burn, rec0)
        rec = np.zeros((nrec, 7))
        first, maxrb, rb, done = redlib.avancer_obs(w, delta, nrec, rec)
        if maxrb > 0.9:
            esc += 1
            continue
        ex = 26.0 * np.exp(-rec[:, 1])
        d3 = ex / (1.0 + ex)
        sn.append((d3.mean() - dn) / sl)
        sm.append((d3.mean() - np.median(d3)) / sl)
    return sl, np.array(sn), np.array(sm), esc


if __name__ == "__main__":
    ks = [int(a) for a in np.random.default_rng(7).integers(0, 4444, 16)]
    NREC = 2_000_000
    offs = ["-1e-5", "-1e-6", "-1e-7", "-1e-8", "-1e-9", "-3e-10", "-1e-10", "-3e-11", "-1e-11", "-3e-12", "-1e-12"]
    full = {"-1e-5": -4.977e-8, "-1e-6": -4.235e-8, "-1e-7": -2.779e-8, "-1e-8": -1.458e-8,
            "-1e-9": -6.376e-9, "-3e-10": -3.856e-9, "-1e-10": -2.350e-9}
    print(f"{len(ks)} phases, {NREC} pas enregistres chacune ; s = (moy d3 - noeud)/pente ; 'reseau complet' = E60-1/E60-3")
    print("offset     pente        s (reduction)            s (reseau complet)   ecart    s_med (moy-med)     phases echappees   p_local")
    prec = None
    for off in offs:
        sl, sn, sm, esc = mesurer(off, ks, NREC)
        if len(sn) == 0:
            print(f"{off:8s}  toutes les phases echappees")
            continue
        m = sn.mean(); e = sn.std(ddof=1) / np.sqrt(len(sn)) if len(sn) > 1 else 0.0
        ligne = f"{off:8s} {sl:10.1f}   {m:+.3e} +- {e:.1e}    "
        if off in full:
            ligne += f"{full[off]:+.3e}        {100 * (m / full[off] - 1):+5.1f} %"
        else:
            ligne += "-                        "
        ligne += f"   {sm.mean():+.3e}      {esc:2d}"
        # biais de d3 = s x pente (en unites de d3) et ecart noeud-jumeau en d3 : la conversion en unites de delta
        # ne vaut que si le biais est petit devant l'ecart
        gap = float(d3_noeud(PLI + mpf(off), -1) - d3_noeud(PLI + mpf(off), +1))
        biais = m * sl
        ligne += f"   biais_d3 = {biais:+.3e}  ecart noeud-jumeau = {gap:.3e}  |biais|/ecart = {abs(biais) / gap:7.2f}"
        if prec is not None:
            ligne += f"              {np.log(abs(prec[1]) / abs(m)) / np.log(abs(prec[0]) / abs(float(mpf(off)))):.3f}"
        prec = (abs(float(mpf(off))), m)
        print(ligne, flush=True)
