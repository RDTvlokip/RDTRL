"""Tour 61 (29/09/2026) : le logit e4 - o4 de la ligne 4 reste gele a d4 = 8,04e-13
(2,57e4 fois son noeud, 3,1e-17) parce que son gradient est sous le plancher d'Adam
(eps 1e-10). La fermeture idealisee (d4 = 26 e^-X4/(1+...)) est donc fausse a 1e-12
pres et le pli effectif de la dynamique n'est pas celui du systeme ideal.

Ici : fermeture G_eff(u, d) = -u - ((1+d)(1-d4c) - (1-d) s3(u,d))/beta avec d4c = 8,04e-13
FIXE ; pli effectif (G = 0 et dG/du = 0) en mpmath ; noeud, pente et delta_c effectifs ;
puis, dans la reduction numba (memes phases que verifier_tour60_exposant_reduction.py),
s(delta) = (moyenne d3 - noeud_eff)/pente_eff aux offsets mesures depuis le pli EFFECTIF,
et l'exposant local p entre offsets successifs.
"""

import sys
import numpy as np
from mpmath import mp, mpf, exp, findroot, sqrt as msqrt

sys.path.insert(0, ".")
import reduction_numba_tour59 as redlib
from hasard_reduction_tour59 import base_state, warmed
from verifier_tour60_exposant_reduction import PLI as PLI_IDEAL, U_STAR, G2, GD, C_, BETA_

mp.dps = 50
D4C = mpf("8.04e-13")


def G_eff(u, d):
    r3 = 1 / (1 + exp(-u))
    s3 = 1 / (1 + C_ * exp(-(1 - d) * r3 / BETA_))
    return -u - ((1 + d) * (1 - D4C) - (1 - d) * s3) / BETA_


def pli_effectif():
    sol = findroot(lambda u, d: [G_eff(u, d), mp.diff(lambda x: G_eff(x, d), u)], [U_STAR, PLI_IDEAL], tol=mpf(10) ** -40)
    return sol[0], sol[1]


U_EFF, PLI_EFF = pli_effectif()


def u_noeud(delta):
    delta = mpf(delta)
    u0 = U_EFF + msqrt(-2 * GD * (delta - PLI_EFF) / G2)
    return findroot(lambda x: G_eff(x, delta), u0, tol=mpf(10) ** -40, maxsteps=300)


def d3_noeud(delta):
    u = u_noeud(delta)
    r3 = 1 / (1 + exp(-u))
    e = C_ * exp(-(1 - mpf(delta)) * r3 / BETA_)
    return e / (1 + e)


def pente(delta):
    h = abs(mpf(delta) - PLI_EFF) * mpf("1e-3")
    return float((d3_noeud(mpf(delta) + h) - d3_noeud(mpf(delta) - h)) / (2 * h))


if __name__ == "__main__":
    print(f"pli ideal   = {mp.nstr(PLI_IDEAL, 22)}")
    print(f"pli effectif (d4 = {float(D4C):.3e}) = {mp.nstr(PLI_EFF, 22)}   decalage = {float(PLI_EFF - PLI_IDEAL):+.3e}")
    ks = [int(a) for a in np.random.default_rng(7).integers(0, 4444, 16)]
    offs = ["-1e-8", "-1e-9", "-3e-10", "-1e-10", "-3e-11", "-1e-11", "-3e-12", "-1e-12"]
    NREC = 2_000_000
    s1 = base_state()
    rng = np.random.default_rng(1)
    print("\noffsets depuis le pli EFFECTIF ; s = (moy d3 - noeud_eff)/pente_eff ; biais_d3 = s x pente")
    print("offset     pente_eff     s (eff.)         biais_d3        p_local (eff.)   [s ideal, p ideal du tour 60]")
    ideal = {"-1e-8": (-1.459e-8, 0.282), "-1e-9": (-6.410e-9, 0.357), "-3e-10": (-3.866e-9, 0.420), "-1e-10": (-2.355e-9, 0.451),
             "-3e-11": (-1.334e-9, 0.472), "-1e-11": (-7.844e-10, 0.484), "-3e-12": (-4.343e-10, 0.491), "-1e-12": (-2.523e-10, 0.494)}
    prec = None
    for off in offs:
        delta_mp = PLI_EFF + mpf(off)
        delta = float(delta_mp)
        dn = float(d3_noeud(delta_mp)); sl = pente(delta_mp)
        sn, esc = [], 0
        for k in ks:
            w = warmed(s1, k, rng, 1e-9)
            redlib.avancer_obs(w, delta, 10000, np.zeros((1, 7)))
            rec = np.zeros((NREC, 7))
            first, maxrb, rb, done = redlib.avancer_obs(w, delta, NREC, rec)
            if maxrb > 0.9:
                esc += 1
                continue
            ex = 26.0 * np.exp(-rec[:, 1]); d3 = ex / (1.0 + ex)
            sn.append((d3.mean() - dn) / sl)
        m = float(np.mean(sn))
        p = "" if prec is None else f"{np.log(abs(prec[1]) / abs(m)) / np.log(abs(prec[0]) / abs(float(mpf(off)))):.3f}"
        prec = (abs(float(mpf(off))), m)
        si, pi = ideal[off]
        print(f"{off:8s}  {sl:10.1f}   {m:+.3e}     {m * sl:+.3e}     {p:>6s}           [{si:+.3e}  {pi:.3f}]   echappees {esc}", flush=True)
