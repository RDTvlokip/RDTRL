"""Tour 70 (vraie critique de dipankarsarkar, 16/10/2026) : zero = a + b lr^2 + c lr^4, ou + d lr^3, sur mes zeros a x = 1e-8 ;
le confondant x ; predictions a lr 0,15 et 0,3.

Zeros (1e-7), x = 1e-8, reference = racine g_l3 = g_l4 au z4 du run pour les lr <= 0,05 (tour 69) :
0,005 : 6,2285 ; 0,01 : 6,2308 ; 0,02 : 6,2305 ; 0,05 : 6,2214 ; 0,1 : 6,1896 ; 0,2 : 6,0838.
x = 1e-7 : 0,05 : 6,2197 ; 0,1 : 6,1823 ; 0,2 : 6,0752.
Arguments optionnels lr=zero (x = 1e-8) ajoutes ou remplaces ; lr=zero@7 pour x = 1e-7.
"""

import sys

import numpy as np

Z8 = {0.005: 6.2285, 0.01: 6.2308, 0.02: 6.2305, 0.05: 6.2214, 0.1: 6.1896, 0.2: 6.0838}
Z7 = {0.05: 6.2197, 0.1: 6.1823, 0.2: 6.0752}


def fit(lrs, z, puissances):
    lrs = np.array(lrs, float); z = np.array(z, float)
    A = np.vstack([np.ones_like(lrs)] + [lrs ** p for p in puissances]).T
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    return coef, z - A @ coef


def predit(coef, puissances, lr):
    return coef[0] + sum(c * lr ** p for c, p in zip(coef[1:], puissances))


if __name__ == "__main__":
    for a in sys.argv[1:]:
        k, v = a.split("=")
        if v.endswith("@7"):
            Z7[float(k)] = float(v[:-2])
        else:
            Z8[float(k)] = float(v)
    print("x = 1e-8, zeros :", dict(sorted(Z8.items())))
    hi_max = max(Z8)
    plages = [("0,01-0,1", 0.01, 0.1), (f"0,01-{hi_max}", 0.01, hi_max), (f"0,005-{hi_max}", 0.005, hi_max)]
    for titre, puis in (("a + b lr^2 + c lr^4", (2, 4)), ("a + b lr^2 + d lr^3", (2, 3)), ("a + b lr^2 (p = 2 pur)", (2,))):
        print(f"\nzero = {titre}")
        for nom, lo, hi in plages:
            lrs = [l for l in sorted(Z8) if lo <= l <= hi]
            if len(lrs) <= len(puis):
                continue
            coef, res = fit(lrs, [Z8[l] for l in lrs], puis)
            cs = " ".join(f"{c:+.3f}" for c in coef[1:])
            print(f"   {nom:12s} a = {coef[0]:.4f}  coef ({cs})  residu max {np.abs(res).max():.4f}   lr 0,15 : {predit(coef, puis, 0.15):.4f}  lr 0,3 : {predit(coef, puis, 0.3):.4f}  lr 0,4 : {predit(coef, puis, 0.4):.4f}")
    print("\nconfondant x : trois points lr 0,05 ; 0,1 ; 0,2, a + b lr^2 + c lr^4 (exact) :")
    for nom, Z in (("x = 1e-7", Z7), ("x = 1e-8", Z8)):
        lrs = [0.05, 0.1, 0.2]
        coef, _ = fit(lrs, [Z[l] for l in lrs], (2, 4))
        print(f"   {nom} : a = {coef[0]:.4f} b = {coef[1]:+.3f} c = {coef[2]:+.2f}")
    lr7 = sorted(Z7)
    if len(lr7) >= 4:
        print(f"\nx = 1e-7, tous les points ({lr7}) :")
        for titre, puis in (("a + b lr^2 + c lr^4", (2, 4)), ("a + b lr^2 + d lr^3", (2, 3))):
            coef, res = fit(lr7, [Z7[l] for l in lr7], puis)
            print(f"   {titre} : a = {coef[0]:.4f} coef {np.round(coef[1:], 3)} residus {np.round(res, 4)} residu max {np.abs(res).max():.4f}")
    print(f"\ndecalage en x (1e-7 moins 1e-8) par lr : " + ", ".join(f"{l}: {Z7[l] - Z8[l]:+.4f}" for l in lr7 if l in Z8))
    lr_b = [l for l in sorted(Z8) if 0.01 <= l <= 0.1]
    cb, _ = fit(lr_b, [Z8[l] for l in lr_b], (2,))
    print(f"   decalage en x a lr 0,2 (1e-7 moins 1e-8) : {Z7[0.2] - Z8[0.2]:+.4f} ; exces de lr 0,2 sur le quadratique de lr 0,01-0,1 : {Z8[0.2] - predit(cb, (2,), 0.2):+.4f}")
