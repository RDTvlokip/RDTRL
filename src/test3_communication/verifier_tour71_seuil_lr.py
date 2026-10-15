"""Tour 71 : le plafond en lr (chauffe de 20 000 pas a pli - 1e-6 depuis l'etat de base JSON, eps/lr = q fixe) : seuil lr* par bissection,
temps d'effondrement durable au-dessus, temps de retour au-dessous, exposant.

Usage :  python verifier_tour71_seuil_lr.py bissection <lo> <hi> <tol> [q] [echelle]
         python verifier_tour71_seuil_lr.py balayage <lr*> [q] [echelle]     (decalages 1e-7 ... 1e-2 de part et d'autre)
         python verifier_tour71_seuil_lr.py liste <lr> [<lr> ...] [-- q echelle]
echelle = 1 : m et v de l'etat de base multiplies par lr/0,05 et (lr/0,05)^2 (sqrt(v) et m vont comme lr), sinon 0.
Etat de base brut (sans perturbation) : une seule trajectoire par lr, deterministe. rb = (1 - d4) r4. Effondrement durable : rb final > 0,9 ;
temps d'effondrement = dernier pas avec rb <= 0,9, plus un ; temps de retour (au-dessous du seuil) = dernier pas avec rb > 0,9, plus un.
"""

import json
import sys

import numpy as np
from mpmath import mp, mpf

sys.path.insert(0, ".")
import reduction_numba_tour71_lrvar as red
from verifier_tour61_decomposition_d3 import PLI

mp.dps = 40
NPAS = 20000


def etat_base():
    d = json.load(open("etat_base_reduction_mur23.json", encoding="utf-8"))
    return dict(x=np.array(d["x"]), m=np.array(d["m"]), v=np.array(d["v"]), Se=d["Se"], Sed=d["Sed"], t=d["t"])


S1 = etat_base()
DELTA = float(PLI - mpf("1e-6"))


def issue(lr, q, echelle, n=NPAS):
    red.LRV = float(lr)
    w = red.copie(S1)
    if echelle:
        c = lr / 0.05
        w["m"] = w["m"] * c; w["v"] = w["v"] * c * c
    rec = np.zeros((n, 7))
    red.avancer(w, DELTA, n, eps=q * lr, cross=2.0, stop=False, rec=rec)
    r = rec[:, 0]
    sup = np.nonzero(r > 0.9)[0]
    sous = np.nonzero(r <= 0.9)[0]
    if r[-1] > 0.9:
        return True, (int(sous[-1]) + 1 if len(sous) else 0)
    return False, (int(sup[-1]) + 1 if len(sup) else 0)


if __name__ == "__main__":
    mode = sys.argv[1]
    args = sys.argv[2:]
    if mode == "bissection":
        lo, hi, tol = float(args[0]), float(args[1]), float(args[2])
        q = float(args[3]) if len(args) > 3 else 5.95e-7
        ech = int(args[4]) if len(args) > 4 else 0
        c_lo, _ = issue(lo, q, ech); c_hi, _ = issue(hi, q, ech)
        print(f"lo {lo} : {'effondre' if c_lo else 'revient'} ; hi {hi} : {'effondre' if c_hi else 'revient'} (q = {q:.3e}, echelle {ech})")
        if c_lo == c_hi:
            print("pas de seuil dans l'intervalle")
            sys.exit(0)
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            c, t = issue(mid, q, ech)
            if c == c_hi:
                hi = mid
            else:
                lo = mid
        print(f"lr* entre {lo:.9f} et {hi:.9f}")
    elif mode == "balayage":
        ls = float(args[0]); q = float(args[1]) if len(args) > 1 else 5.95e-7; ech = int(args[2]) if len(args) > 2 else 0
        print(f"lr* = {ls} ; q = {q:.3e} ; echelle {ech}")
        print("  lr - lr*      issue       temps (pas)")
        pts_h, pts_b = [], []
        for e in (1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
            for sg in (+1, -1):
                lr = ls + sg * e
                c, t = issue(lr, q, ech)
                print(f"  {sg * e:+.1e}  {'effondre' if c else 'revient ':9s}  {t}", flush=True)
                (pts_h if c else pts_b).append((e, t))
        for nom, pts in (("effondrement (au-dessus)", pts_h), ("retour (au-dessous)", pts_b)):
            pts = [(e, t) for e, t in pts if t > 0]
            if len(pts) >= 4:
                le = np.log([p[0] for p in pts]); lt = np.log([p[1] for p in pts])
                pente = np.polyfit(le, lt, 1)[0]
                print(f"  {nom} : pente log t / log |lr - lr*| = {pente:+.3f}  ({len(pts)} points) ; t entre {min(p[1] for p in pts)} et {max(p[1] for p in pts)}")
    else:
        sep = args.index("--") if "--" in args else len(args)
        q = float(args[sep + 1]) if sep + 1 < len(args) else 5.95e-7
        ech = int(args[sep + 2]) if sep + 2 < len(args) else 0
        for a in args[:sep]:
            c, t = issue(float(a), q, ech)
            print(f"lr {float(a):.6f} : {'effondrement durable, installe au pas' if c else 'revient sur le noeud, dernier rb > 0,9 au pas'} {t}", flush=True)
