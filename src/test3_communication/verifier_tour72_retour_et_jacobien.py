"""Tour 72 (vraie critique de dipankarsarkar, 18/10/2026) : le col du transitoire de demarrage.

Modes :
  retour  <q> <xw>             : lr* par bissection (chauffe de 20 000 pas a delta = pli - xw, eps/lr = q, etat de base brut, un trajet), puis,
                                 pour des decalages delta = |lr - lr*| de 1e-7 a 1e-2 de part et d'autre : temps d'installation de l'effondrement
                                 au-dessus (dernier pas avec rb <= 0,9, plus un) et, au-dessous, temps de retour au noeud a la tolerance tol
                                 (premier pas apres lequel |rb - rb_final| < tol jusqu'a la fin), pour tol = 1e-1, 1e-2, 1e-3 ; pentes
                                 en pas par decade (regression de t sur log10 delta).
  jac     <q> <xw> <delta_lr> <t1> [<t2> ...] : trajet a lr = lr* + delta_lr, puis, aux pas t1, t2, ... , jacobien (differences finies centrees) de
                                 l'application d'un pas sur l'etat complet (x, m, v) (18 dimensions) et ses valeurs propres de plus grand module.
rb = (1 - d4) r4 ; noeud : rb -> 0,81346.
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


def trajet(lr, q, xw, n=NPAS):
    red.LRV = float(lr)
    w = red.copie(S1)
    rec = np.zeros((n, 7))
    red.avancer(w, float(PLI - mpf(xw)), n, eps=q * lr, cross=2.0, stop=False, rec=rec)
    return rec[:, 0].copy()


def effondre(lr, q, xw):
    return trajet(lr, q, xw)[-1] > 0.9


def bissection(q, xw, lo=0.2, hi=0.45):
    c_lo, c_hi = effondre(lo, q, xw), effondre(hi, q, xw)
    if c_lo == c_hi:
        return None
    while hi - lo > 1e-11:
        mid = 0.5 * (lo + hi)
        if effondre(mid, q, xw) == c_hi:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def temps_retour(r, tol):
    """premier pas t tel que |r[s] - r[-1]| < tol pour tout s >= t."""
    hors = np.nonzero(np.abs(r - r[-1]) >= tol)[0]
    return int(hors[-1]) + 1 if len(hors) else 0


def pente(deltas, temps):
    ok = [(d, t) for d, t in zip(deltas, temps) if t > 0]
    if len(ok) < 4:
        return float("nan")
    return float(np.polyfit(np.log10([a for a, _ in ok]), [b for _, b in ok], 1)[0])


def pas(state, lr, q, xw):
    s = dict(x=state[0:6].copy(), m=state[6:12].copy(), v=state[12:18].copy(), Se=S1["Se"], Sed=S1["Sed"], t=state[18])
    red.LRV = float(lr)
    red.avancer_obs(s, float(PLI - mpf(xw)), 1, np.zeros((1, 7)), eps=q * lr, cross=2.0, stop=False)
    return np.concatenate([s["x"], s["m"], s["v"]])


def jacobien(state18, t, lr, q, xw):
    st = np.concatenate([state18, [t]])
    J = np.zeros((18, 18))
    for j in range(18):
        h = 1e-6 * max(abs(st[j]), 1e-3 if j < 6 else 1e-12)
        if j >= 12:
            h = 1e-6 * abs(st[j])
        sp, sm = st.copy(), st.copy()
        sp[j] += h; sm[j] -= h
        J[:, j] = (pas(sp, lr, q, xw) - pas(sm, lr, q, xw)) / (2 * h)
    return J


if __name__ == "__main__":
    mode = sys.argv[1]
    q, xw = float(sys.argv[2]), sys.argv[3]
    if mode == "retour":
        ls = bissection(q, xw)
        print(f"q = {q:.3e}, xw = {xw} : lr* = {ls}")
        if ls is None:
            sys.exit(0)
        deltas = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
        # le seuil exact est connu a 1e-11 ; on prend lr* = borne haute (effondre) pour etre du bon cote
        ht, bt = [], {1e-1: [], 1e-2: [], 1e-3: []}
        print("  delta     au-dessus : installation   au-dessous : retour a tol 1e-1, 1e-2, 1e-3")
        for d in deltas:
            ra = trajet(ls + d, q, xw)
            sous = np.nonzero(ra <= 0.9)[0]
            ta = int(sous[-1]) + 1 if ra[-1] > 0.9 and len(sous) else -1
            rb_ = trajet(ls - d, q, xw)
            tb = {tol: temps_retour(rb_, tol) for tol in (1e-1, 1e-2, 1e-3)}
            ht.append(ta)
            for tol in tb:
                bt[tol].append(tb[tol])
            print(f"  {d:.0e}   {ta:6d}               {tb[1e-1]:6d} {tb[1e-2]:6d} {tb[1e-3]:6d}   (collapse au-dessous : {rb_[-1] > 0.9})", flush=True)
        print(f"  pente au-dessus : {pente(deltas, ht):+.1f} pas par decade de delta (negatif = temps croissant quand delta diminue)")
        for tol in bt:
            print(f"  pente au-dessous, tol {tol:.0e} : {pente(deltas, bt[tol]):+.1f}")
    elif mode == "jac":
        dl = float(sys.argv[4]); ts = [int(a) for a in sys.argv[5:]]
        ls = bissection(q, xw)
        lr = ls + dl
        print(f"q = {q:.3e}, xw = {xw} : lr* = {ls:.10f}, lr = {lr:.10f}")
        red.LRV = lr
        w = red.copie(S1)
        t_prec = 0
        for t in sorted(ts):
            red.avancer(w, float(PLI - mpf(xw)), t - t_prec, eps=q * lr, cross=2.0, stop=False)
            t_prec = t
            st = np.concatenate([w["x"], w["m"], w["v"]])
            J = jacobien(st, w["t"], lr, q, xw)
            ev = np.linalg.eigvals(J)
            ordre = np.argsort(-np.abs(ev))
            top = ev[ordre][:4]
            print(f"  t = {t:5d} : rb ~ ? ; valeurs propres de plus grand module : " + ", ".join(f"{abs(z):.5f}{'+' if z.imag >= 0 else '-'}{abs(z.imag):.3f}i" if abs(z.imag) > 1e-9 else f"{z.real:.5f}" for z in top), flush=True)
