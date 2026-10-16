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
    """colonnes de rec (avancer_obs) : 0 rb, 1 z3 = e3 - o3, 2 u = l3 - l4, 3 sqrt(v_e3), 4 sqrt(v_l3), 5 m_e3, 6 m_l3."""
    red.LRV = float(lr)
    w = red.copie(S1)
    rec = np.zeros((n, 7))
    red.avancer_obs(w, float(PLI - mpf(xw)), n, rec, eps=q * lr, cross=2.0, stop=False)
    return rec


def effondre(lr, q, xw):
    return trajet(lr, q, xw)[-1, 0] > 0.9


def bissection(q, xw, lo=None, hi=None):
    """premier passage retour -> effondrement en balayant lr de 0,05 a 0,8 par pas de 0,01 (le plafond n'est pas monotone au-dela), puis bissection."""
    if lo is None:
        grille = np.arange(0.05, 0.80, 0.01)
        prec = effondre(grille[0], q, xw)
        if prec:
            return None
        for g in grille[1:]:
            if effondre(g, q, xw):
                lo, hi = g - 0.01, g
                break
            prec = False
        if hi is None:
            return None
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
        from verifier_tour61_decomposition_d3 import d3_noeud
        d3j = float(d3_noeud(PLI - mpf(xw), -1))   # jumeau instable (col) de la forme fermee
        d3n = float(d3_noeud(PLI - mpf(xw), +1))
        z3_col = float(np.log(26 * (1 - d3j) / d3j)); z3_noeud = float(np.log(26 * (1 - d3n) / d3n))
        print(f"  col : d3 = {d3j:.6e}, z3 = {z3_col:.4f} ; noeud : d3 = {d3n:.6e}, z3 = {z3_noeud:.4f}")
        deltas = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
        # le seuil exact est connu a 1e-11 ; lr* = milieu de l'intervalle
        tols = (0.02, 0.005)
        ht, bt = [], {tol: [] for tol in tols}
        print("  delta   au-dessus : installation(rb)  depart du col a tol 0,02 / 0,005   |  au-dessous : depart du col a tol 0,02 / 0,005   (z3 final)")
        for d in deltas:
            ra = trajet(ls + d, q, xw)
            sous = np.nonzero(ra[:, 0] <= 0.9)[0]
            ta = int(sous[-1]) + 1 if ra[-1, 0] > 0.9 and len(sous) else -1
            da = {tol: int(np.nonzero(np.abs(ra[:, 1] - z3_col) < tol)[0][-1]) + 1 if np.any(np.abs(ra[:, 1] - z3_col) < tol) else 0 for tol in tols}
            rb_ = trajet(ls - d, q, xw)
            db = {tol: int(np.nonzero(np.abs(rb_[:, 1] - z3_col) < tol)[0][-1]) + 1 if np.any(np.abs(rb_[:, 1] - z3_col) < tol) else 0 for tol in tols}
            ht.append(da[0.02])
            for tol in tols:
                bt[tol].append(db[tol])
            print(f"  {d:.0e}   {ta:6d}                {da[0.02]:6d} {da[0.005]:6d}                |   {db[0.02]:6d} {db[0.005]:6d}   ({rb_[-1, 1]:.4f}; effondre : {rb_[-1, 0] > 0.9})", flush=True)
        print(f"  pente au-dessus (depart du col, tol 0,02) : {pente(deltas, ht):+.1f} pas par decade de delta (negatif = temps croissant quand delta diminue)")
        for tol in tols:
            print(f"  pente au-dessous (depart du col), tol {tol}: {pente(deltas, bt[tol]):+.1f}")
    elif mode == "taux":
        # taux de depart du col predit : lr * lambda_s(x) * [1/(s+eps) + 1/(s+26 eps)], lambda_s = 7,30e-5 sqrt(x) (jacobien du tour 66),
        # s = sqrt(v_e3) mediane entre les pas 150 et 350 du trajet a lr* + 1e-7 ; compare au taux mesure par la pente (ln 10 / |pente|).
        ls = bissection(q, xw)
        r = trajet(ls + 1e-7, q, xw)
        s = float(np.median(r[150:350, 3]))
        eps = q * ls
        lam_s = 7.30e-5 * float(mpf(xw)) ** 0.5
        pred = ls * lam_s * (1.0 / (s + eps) + 1.0 / (s + 26 * eps))
        from verifier_tour61_decomposition_d3 import d3_noeud
        d3j = float(d3_noeud(PLI - mpf(xw), -1)); z3_col = float(np.log(26 * (1 - d3j) / d3j))
        deltas = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
        td = []
        for d in deltas:
            ra = trajet(ls + d, q, xw)
            td.append(int(np.nonzero(np.abs(ra[:, 1] - z3_col) < 0.005)[0][-1]) + 1)
        sl = pente(deltas, td)
        mes = np.log(10) / abs(sl)
        # jacobien a mi-sejour (t = moitie du temps de depart a delta = 1e-7)
        t_mid = td[0] // 2
        red.LRV = ls + 1e-7
        w = red.copie(S1)
        red.avancer(w, float(PLI - mpf(xw)), t_mid, eps=q * (ls + 1e-7), cross=2.0, stop=False)
        J = jacobien(np.concatenate([w["x"], w["m"], w["v"]]), w["t"], ls + 1e-7, q, xw)
        ev = np.linalg.eigvals(J)
        lead = float(np.max(np.abs(ev)))
        print(f"q = {q:.2e} xw = {xw} : lr* = {ls:.6f}  s(dwell) = {s:.3e}  eps = {eps:.2e}  lambda_s = {lam_s:.3e}")
        print(f"   taux predit {pred:.5f} /pas ; taux mesure (pente {sl:+.1f}, tol 0,005) {mes:.5f} /pas ; rapport predit/mesure {pred / mes:.3f} ; valeur propre dominante a t = {t_mid} : {lead:.5f} (ln = {np.log(lead):.5f})")
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
