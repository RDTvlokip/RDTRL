"""Tour 74 (vraie critique de dipankarsarkar, 19/10/2026) : d'ou vient le facteur 0,84 entre gain x lambda_s et le jacobien ?

Il propose deux separations : (1) la valeur propre molle de la forme fermee a x = 1e-6 divisee par 7,30e-8 (= 7,30e-5 sqrt(x)) : proche de 0,84,
c'est la courbure ; proche de 1,00, c'est le gain ; (2) deplacer x : une correction d'ordre suivant vaut 1 - a sqrt(x), 0,84 a 1e-6 donne
0,95 a 1e-7 et 0,98 a 1e-8.

Modes :
  fermee            : jacobien 3x3 de (g_e3, g_e4, g_l3) en (z3, z4, u) au noeud et au col de la forme fermee (G(u, delta) = 0, Se = 0 pour la racine,
                      Se et Sed de l'etat de base dans le gradient), pour x = pli - delta de 1e-9 a 1e-4 ; valeur propre molle / (7,30e-5 sqrt(x)).
  precond <q> <xw> <lr_delta> <t1> [<t2>...] : jacobien 6x6 du gradient en coordonnees brutes (e3, o3, e4, o4, l3, l4), evalue sur l'etat du
                      trajet a lr* + lr_delta (instants t1, ...), preconditionne par 1/(sqrt(v_i) + eps) mesures sur ce meme etat (m lent) :
                      A = I + lr P J6. Valeur propre dominante de A, a comparer au jacobien complet a 18 dimensions du tour 73
                      (1,01692 a t = 200, 1,0177 a t = 350, 1,0199 a t = 700 pour x = 1e-6).
"""

import json
import sys

import numpy as np
from mpmath import mp, mpf, exp, log, matrix, eig, findroot, sqrt as msqrt

sys.path.insert(0, ".")
from verifier_tour60_exposant_reduction import PLI, G, U_STAR, G2, GD

mp.dps = 50
B, C, N = mpf("0.02"), mpf(26), mpf(27)
_b = json.load(open("etat_base_reduction_mur23.json", encoding="utf-8"))
SE, SED = mpf(_b["Se"]), mpf(_b["Sed"])


def grad6(x, delta, Se=SE, Sed=SED):
    """gradient (ascension) des six coordonnees brutes, comme dans run() de reduction_numba_tour71_lrvar.py."""
    e3, o3, e4, o4, l3, l4 = x
    z3, z4, u = e3 - o3, e4 - o4, l3 - l4
    ex3 = C * exp(-z3); s3 = 1 / (1 + ex3); d3 = ex3 * s3
    ex4 = C * exp(-z4); d4 = ex4 / (1 + ex4); s4 = 1 - d4
    w3 = exp(u); Z = w3 + 1 + Se; r3 = w3 / Z; r4 = 1 / Z; logZ = log(Z)
    Hr = -(r3 * (u - logZ) + r4 * (-logZ) + (Sed - logZ * Se) / Z)
    a3 = (1 - delta) * s3; a4 = (1 + delta) * s4; abar = r3 * a3 + r4 * a4
    g_l3 = r3 * ((a3 - abar) - B * ((u - logZ) + Hr)) / N
    g_l4 = r4 * ((a4 - abar) - B * ((-logZ) + Hr)) / N
    X3 = (1 - delta) * r3 / B; X4 = (1 + delta) * r4 / B
    ge3 = s3 * d3 * B * (X3 - z3) / N
    ge4 = s4 * d4 * B * (X4 - z4) / N
    return [ge3, -ge3 / C, ge4, -ge4 / C, g_l3, g_l4]


def jac6(x, delta, h=mpf("1e-18")):
    J = matrix(6, 6)
    for j in range(6):
        xp = list(x); xm = list(x); xp[j] += h; xm[j] -= h
        gp = grad6(xp, delta); gm = grad6(xm, delta)
        for i in range(6):
            J[i, j] = (gp[i] - gm[i]) / (2 * h)
    return J


def point_ferme(delta, signe):
    """(z3, z4, u) du noeud (signe +1) ou du col (-1) de la forme fermee (racine de G a Se = 0), placee dans les coordonnees brutes e3 = z3, o3 = 0, etc."""
    delta = mpf(delta)
    u0 = U_STAR + signe * msqrt(-2 * GD * (delta - PLI) / G2)
    u = findroot(lambda t: G(t, delta), u0, tol=mpf(10) ** -35, maxsteps=200)
    r3 = 1 / (1 + exp(-u)); r4 = 1 / (1 + exp(u))
    z3 = (1 - delta) * r3 / B; z4 = (1 + delta) * r4 / B
    return z3, z4, u


def valeurs_propres_3x3(delta, signe):
    z3, z4, u = point_ferme(delta, signe)
    x = [z3, mpf(0), z4, mpf(0), u, mpf(0)]
    J6 = jac6(x, delta)
    # reduction a (z3, z4, u) : les gradients g_e3, g_e4, g_l3 dependent de (z3, z4, u) seulement ; derivee par rapport a e3 (o3 = 0 fixe) = derivee en z3
    J = matrix(3, 3)
    idx_col = (0, 2, 4)
    idx_row = (0, 2, 4)
    for a, i in enumerate(idx_row):
        for b, j in enumerate(idx_col):
            J[a, b] = J6[i, j]
    ev, vec = eig(J)
    ordre = sorted(range(3), key=lambda k: abs(ev[k]))
    # z4 est mort (d4 ~ 1e-12 : la ligne g_e4 est nulle a 1e-20 pres) : la premiere valeur propre est nulle, la molle est la deuxieme
    return [complex(ev[k]) for k in ordre[1:]], [[complex(vec[i, k]) for i in range(3)] for k in ordre[1:]], (z3, z4, u)


def mode_fermee():
    print("x = pli - delta ; col et noeud ; valeur propre molle lambda_s, lambda_s / sqrt(x), lambda_s / (7,30e-5 sqrt(x)), z3")
    res = {}
    for e in (-9, -8, -7, -6.5, -6, -5.5, -5, -4.5, -4):
        x = mpf(10) ** mpf(e)
        for signe, nom in ((+1, "noeud"), (-1, "col  ")):
            ev, vec, (z3, z4, u) = valeurs_propres_3x3(PLI - x, signe)
            ls = ev[0].real
            res[(e, signe)] = ls
            print(f"   x = 1e{e:+5.1f} {nom} : lambda_s = {ls:+.5e}, /sqrt(x) = {ls / float(x) ** 0.5:+.5e}, / (7,30e-5 sqrt x) = {ls / (7.30e-5 * float(x) ** 0.5):+.4f}   z3 = {float(z3):.5f}  (raide : {ev[1].real:+.3e})")
    print("\nrapport col/noeud a chaque x :")
    for e in (-9, -8, -7, -6.5, -6, -5.5, -5, -4.5, -4):
        print(f"   1e{e:+5.1f} : {res[(e, -1)] / res[(e, 1)]:+.4f}")


def analyser(q, xw, lr_delta, ts, ablation=False, j18=True):
    """sur le trajet a lr* + lr_delta : jacobien 18-dim (reference), A = I + lr P J6 (momentum lent), modele du tour 73, modele avec la valeur propre du col fermee."""
    import importlib
    import os
    import verifier_tour72_retour_et_jacobien as v72
    red = importlib.import_module(os.environ.get("TOUR74_MODULE", "reduction_numba_tour71_lrvar"))
    v72.red = red   # bissection, trajet, pas et jacobien de verifier_tour72 utilisent le module choisi
    b1 = red.B1
    print(f"   module {red.__name__}, b1 = {b1}")
    ls = v72.bissection(q, xw)
    if ls is None:
        print(f"q = {q:.3e} xw = {xw} : pas de lr* dans [0,05 ; 0,8]")
        return None
    lr = ls + lr_delta
    print(f"q = {q:.3e} xw = {xw} : lr* = {ls:.12f}, lr = {lr:.12f}", flush=True)
    red.LRV = lr
    w = red.copie(v72.S1)
    delta = PLI - mpf(xw)
    eps = q * lr
    lam_col = valeurs_propres_3x3(delta, -1)[0][0].real
    lam73 = 7.30e-5 * float(mpf(xw)) ** 0.5
    print(f"   lambda_s col (forme fermee) = {lam_col:.4e}, 7,30e-5 sqrt(x) = {lam73:.4e}, rapport {lam_col / lam73:.4f}")
    t_prec = 0
    out = []
    for t in sorted(ts):
        red.avancer(w, float(delta), t - t_prec, eps=eps, cross=2.0, stop=False)
        t_prec = t
        x = [mpf(float(a)) for a in w["x"]]
        s = [float(np.sqrt(a)) for a in w["v"]]
        J6 = jac6(x, delta)
        J = np.array([[float(J6[i, j]) for j in range(6)] for i in range(6)])
        gains = np.array([1.0 / (si + eps) for si in s])
        def lead_A(k=1.0):
            g = gains.copy(); g[4] *= k; g[5] *= k
            A = np.eye(6) + lr * np.diag(g) @ J
            ev = np.linalg.eigvals(A)
            return ev[np.argmax(ev.real)].real
        la = lead_A()
        # application lineaire a 12 dimensions (x, m), v fige : m' = b1 m + (1 - b1) J x ; x' = x + lr P m'
        Pm = np.diag(gains)
        M12 = np.block([[np.eye(6) + lr * (1 - b1) * Pm @ J, lr * b1 * Pm], [(1 - b1) * J, b1 * np.eye(6)]])
        ev12 = np.linalg.eigvals(M12)
        l12 = ev12[np.argmax(ev12.real)].real
        # equation caracteristique du mode scalaire : eps^2 + eps (1 - b1)(1 - cJ) - cJ (1 - b1) = 0, cJ = A - 1
        cJ = la - 1.0
        eps_c = 0.5 * (-(1 - b1) * (1 - cJ) + np.sqrt(((1 - b1) * (1 - cJ)) ** 2 + 4 * cJ * (1 - b1)))
        mod73 = lr * lam73 * (1.0 / (s[0] + eps) + 1.0 / (s[0] + 26 * eps))
        modcol = mod73 * lam_col / lam73
        l18 = None
        if j18:
            st = np.concatenate([w["x"], w["m"], w["v"]])
            ev = np.linalg.eigvals(v72.jacobien(st, w["t"], lr, q, xw))
            l18 = float(np.max(np.abs(ev)))
        print(f"  t = {t:5d} : z3 = {float(x[0] - x[1]):.5f} ; sqrt(v) = " + " ".join(f"{si:.2e}" for si in s))
        print(f"      ln A = {np.log(la):.5f} ; jacobien 18-dim ln = {np.log(l18) if l18 else float('nan'):.5f} ; modele tour 73 = {mod73:.5f} ; modele avec lambda_s col = {modcol:.5f}")
        print(f"      application 12-dim (x, m) : ln = {np.log(l12):.5f} ; equation caracteristique : ln = {np.log1p(eps_c):.5f} (rapport a ln A : {np.log1p(eps_c) / np.log(la):.4f})")
        print(f"      rapports : lnA / modele73 = {np.log(la) / mod73:.4f} ; lnA / modele col = {np.log(la) / modcol:.4f}" + (f" ; ln18 / modele73 = {np.log(l18) / mod73:.4f} ; ln18 / lnA = {np.log(l18) / np.log(la):.4f}" if l18 else ""), flush=True)
        if ablation:
            print("      ablation du gain de (l3, l4) x k : " + "  ".join(f"k={k:g}: {np.log(lead_A(k)) / mod73:.4f}" for k in (0.25, 1, 4, 16, 100, 1e4, 1e6)))
        out.append((t, np.log(la), np.log(l18) if l18 else None, mod73, np.log(l12), np.log1p(eps_c)))
    return out


def b1_meme_etat(q, xw, lr_delta, t):
    """P74-9 : l'etat du trajet a b1 = 0,9 (lr* + lr_delta) au pas t, puis jacobien 18-dim de l'application des trois modules (B1 = 0,8 ; 0,9 ; 0,95) EN CE MEME ETAT
    (le sejour sur le col n'existe pas aux autres b1 : a b1 = 0,8 et 0,95 le trajet a lr* est au noeud, z3 = 19,0 et 9,205 a t = 200)."""
    import importlib
    import verifier_tour72_retour_et_jacobien as v72
    mods = [importlib.import_module(n) for n in ("reduction_numba_tour74_b1_08", "reduction_numba_tour71_lrvar", "reduction_numba_tour74_b1_095")]
    v72.red = mods[1]
    ls = v72.bissection(q, xw)
    lr = ls + lr_delta
    eps = q * lr
    delta = PLI - mpf(xw)
    mods[1].LRV = lr
    w = mods[1].copie(v72.S1)
    mods[1].avancer(w, float(delta), t, eps=eps, cross=2.0, stop=False)
    st = np.concatenate([w["x"], w["m"], w["v"]])
    s = np.sqrt(w["v"])
    x = [mpf(float(a)) for a in w["x"]]
    J6 = jac6(x, delta)
    J = np.array([[float(J6[i, j]) for j in range(6)] for i in range(6)])
    A = np.eye(6) + lr * np.diag([1.0 / (si + eps) for si in s]) @ J
    ev = np.linalg.eigvals(A)
    la = ev[np.argmax(ev.real)].real
    print(f"q = {q:.3e} xw = {xw} lr = {lr:.12f}, etat a t = {t} (b1 = 0,9) : z3 = {float(x[0] - x[1]):.5f}, ln A = {np.log(la):.5f}")
    for mod in mods:
        v72.red = mod
        mod.LRV = lr
        b1 = mod.B1
        ev18 = np.linalg.eigvals(v72.jacobien(st, w["t"], lr, q, xw))
        l18 = float(np.max(np.abs(ev18)))
        cJ = la - 1.0
        eps_c = 0.5 * (-(1 - b1) * (1 - cJ) + np.sqrt(((1 - b1) * (1 - cJ)) ** 2 + 4 * cJ * (1 - b1)))
        print(f"   b1 = {b1} : jacobien 18-dim ln = {np.log(l18):.5f}, rapport a ln A = {np.log(l18) / np.log(la):.4f} ; equation caracteristique ln = {np.log1p(eps_c):.5f}, rapport {np.log1p(eps_c) / np.log(la):.4f}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "b1etat":
        b1_meme_etat(float(sys.argv[2]), sys.argv[3], float(sys.argv[4]), int(sys.argv[5]))
    elif mode == "fermee":
        mode_fermee()
    elif mode == "precond":
        analyser(float(sys.argv[2]), sys.argv[3], float(sys.argv[4]), [int(a) for a in sys.argv[5:]], ablation=True)
    elif mode == "x":
        # usage : x <q> <lr_delta> <t1,t2,..> <xw1> <xw2> ...
        for xw in sys.argv[5:]:
            analyser(float(sys.argv[2]), xw, float(sys.argv[3]), [int(a) for a in sys.argv[4].split(",")], ablation=True)
