"""Tour 76 (vraie critique de dipankarsarkar, 21/10/2026) : somme ou repulsion de niveaux ?

Il dit : dans la trace du 23/09 (reseau complet, delta reel), S_gap = 37,91 (diagonale du gap), S_max = 39,74, poids du vecteur propre sur la ligne 3 de l'emetteur 4,9 % (dans P^(1/2) H P^(1/2),
c'est la matrice dont la trace fait l'iteration de puissance). Bloc 2 x 2 symetrique (gap, emetteur) : (S_max - d_gap)/(S_max - d_s) = 0,049/0,951 => d_s = 4,22, partenaire 2,39, couplage 8,06 :
repulsion de niveaux ; une somme pure donnerait d_s = 1,95 et S_gap = 37,79. Ce qui tranche : la diagonale propre de la ligne emetteur de lr P H, et le second niveau du bloc.

Modes :
  arithmetique                         : son calcul et celui avec les valeurs exactes du journal (pas 59536)
  reseau <pas_debut> <pas_fin>          : rejeu du reseau complet (checkpoint du 23/09, delta reel 0,013026615), pour chaque pas : S_gap, S_max (iteration de puissance), part e3,
                                          puis le bloc 2 x 2 de lr P^(-1/2) H P^(-1/2) dans (direction du gap, direction du couplage sur la ligne 3 de l'emetteur = projection de M g
                                          sur la ligne 3, normalisee) : d_gap, d_s, couplage, niveaux, fermeture (part de M g hors du bloc)
  reduction <b1> <eps> <x> [<nrec>]     : meme bloc dans la reduction (6 coordonnees, 26 concurrents fusionnes en o3, symetrise par o3' = sqrt(26) o3), a trois instants de franchissement S_c = seuil
"""

import math
import os
import sys

import numpy as np
from mpmath import mp, mpf

sys.path.insert(0, ".")

mp.dps = 50


def mode_arithmetique():
    for nom, S, dg, ws in (("ses nombres arrondis", 39.74, 37.91, 0.049), ("journal, pas 59536", 39.739, 37.912, 0.0491)):
        wg = 1 - ws
        r = ws / wg
        Sd = S - (S - dg) / r
        print(f"{nom} : (S - d_gap)/(S - d_s) = {r:.4f} ; d_s = {Sd:.3f} ; partenaire = {dg + Sd - S:.3f} ; couplage = {math.sqrt((S - dg) * (S - Sd)):.3f} ; somme pure : d_s = {ws * S:.3f}, S_gap = {S - ws * S:.3f}")


def mode_reseau(debut, fin):
    import torch
    from replay_mur23_referent3 import BETA
    from representable_atteignable_stable import parametres
    from verifier_prior_asymetrique import objectif_pondere
    from sauver_checkpoints_mur23_tour58 import reprendre, LR, ADAM_EPS
    torch.set_num_threads(1)
    torch.manual_seed(0)
    delta = 0.013026615

    def pas_adam(e, r, opt, poids):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()

    def dot(a, b):
        return sum((x * y).sum() for x, y in zip(a, b))

    def mesurer(e, r, opt, poids, w_init, niter):
        params = parametres(e, r)
        j, _ = objectif_pondere(e, r, BETA, poids)
        grads = torch.autograd.grad(-j, params, create_graph=True)
        D = [opt.state[p]["exp_avg_sq"].sqrt() + ADAM_EPS for p in params]
        dmh = [d.rsqrt() for d in D]

        def M(w):
            u = [a * b for a, b in zip(dmh, w)]
            hv = torch.autograd.grad(grads, params, grad_outputs=u, retain_graph=True)
            return [a * b for a, b in zip(dmh, hv)]

        d = [torch.zeros_like(p) for p in params]
        d[1][10, 4], d[1][10, 3] = 1.0, -1.0
        d = [a / b for a, b in zip(d, dmh)]
        nd = torch.sqrt(dot(d, d))
        g = [a / nd for a in d]
        Mg = M(g)
        d_gap = dot(g, Mg).item()
        # direction de couplage : projection de M g sur la ligne 3 de l'emetteur
        s = [torch.zeros_like(p) for p in params]
        s[0][3, :] = Mg[0][3, :]
        c = torch.sqrt(dot(s, s)).item()
        s = [a / c for a in s]
        Ms = M(s)
        d_s = dot(s, Ms).item()
        c2 = dot(s, Mg).item()
        # fermeture : part de M g hors du bloc (g, s)
        res = [a - d_gap * x - c2 * y for a, x, y in zip(Mg, g, s)]
        ferm = (torch.sqrt(dot(res, res)) / torch.sqrt(dot(Mg, Mg))).item()
        # iteration de puissance (comme verifier_tour58_bord_stabilite_salve.py)
        w = w_init if w_init is not None else [torch.randn_like(p) for p in params]
        lam = 0.0
        for _ in range(niter):
            n = torch.sqrt(dot(w, w))
            w = [a / n for a in w]
            mw = M(w)
            lam = dot(w, mw).item()
            w = [x.detach() for x in mw]
        n = torch.sqrt(dot(w, w))
        w = [a / n for a in w]
        part_gap = (w[1][10, 3] ** 2 + w[1][10, 4] ** 2).item()
        part_e3 = (w[0][3] ** 2).sum().item()
        return dict(S_gap=LR * d_gap, S_max=LR * lam, part_gap=part_gap, part_e3=part_e3, d_gap=LR * d_gap, d_s=LR * d_s, c=LR * c2, ferm=ferm), w

    e, r, opt, poids = reprendre(delta)
    for _ in range(debut - 54000):
        pas_adam(e, r, opt, poids)
    w = None
    print(f"rejeu du reseau complet, delta = {delta}, pas {debut} a {fin} (lr 0,05, eps {ADAM_EPS})")
    print("  pas    S_gap   S_max  part gap part e3 | d_gap    d_s    couplage  niveau+  niveau-  fermeture | S_max(bloc)")
    for pas in range(debut, fin):
        m, w = mesurer(e, r, opt, poids, w, 120 if w is None else 40)
        pas_adam(e, r, opt, poids)
        B = np.array([[m["d_gap"], m["c"]], [m["c"], m["d_s"]]])
        ev = np.linalg.eigvalsh(B)
        print(f"  {pas}  {m['S_gap']:7.3f} {m['S_max']:7.3f}  {m['part_gap']:.3f}   {m['part_e3']:.4f}  | {m['d_gap']:7.3f} {m['d_s']:7.3f} {m['c']:8.3f} {ev[1]:8.3f} {ev[0]:8.3f}   {m['ferm']:.3f}   | {ev[1]:.3f}", flush=True)


def bloc_sym(J6, v6, eps, lr):
    """matrice M = lr D^(-1/2) (-Js) D^(-1/2) symetrisee (o3, o4 -> sqrt(26) o3, sqrt(26) o4), D = sqrt(v) + eps."""
    Q = np.diag([1, math.sqrt(26), 1, math.sqrt(26), 1, 1])
    Js = Q @ J6 @ np.linalg.inv(Q)
    asym = np.max(np.abs(Js - Js.T)) / np.max(np.abs(Js))
    D = np.sqrt(v6) + eps
    Dm = 1.0 / np.sqrt(D)
    M = lr * (Dm[:, None] * (-(Js + Js.T) / 2) * Dm[None, :])
    D12 = np.sqrt(D)
    # direction du gap (l3 = -1, l4 = +1) dans les coordonnees de M : D^(1/2) * indicateur
    g = np.zeros(6); g[4], g[5] = -1.0, 1.0
    g = D12 * g; g = g / np.linalg.norm(g)
    Mg = M @ g
    d_gap = g @ Mg
    res = Mg - d_gap * g
    s = np.zeros(6); s[0:2] = res[0:2]
    c = np.linalg.norm(s)
    s = s / c
    d_s = s @ M @ s
    blocs = np.linalg.eigvalsh(np.array([[d_gap, s @ Mg], [s @ Mg, d_s]]))
    ev, vec = np.linalg.eigh(M)
    k = int(np.argmax(ev))
    w = vec[:, k]
    return dict(S_max=ev[k], second=np.sort(ev)[::-1][1:4], asym=asym, d_gap=d_gap, d_s=d_s, c=s @ Mg, niveaux=blocs, part_gap=w[4] ** 2 + w[5] ** 2, part_s=w[0] ** 2 + w[1] ** 2,
                part_autre=w[2] ** 2 + w[3] ** 2, ferm=float(np.linalg.norm(Mg - d_gap * g - (s @ Mg) * s) / np.linalg.norm(Mg)))


def mode_reduction(b1, eps, xs, nrec):
    import importlib
    from verifier_tour75_salves_depart import MODULES, etat_base, courbe_Sc, LR, PLI
    from verifier_tour74_valeur_propre import jac6
    redlib = importlib.import_module(MODULES[b1])
    thr = 2 * (1 + b1) / (1 - b1)
    S1 = etat_base()
    delta = float(PLI - mpf(xs))
    ks = [int(a) for a in np.random.default_rng(7).integers(0, 4444, 16)]
    rng = np.random.default_rng(11)
    w = redlib.copie(S1)
    redlib.avancer(w, float(PLI - mpf("1e-6")), 20000 + ks[0], eps=eps, cross=2.0, stop=False)
    w["x"] = w["x"] * (1.0 + 1e-9 * rng.standard_normal(6))
    redlib.avancer_obs(w, delta, 10000, np.zeros((1, 7)), eps=eps)
    x6 = w["x"].copy()
    w0 = redlib.copie(w)
    rec = np.zeros((nrec, 7))
    first, maxrb, rb, done = redlib.avancer_obs(w, delta, nrec, rec, eps=eps)
    if maxrb > 0.9:
        print(f"b1 = {b1} eps = {eps:.1e} x = {xs} : effondre")
        return
    u = rec[:, 2]; s_e3 = rec[:, 3]; s_l3 = rec[:, 4]
    grille, sc_g, J = courbe_Sc(x6, delta, LR, eps, float(np.median(s_e3 / s_l3)), float(np.median(s_l3)), b1)
    Sc = np.interp(np.log(s_l3), np.log(grille), sc_g)
    D2 = u[2:] - 2 * u[1:-1] + u[:-2]
    A = np.abs(D2) / 4.0
    E = np.maximum(A[:-1], A[1:])
    idx = np.nonzero(E > 3e-4)[0]
    if len(idx) == 0:
        print(f"b1 = {b1} eps = {eps:.1e} x = {xs} : pas de salve (amplitude max {E.max():.2e}), S_c max {Sc.max():.2f} pour un seuil {thr:.0f}")
        return
    coupe = np.nonzero(np.diff(idx) > 30)[0]
    debuts = np.concatenate([[0], coupe + 1]); fins = np.concatenate([coupe, [len(idx) - 1]])
    groupes = [(idx[a], idx[b]) for a, b in zip(debuts, fins)]
    res = []
    for g in range(1, min(len(groupes), 4)):
        prev_end = groupes[g - 1][1]; a = groupes[g][0]
        cand = np.nonzero(Sc[prev_end + 5:a] >= thr)[0]
        if len(cand) == 0:
            continue
        t_c = prev_end + 5 + int(cand[0])
        w1 = redlib.copie(w0)
        redlib.avancer(w1, delta, t_c + 2, eps=eps, cross=2.0, stop=False)
        J6 = jac6([mpf(float(q)) for q in w1["x"]], mpf(delta))
        Jn = np.array([[float(J6[i, j]) for j in range(6)] for i in range(6)])
        bl = bloc_sym(Jn, w1["v"], eps, LR)
        s_e = math.sqrt(w1["v"][0])
        pz = 1.0 / (s_e + eps) + 1.0 / (s_e + 26.0 * eps)
        from verifier_tour74_valeur_propre import valeurs_propres_3x3
        lam_noeud = valeurs_propres_3x3(PLI - mpf(xs), +1)[0][0].real
        bl["pred_soft"] = LR * pz * abs(lam_noeud)
        res.append(bl)
    if not res:
        print(f"b1 = {b1} eps = {eps:.1e} x = {xs} : aucun franchissement trouve")
        return
    f = lambda key: np.mean([r_[key] for r_ in res])
    nv = np.mean([r_["niveaux"] for r_ in res], axis=0)
    sec = np.mean([r_["second"] for r_ in res], axis=0)
    print(f"b1 = {b1} eps = {eps:.1e} x = {xs} ({len(res)} salves) : S_max(6x6) = {f('S_max'):.3f} ; second niveau du 6x6 = {sec[0]:.4f} ; asymetrie de J symetrise {f('asym'):.1e}")
    print(f"   bloc (gap, sender) : d_gap = {f('d_gap'):.3f} ; d_s = {f('d_s'):.3f} ; couplage = {f('c'):.3f} ; niveaux {nv[1]:.3f} et {nv[0]:.4f} ; partenaire/S_max = {nv[0] / f('S_max'):.4f} ; fermeture {f('ferm'):.4f}")
    print(f"   mode mou (H76-4) : lr p_z |lambda_s,noeud(x)| = {f('pred_soft'):.4f} ; second niveau du 6x6 / cette valeur = {sec[0] / f('pred_soft'):.3f} ; partenaire du bloc / cette valeur = {nv[0] / f('pred_soft'):.3f}")
    print(f"   poids du vecteur propre raide : gap {f('part_gap'):.4f} ; sender (e3, sqrt26 o3) {f('part_s'):.4f} ; e4/o4 {f('part_autre'):.1e} ; d_s / S_max = {f('d_s') / f('S_max'):.4f}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "arithmetique":
        mode_arithmetique()
    elif mode == "reseau":
        mode_reseau(int(sys.argv[2]), int(sys.argv[3]))
    else:
        mode_reduction(float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 300000)
