"""Tour 59 (28/09/2026) : verification INDEPENDANTE (regle 5bis) de deux
affirmations de l'agent style dipankar, sur le RESEAU COMPLET (2 x 729
parametres), sans passer par sa reduction a deux lignes.

(1) Le mode critique du pli est-il ~98 % logit de l'emetteur (ligne 3),
    ~2 % recepteur (u) ? On converge vers le noeud a delta = pli - 1e-6
    (Newton, pseudo-inverse : les modes de translation du softmax sont
    des zeros exacts), on prend le hessien de l'objectif pondere, on
    retire les zeros de translation, et on lit le vecteur propre de plus
    petite |valeur propre| : poids par groupe de coordonnees.
(2) Pentes de la courbe des noeuds : dd3/ddelta et dr4/ddelta a
    pli - 1e-6 (l'agent : 53,89 et 418,6), depuis ma propre fermeture
    (verifier_tour59_branches_fermees.py).
Etat de depart : cache chauffe eps 1e-10 (20 000 + 20 000 pas).
"""

import sys
sys.path.insert(0, '.')
import numpy as np
import torch

from replay_mur23_referent3 import BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere
from verifier_tour59_delta_c_dynamique import depart
from verifier_tour59_branches_fermees import branche_et_jumeau, mesures, F_PLI

MSG, A, B_ = 10, 3, 4


def charger(delta):
    (e, r), _ = depart("mur23")
    activer(e, r)
    c = torch.load("D:/tmp/rdtrl_tour59_chaud_mur23_3_4_eps1e-10_20000_20000.pt")
    params = parametres(e, r)
    with torch.no_grad():
        for p, v in zip(params, c["params"]):
            p.copy_(v)
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[A], poids[B_] = (1 - delta) / N, (1 + delta) / N
    return e, r, params, poids


def main():
    torch.set_num_threads(4)
    delta = F_PLI - 1e-6
    e, r, params, poids = charger(delta)
    shapes = [p.shape for p in params]
    sizes = [p.numel() for p in params]

    def vers_params(x):
        out, i = [], 0
        for s, n in zip(shapes, sizes):
            out.append(x[i:i + n].view(s)); i += n
        return out

    def J(x):
        pe, pr = vers_params(x)
        e.p[0], r.p[0] = pe, pr
        j, _ = objectif_pondere(e, r, BETA, poids)
        return j

    x = torch.cat([p.detach().reshape(-1) for p in params]).clone()
    for it in range(12):
        x.requires_grad_(True)
        g = torch.autograd.grad(J(x), x)[0]
        H = torch.autograd.functional.hessian(J, x.detach())
        x = x.detach()
        w, V = torch.linalg.eigh(H)
        # pseudo-inverse en ignorant les zeros de translation (|w| < 1e-11)
        mask = w.abs() > 1e-11
        # Newton sans col (ascension de J) : pas g_i/|w_i| sur chaque mode, pour converger vers
        # un MAXIMUM (le noeud) et non vers le point stationnaire le plus proche (le jumeau, un col).
        dx = V[:, mask] @ ((V[:, mask].T @ g) / w[mask].abs())
        x = x + dx
        if it % 3 == 0 or dx.norm() < 1e-13:
            print(f"Newton {it} : |g|={g.norm():.3e}  |dx|={dx.norm():.3e}")
        if dx.norm() < 1e-13:
            break
    x.requires_grad_(True)
    g = torch.autograd.grad(J(x), x)[0]
    H = torch.autograd.functional.hessian(J, x.detach()).detach()
    x = x.detach()
    w, V = torch.linalg.eigh(H)
    print(f"|g| au point stationnaire = {g.norm():.3e}")
    with torch.no_grad():
        pe, pr = vers_params(x)
        e.p[0], r.p[0] = pe, pr
        S_, R_ = e.loi(), r.loi()
        d3_, r4_ = 1 - S_[A, MSG].item(), R_[MSG, B_].item()
    rs = branche_et_jumeau(delta)
    (dA, _, rA), (dJ, _, rJ) = mesures(rs[0], delta), mesures(rs[1], delta)
    print(f"point atteint : d3={d3_:.6e} r4={r4_:.6f} ; noeud ferme d3={dA:.6e} r4={rA:.6f} ; jumeau ferme d3={dJ:.6e} r4={rJ:.6f}")
    # Le reseau est presque partout sature : l'essentiel du spectre est ~0 (lignes
    # decodees a 1e-10), et la plus petite |valeur propre| n'est PAS le mode critique.
    # On cherche donc le mode dont le poids est concentre sur le bloc de la collision
    # (emetteur lignes 3 et 4, recepteur ligne 10), hors translations du softmax.
    bloc = torch.zeros(2 * N * N, dtype=torch.bool)
    bloc[A * N:(A + 1) * N] = True
    bloc[B_ * N:(B_ + 1) * N] = True
    bloc[N * N + MSG * N:N * N + (MSG + 1) * N] = True
    poids_bloc = (V[bloc] ** 2).sum(0)
    cand = ((poids_bloc > 0.9) & (w.abs() > 1e-13)).nonzero().flatten()
    print(f"modes a >90 % dans le bloc de la collision et hors translation : {len(cand)}")
    for k in cand[torch.argsort(w[cand].abs())][:6]:
        print(f"  valeur propre {w[k]:+.4e}   poids dans le bloc {float(poids_bloc[k]):.4f}")
    # sous-bloc de la collision seul (81 coordonnees), pour comparaison
    idx = bloc.nonzero().flatten()
    wb, Vb = torch.linalg.eigh(H[idx][:, idx])
    print("valeurs propres du sous-bloc (81 coordonnees), les plus petites en |.| hors zeros :")
    nzb = (wb.abs() > 1e-13).nonzero().flatten()
    for k in nzb[torch.argsort(wb[nzb].abs())][:4]:
        print(f"  {wb[k]:+.4e}")
    print("tous les modes du bloc, hors translation (|lambda| > 1e-13) : lambda / poids e3,msg / e3,conc. / e4 / r(10,3-4) / r reste")
    def repartition(v):
        ve_, vr_ = v[:N * N].view(N, N), v[N * N:].view(N, N)
        t = float((v ** 2).sum())
        return (float(ve_[A, MSG] ** 2) / t, float((ve_[A] ** 2).sum() - ve_[A, MSG] ** 2) / t, float((ve_[B_] ** 2).sum()) / t,
                float(vr_[MSG, A] ** 2 + vr_[MSG, B_] ** 2) / t, float((vr_[MSG] ** 2).sum() - vr_[MSG, A] ** 2 - vr_[MSG, B_] ** 2) / t)
    for k in cand[torch.argsort(w[cand].abs(), descending=True)]:
        print(f"  {w[k]:+.4e}   " + " / ".join(f"{100 * x:5.1f}" for x in repartition(V[:, k])))
    # le mode critique melange logit de l'emetteur ET recepteur (u) : plus grand poids croise e3,msg x r(10,3-4)
    croise = torch.tensor([float(min(repartition(V[:, k])[0] + repartition(V[:, k])[1], repartition(V[:, k])[3])) for k in cand])
    k = cand[int(croise.argmax())]
    print(f"mode le plus MELANGE emetteur/recepteur : lambda = {w[k]:+.4e}")
    v = V[:, k]
    ve = v[:N * N].view(N, N); vr = v[N * N:].view(N, N)
    groupes = {
        "emetteur ligne 3, message 10": ve[A, MSG] ** 2,
        "emetteur ligne 3, 26 concurrents": (ve[A] ** 2).sum() - ve[A, MSG] ** 2,
        "emetteur ligne 4 (toute)": (ve[B_] ** 2).sum(),
        "recepteur (10,3) et (10,4)": vr[MSG, A] ** 2 + vr[MSG, B_] ** 2,
        "recepteur reste de la ligne 10": (vr[MSG] ** 2).sum() - vr[MSG, A] ** 2 - vr[MSG, B_] ** 2,
    }
    total = float((v ** 2).sum())
    reste = total - float(sum(groupes.values()))
    for nom, val in groupes.items():
        print(f"  poids {100 * float(val) / total:7.3f} %  {nom}")
    print(f"  poids {100 * reste / total:7.3f} %  tout le reste (25 autres lignes des deux tables)")
    # sensibilite pure au message : composante sur z3 = e[3,10] - moyenne des concurrents
    print(f"  ve[3,10]={ve[A, MSG]:+.4f}  moyenne ve[3,autres]={float((ve[A].sum() - ve[A, MSG]) / (N - 1)):+.4f}")

    print("\n(2) pentes de la courbe des noeuds a pli-1e-6 (ma fermeture, difference centree)")
    h = 1e-9
    vals = []
    for d in (F_PLI - 1e-6 - h, F_PLI - 1e-6 + h):
        rs = branche_et_jumeau(d)
        d3, _, r4 = mesures(rs[0], d)
        vals.append((d3, r4))
    print(f"  dd3/ddelta = {(vals[1][0] - vals[0][0]) / (2 * h):.3f}  (agent 53,89)   "
          f"dr4/ddelta = {(vals[1][1] - vals[0][1]) / (2 * h):.2f}  (agent 418,6)")


if __name__ == "__main__":
    main()
