"""Tour 59 (29/09/2026) : pente asymptotique de kappa(eps) recalculee de
facon INDEPENDANTE de la reduction de l'agent style dipankar, sur le
reseau complet (2 x 729 parametres).

A eps >> sqrt(v), Adam se reduit a (lr/eps) * gradient. Le long du mode
nul unitaire n du hessien au pli (ascension de J) :
    ds/dt = (lr/eps) (alpha * mu + beta * s^2),
    alpha = n . d(grad J)/d delta,   beta = 1/2 D3J[n, n, n],
    mu = delta - delta_c.
Temps de passage total pi / ((lr/eps) sqrt(alpha beta mu)), donc
    kappa / eps = pi / (lr sqrt(alpha beta)).
Evalue au noeud a pli - 1e-9 (le mode mou y est presque exactement le mode
nul du pli). D3J[n,n,n] par difference seconde de psi(h) = n . grad J(x + h n).
Compare a 1,617e6 (mesure E2, D = 1e-7) et 1,637e6 (formule de l'agent).
"""

import sys
sys.path.insert(0, '.')
import math
import torch

from replay_mur23_referent3 import BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere
from verifier_tour59_delta_c_dynamique import depart
from verifier_tour59_branches_fermees import F_PLI

MSG, A, B_ = 10, 3, 4
LR = 0.05


def main():
    torch.set_num_threads(4)
    (e, r), _ = depart("mur23")
    activer(e, r)
    c = torch.load("D:/tmp/rdtrl_tour59_chaud_mur23_3_4_eps1e-10_20000_20000.pt")
    params = parametres(e, r)
    with torch.no_grad():
        for p, v in zip(params, c["params"]):
            p.copy_(v)
    shapes = [p.shape for p in params]
    sizes = [p.numel() for p in params]

    def poids_pour(delta):
        p = torch.full((N,), 1.0 / N, dtype=torch.float64)
        p[A], p[B_] = (1 - delta) / N, (1 + delta) / N
        return p

    def vers_params(x):
        out, i = [], 0
        for s, n in zip(shapes, sizes):
            out.append(x[i:i + n].view(s)); i += n
        return out

    def J(x, delta):
        e.p[0], r.p[0] = vers_params(x)
        return objectif_pondere(e, r, BETA, poids_pour(delta))[0]

    def grad(x, delta):
        x = x.detach().clone().requires_grad_(True)
        return torch.autograd.grad(J(x, delta), x)[0]

    delta = F_PLI - 1e-9
    x = torch.cat([p.detach().reshape(-1) for p in params]).clone()
    for it in range(40):  # Newton sans col vers le noeud
        g = grad(x, delta)
        H = torch.autograd.functional.hessian(lambda z: J(z, delta), x)
        w, V = torch.linalg.eigh(H)
        m = w.abs() > 1e-11
        dx = V[:, m] @ ((V[:, m].T @ g) / w[m].abs())
        x = x + dx
        if dx.norm() < 1e-13:
            break
    g = grad(x, delta)
    H = torch.autograd.functional.hessian(lambda z: J(z, delta), x)
    w, V = torch.linalg.eigh(H)
    bloc = torch.zeros(2 * N * N, dtype=torch.bool)
    bloc[A * N:(A + 1) * N] = True; bloc[B_ * N:(B_ + 1) * N] = True; bloc[N * N + MSG * N:N * N + (MSG + 1) * N] = True
    pb = (V[bloc] ** 2).sum(0)
    # mode mou : dans le bloc, hors translation, dominant sur la ligne 3 de l'emetteur (logit du message)
    poids_e3 = V[A * N + MSG] ** 2
    cand = ((pb > 0.9) & (w.abs() > 1e-13) & (poids_e3 > 0.5)).nonzero().flatten()
    k = cand[int(w[cand].abs().argmin())]
    n = V[:, k]
    print(f"noeud a pli-1e-9 : |g|={g.norm():.2e} ; mode mou lambda={w[k]:+.3e}, poids e3,msg={float(poids_e3[k]):.3f}")

    h_d = 1e-9
    dg = (grad(x, delta + h_d) - grad(x, delta - h_d)) / (2 * h_d)
    alpha = float(n @ dg)
    print(f"alpha = n . d(grad J)/d delta = {alpha:+.6e}")

    print("beta = 1/2 D3J[n,n,n] par difference seconde de psi(h) = n . grad J(x + h n) :")
    betas = []
    for h in (3e-3, 1e-3, 3e-4):
        psi = lambda s: float(n @ grad(x + s * n, delta))
        d3 = (psi(h) + psi(-h) - 2 * psi(0.0)) / h ** 2
        betas.append(0.5 * d3)
        print(f"   h={h:.0e} : D3J[n,n,n] = {d3:+.6e}   beta = {0.5 * d3:+.6e}")
    beta = betas[1]
    if alpha * beta <= 0:
        print("alpha*beta <= 0 : pas de passage de goulot avec ces signes ; verifier le signe de n")
        return
    pente = math.pi / (LR * math.sqrt(alpha * beta))
    print(f"kappa/eps = pi/(lr sqrt(alpha beta)) = {pente:.4e}   (E2 mesure 1,617e6 ; agent 1,637e6)")


if __name__ == "__main__":
    main()
