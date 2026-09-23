"""agent (dipankar role), 23/09 : census of edge-of-stability directions in the
WHOLE system (1458 parameters) at the mur23 checkpoint, delta real and delta 0.
Full Hessian of -J (autograd), preconditioned spectrum lr * eig(P^-1/2 H P^-1/2),
P = sqrt(v) + eps from the saved Adam state. Criterion to test (precommitted):
a direction can sit at the edge of stability only if lr * lambda / eps > 38,
i.e. lambda > 38 eps / lr = 7.6e-8 ; every such direction should show
lr*lambda/P within a few units of 38, every other one far below.
Usage: python agent_t58_g_census.py <delta> [n_steps_before]"""
import sys
sys.path.insert(0, '.')
import torch
from replay_mur23_referent3 import BETA
from representable_atteignable_stable import parametres
from verifier_prior_asymetrique import objectif_pondere
from sauver_checkpoints_mur23_tour58 import reprendre, LR, ADAM_EPS

torch.set_num_threads(1)
N = 27
delta = float(sys.argv[1])
avance = int(sys.argv[2]) if len(sys.argv) > 2 else 0
e, r, opt, poids = reprendre(delta)
params = parametres(e, r)
for _ in range(avance):
    j, _ = objectif_pondere(e, r, BETA, poids)
    opt.zero_grad()
    (-j).backward()
    opt.step()
theta0 = torch.cat([p.detach().flatten() for p in params])
shapes = [p.shape for p in params]


class Box:
    pass


def f(theta):
    pe = theta[:729].reshape(27, 27)
    pr = theta[729:].reshape(27, 27)
    s = torch.softmax(pe, 1)
    R = torch.softmax(pr, 1)
    rew = (poids.unsqueeze(1) * s * R.t()).sum()
    ent_s = -(s * torch.log(s.clamp_min(1e-300))).sum() / N
    ent_r = -(R * torch.log(R.clamp_min(1e-300))).sum() / N
    return -(rew + BETA * (ent_s + ent_r))


# sanity : f matches the training objective
with torch.no_grad():
    j, _ = objectif_pondere(e, r, BETA, poids)
print(f"f(theta0) = {f(theta0).item():.15e}   -J = {(-j).item():.15e}")
H = torch.autograd.functional.hessian(f, theta0)
H = 0.5 * (H + H.T)
V = torch.cat([opt.state[p]["exp_avg_sq"].flatten() for p in params])
P = V.sqrt() + ADAM_EPS
lamH = torch.linalg.eigvalsh(H)
print(f"raw Hessian of -J : min eig {lamH.min().item():.3e}  max eig {lamH.max().item():.3e}  "
      f"#eig < -1e-12 : {(lamH < -1e-12).sum().item()}  #eig > 38 eps/lr = {38*ADAM_EPS/LR:.2e} : {(lamH > 38*ADAM_EPS/LR).sum().item()}")
neg = lamH[lamH < -1e-14]
if len(neg):
    print("   most negative eigenvalues:", [f"{x:.3e}" for x in neg[:8].tolist()])
Dm = P.rsqrt()
M = Dm[:, None] * H * Dm[None, :]
lam, W = torch.linalg.eigh(M)
S = LR * lam
order = torch.argsort(S, descending=True)
print(f"preconditioned: #S>30 : {(S > 30).sum().item()}  #S in (10,30] : {((S > 10) & (S <= 30)).sum().item()}  "
      f"#S in (1,10] : {((S > 1) & (S <= 10)).sum().item()}  max S {S.max().item():.3f}")


def where(w):
    w2 = w * w
    e2 = w2[:729].reshape(27, 27)
    r2 = w2[729:].reshape(27, 27)
    rows_e = e2.sum(1)
    rows_r = r2.sum(1)
    out = []
    for i in torch.argsort(rows_e, descending=True)[:2].tolist():
        if rows_e[i] > 0.05:
            out.append(f"e{i}:{rows_e[i]:.2f}")
    for i in torch.argsort(rows_r, descending=True)[:2].tolist():
        if rows_r[i] > 0.05:
            out.append(f"r{i}:{rows_r[i]:.2f}")
    return " ".join(out)


for k in order[:40].tolist():
    print(f"  S = {S[k].item():8.3f}   {where(W[:, k])}")
# curvature criterion : for each of the top preconditioned modes, raw curvature w^T H w along the
# unpreconditioned direction
