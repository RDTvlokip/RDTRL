"""agent (dipankar role), 23/09 : (a) is the uniform the exact optimum of
row 5 given the receiver, and how far does the reward term move it?
Also for the synonym pairs 8/12. Read-only on the checkpoint."""
import sys
sys.path.insert(0, '.')
import math
import torch
from replay_mur23_referent3 import BETA
from verifier_prior_asymetrique import objectif_pondere
from sauver_checkpoints_mur23_tour58 import reprendre, LR, ADAM_EPS

torch.set_num_threads(1)
N = 27
e, r, opt, poids = reprendre(0.013026615)
p_e, p_r = e.p[0], r.p[0]
with torch.no_grad():
    S, R = e.loi(), r.loi()
    col5 = R[:, 5]
    print("r[:,5] top 6:", sorted([(round(col5[m].item(), 15), m) for m in range(N)], reverse=True)[:6])
    print("sum r[:,5] =", col5.sum().item(), " reward_5 =", (S[5] * col5).sum().item())
    # conditional optimum of row 5 given receiver: s* = softmax(N poids_5 r[:,5] / beta)
    zstar = N * poids[5].item() * col5 / BETA
    sstar = torch.softmax(zstar, 0)
    dstar = math.log(N) + (sstar * sstar.log()).sum().item()
    print(f"optimum logit spread (max-min) = {(zstar.max()-zstar.min()).item():.4e}   deficit of optimum = {dstar:.4e}")
    d_now = math.log(N) + (S[5] * S[5].log()).sum().item()
    print(f"deficit now = {d_now:.4e}")
# gradient decomposition on row 5
s = e.loi()
Rr = r.loi()
rew = (poids.unsqueeze(1) * s * Rr.t()).sum()
g_rew = torch.autograd.grad(rew, p_e)[0][5]
s = e.loi()
ent = BETA * (-(s * torch.log(s)).sum() / N)
g_ent = torch.autograd.grad(ent, p_e)[0][5]
print(f"|grad reward| row5 max = {g_rew.abs().max().item():.4e}   |grad entropy| row5 max = {g_ent.abs().max().item():.4e}")
st = opt.state[p_e]
print(f"sqrt v row5 min/mean/max = {st['exp_avg_sq'][5].sqrt().min().item():.4e} {st['exp_avg_sq'][5].sqrt().mean().item():.4e} {st['exp_avg_sq'][5].sqrt().max().item():.4e}")
print(f"step count = {st['step']}")
# synonyms
for i, (a, b) in {8: (23, 19), 12: (1, 21)}.items():
    with torch.no_grad():
        ra, rb = Rr[a, i].item(), Rr[b, i].item()
        # conditional optimum of split d = z_a - z_b: d* = N poids_i (r_a - r_b)/beta
        dstar = N * poids[i].item() * (ra - rb) / BETA
        # rest of row i: mass off the pair
        off = 1 - S[i, a].item() - S[i, b].item()
        print(f"ref {i}: r[a]-r[b] = {ra-rb:.3e}  d* = {dstar:.3e}  p*-1/2 = {dstar/4:.3e}  mass off pair = {off:.3e}  p-1/2 now = {(S[i,a]/(S[i,a]+S[i,b])-0.5).item():.3e}")
        # which other rows put mass on messages a, b
        for m in (a, b):
            col = S[:, m]
            print(f"    message {m}: s[:,m] top =", sorted([(f'{col[k].item():.3e}', k) for k in range(N)], key=lambda x: -float(x[0]))[:4])
