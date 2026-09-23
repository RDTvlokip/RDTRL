"""agent (dipankar role), 23/09 : seed 12345 k=3 has three 'orphans' (no message
decodes to them) whose sender rows are COLLAPSED, not uniform. Why?
Usage: python agent_t58_k_orphelins_effondres.py <seed> <k> <steps>"""
import sys
sys.path.insert(0, '.')
import math
import torch
from replay_mur23_referent3 import replay, BETA
from representable_atteignable_stable import activer, parametres, objectif

torch.set_num_threads(1)
N = 27
graine, k, pas = (int(a) for a in sys.argv[1:4])
e, r = replay(graine, k, 0)
activer(e, r)
opt = torch.optim.Adam(parametres(e, r), lr=0.05)
for t in range(pas):
    j, _ = objectif(e, r, BETA)
    opt.zero_grad()
    (-j).backward()
    opt.step()
p_e, p_r = e.p[0], r.p[0]
j, _ = objectif(e, r, BETA)
opt.zero_grad()
(-j).backward()
with torch.no_grad():
    S, R = e.loi(), r.loi()
    dec = R.argmax(1)
    covered = set(dec.tolist())
    for i in range(N):
        if i in covered:
            continue
        m = S[i].argmax().item()
        other = dec[m].item()
        Hr = -(R[m] * R[m].clamp_min(1e-300).log()).sum().item()
        z = p_e[i]
        gap = (z[m] - torch.cat([z[:m], z[m + 1:]]).max()).item()
        print(f"orphan {i}: s[{i},{m}]={S[i, m].item():.12f}  H(s_{i})={-(S[i]*S[i].clamp_min(1e-300).log()).sum().item():.3e}  "
              f"message {m} decodes to {other} (s[{other},{m}]={S[other, m].item():.6f})  r[{m},{i}]={R[m, i].item():.3e}  "
              f"r[{m},{other}]={R[m, other].item():.6f}  H(r_{m})={Hr:.3e}")
        print(f"   logit gap of row {i} (m vs next) = {gap:.3f} ; conditional optimum logit 50*r[m,i] = {50*R[m, i].item():.3e}")
        g = p_e.grad[i]
        v = opt.state[p_e]["exp_avg_sq"][i]
        print(f"   |grad row {i}| max = {g.abs().max().item():.3e}   sqrt v max = {v.sqrt().max().item():.3e}  "
              f"step size bound lr*|g|/eps = {0.05*g.abs().max().item()/1e-8:.3e} per step")
