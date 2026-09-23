"""agent (dipankar role), 23/09 : the spontaneous collisions (receiver row at
exact 50/50, both sender rows collapsed) found by agent_t58_l_recensement.py
are K=2 entropy blocks on the RECEIVER side. PRECOMMITTED before the run :
same universal constants as the synonym pairs at eps=1e-8 (toy
agent_t58_e_jouet.py AGENT_K=2, eps=1e-8) :
  rms(x)/(lr/38 - eps/lam) in [1.06, 1.10],  <sqrt v>/(lr lam/38 - eps) = 1.079 +- 0.01,
  lam = beta/(2N), x = displacement of each of the two receiver logits from their mid-point.
Usage: python agent_t58_m_collisions_spontanees.py <seed> <k> <steps_before> <window>"""
import sys
sys.path.insert(0, '.')
import math
import torch
from replay_mur23_referent3 import replay, BETA
from representable_atteignable_stable import activer, parametres, objectif

torch.set_num_threads(1)
N = 27
LR, EPS = 0.05, 1e-8
graine, k, avant, W = (int(a) for a in sys.argv[1:5])
e, r = replay(graine, k, 0)
activer(e, r)
opt = torch.optim.Adam(parametres(e, r), lr=LR)
for _ in range(avant):
    j, _ = objectif(e, r, BETA)
    opt.zero_grad()
    (-j).backward()
    opt.step()
p_r = r.p[0]
with torch.no_grad():
    R = r.loi()
    coll = {m: [i for i in range(N) if R[m, i] > 0.05] for m in range(N) if (R[m] > 0.05).sum() > 1}
print(f"seed={graine} k={k}: collisions {coll}")
acc = {m: {"x2": 0.0, "sv": 0.0, "pmax": 0.0} for m in coll}
for t in range(W):
    j, _ = objectif(e, r, BETA)
    opt.zero_grad()
    (-j).backward()
    opt.step()
    with torch.no_grad():
        V = opt.state[p_r]["exp_avg_sq"]
        for m, (a, b) in coll.items():
            x = 0.5 * (p_r[m, a] - p_r[m, b]).item()
            acc[m]["x2"] += x * x / W
            acc[m]["sv"] += 0.5 * (V[m, a].sqrt() + V[m, b].sqrt()).item() / W
            acc[m]["pmax"] = max(acc[m]["pmax"], abs(torch.sigmoid(p_r[m, a] - p_r[m, b]).item() - 0.5))
lam = BETA / (2 * N)
for m in coll:
    print(f"   receiver row {m}: rms(x)/(lr/38-eps/lam) = {math.sqrt(acc[m]['x2'])/(LR/38-EPS/lam):.4f}   "
          f"<sqrt v>/(lr lam/38-eps) = {acc[m]['sv']/(LR*lam/38-EPS):.4f}   max|r-1/2| over window = {acc[m]['pmax']:.2e}")
