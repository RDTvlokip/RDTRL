"""agent (dipankar role), 23/09 : code structure of a standard replay (Adam
default eps=1e-8, lr=0.05, objectif() unweighted) : which referents are
orphans, which messages are synonyms, which rows are non-collapsed, and the
per-row EoS statistics over a window -- to test out of sample the universal
constants c_K measured on the toys (K = size of the entropy block).
Usage: python agent_t58_j_replay_structure.py <seed> <k> <steps_before> <window>"""
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
p_e = e.p[0]
with torch.no_grad():
    S, R = e.loi(), r.loi()
    dec = R.argmax(1)                         # message -> decoded referent
    groups = {}
    for m in range(N):
        groups.setdefault(dec[m].item(), []).append(m)
    orphans = [i for i in range(N) if i not in groups]
    multi = {i: ms for i, ms in groups.items() if len(ms) > 1}
    print(f"seed={graine} k={k} step={avant}: orphans {orphans}, synonym groups {multi}")
    Hrow = -(S * S.clamp_min(1e-300).log()).sum(1)
    nc = [i for i in range(N) if Hrow[i] > 1e-6]
    print("non-collapsed sender rows:", [(i, round(Hrow[i].item(), 6)) for i in nc])
# window statistics for each non-collapsed row
stats = {i: {"x2": [], "sv": []} for i in nc}
for t in range(W):
    j, _ = objectif(e, r, BETA)
    opt.zero_grad()
    (-j).backward()
    opt.step()
    with torch.no_grad():
        V = opt.state[p_e]["exp_avg_sq"]
        for i in nc:
            z = p_e[i]
            s = torch.softmax(z, 0)
            if i in orphans:
                x = z - z.mean()
                stats[i]["x2"].append((x * x).mean().item())
                stats[i]["sv"].append(V[i].sqrt().mean().item())
            else:
                # block = messages carrying the row's mass
                blk = [m for m in range(N) if s[m] > 1e-3]
                zb = z[blk]
                x = zb - zb.mean()
                stats[i]["x2"].append((x * x).mean().item())
                stats[i]["sv"].append(V[i, blk].sqrt().mean().item())
                stats[i]["K"] = len(blk)
for i in nc:
    K = N if i in orphans else stats[i]["K"]
    lam = BETA / (N * K)                       # eigenvalue of -beta H/N on the zero-sum block of size K
    base = LR / 38 - EPS / lam
    rms = math.sqrt(sum(stats[i]["x2"]) / W)
    sv = sum(stats[i]["sv"]) / W
    print(f"row {i:2d} ({'orphan' if i in orphans else 'synonym block'}, K={K}): "
          f"rms(x)/(lr/38-eps/lam) = {rms/base:.4f}   <sqrt v>/(lr lam/38 - eps) = {sv/(LR*lam/38-EPS):.4f}")
