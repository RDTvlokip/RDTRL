"""agent (dipankar role), 23/09 : the orphan's uniform row is the optimum of its
row GIVEN the receiver. Is the code itself optimal? Move referent 5 onto
synonym message 19 (row 5 +30 on 19, row 8 -30 on 19, receiver row 19 +30 on
referent 5), relax both copies 5000 steps under Adam, compare J.
PRECOMMITTED : Delta J = 1/N - (beta/N)(ln 27 + ln 2) = +0.03408, and the moved
state does not revert (referent 5 keeps message 19).
Standard replay 77777 k=3 at 30000 steps (orphans 4 and 5, synonyms 8, 12)."""
import sys
sys.path.insert(0, '.')
import copy
import math
import torch
from replay_mur23_referent3 import replay, BETA
from representable_atteignable_stable import activer, parametres, objectif

torch.set_num_threads(1)
N = 27
e, r = replay(77777, 3, 0)
activer(e, r)
opt = torch.optim.Adam(parametres(e, r), lr=0.05)
for _ in range(30000):
    j, _ = objectif(e, r, BETA)
    opt.zero_grad()
    (-j).backward()
    opt.step()
e2, r2 = copy.deepcopy(e), copy.deepcopy(r)
with torch.no_grad():
    e2.p[0][5, 19] += 30.0
    e2.p[0][8, 19] -= 30.0
    r2.p[0][19, 5] += 30.0
res = []
for (ee, rr) in ((e, r), (e2, r2)):
    activer(ee, rr)
    o = torch.optim.Adam(parametres(ee, rr), lr=0.05)
    for _ in range(5000):
        j, _ = objectif(ee, rr, BETA)
        o.zero_grad()
        (-j).backward()
        o.step()
    with torch.no_grad():
        j, rew = objectif(ee, rr, BETA)
        res.append((j.item(), rew.item(), rr.loi()[19].argmax().item(), ee.loi()[5].max().item()))
pred = 1 / N - BETA / N * (math.log(27) + math.log(2))
print(f"J orphan code = {res[0][0]:.6f} (reward {res[0][1]:.6f})   J moved code = {res[1][0]:.6f} (reward {res[1][1]:.6f})")
print(f"Delta J = {res[1][0]-res[0][0]:.6f}   predicted {pred:.6f}   message 19 decodes to {res[1][2]}, max s[5,:] = {res[1][3]:.6f}")
