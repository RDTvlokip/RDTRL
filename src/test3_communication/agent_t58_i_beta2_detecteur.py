"""agent (dipankar role), 23/09 : is the non-monotone kick spacing in beta2
(carnet 20/09 : 0.999 -> 462, 0.995 -> 110, 0.99 -> 163-165) a property of
the dynamics or of the detector (|R4 - baseline| > 0.0015, events closer
than GAP_FUSION = 50 steps merged)?
PRECOMMITTED (agent, before the run), from the one-mode Adam toy
(agent_t58_h_jouet_kick.py, period ~0.48/(1-beta2)) :
  S-crossing spacing : 0.999 -> 460-470 ; 0.995 -> 96 +- 8 ; 0.99 -> 52 +- 6
  carnet detector     : reproduces ~460 / ~110 / ~160 (fusion of ~3 true cycles)
If the S-crossing spacing at 0.99 comes out > 100, the artefact reading is wrong.
From the mur23 checkpoint (delta real, step 54000), betas switched at the
checkpoint ; first 5000 steps discarded.
Usage: python agent_t58_i_beta2_detecteur.py <beta2> [steps]"""
import sys
sys.path.insert(0, '.')
import torch
from replay_mur23_referent3 import BETA
from verifier_prior_asymetrique import objectif_pondere
from sauver_checkpoints_mur23_tour58 import reprendre, LR, ADAM_EPS
from verifier_kicks_adam_grille_fine import detecter_evenements

torch.set_num_threads(1)
N = 27
b2 = float(sys.argv[1])
T = int(sys.argv[2]) if len(sys.argv) > 2 else 25000
e, r, opt, poids = reprendre(0.013026615)
for g in opt.param_groups:
    g["betas"] = (0.9, b2)
p_r = r.p[0]
vals, S, VV = [], [], []
for t in range(T):
    j, _ = objectif_pondere(e, r, BETA, poids)
    opt.zero_grad()
    (-j).backward()
    opt.step()
    with torch.no_grad():
        R = r.loi()
        R4 = (e.loi()[4, 10] * R[10, 4]).item()
        lam_y = BETA / N * (R[10, 3] * R[10, 4]).item()
        sv = opt.state[p_r]["exp_avg_sq"][10, 3:5].sqrt().mean().item()
        vals.append((t, R4))
        S.append(LR * 2 * lam_y / (sv + ADAM_EPS))
        VV.append(opt.state[p_r]["exp_avg_sq"][10, 4].item())
# detector-light burst count : a 'pumping' step is one where g^2 > v_{t-1}
# (the fresh gradient raises v above its running level), read off v alone :
# g^2 / v_{t-1} = (v_t / v_{t-1} - b2) / (1 - b2). Bursts = clusters of pumping
# steps separated by more than 5 quiet steps.
VV = torch.tensor(VV[5000:], dtype=torch.float64)
ratio = (VV[1:] / VV[:-1] - b2) / (1 - b2)
pump = (ratio > 1.0).nonzero().flatten().tolist()
starts = [pump[0]] if pump else []
for a, b in zip(pump[:-1], pump[1:]):
    if b - a > 5:
        starts.append(b)
if len(starts) > 2:
    ds = sorted(starts[i + 1] - starts[i] for i in range(len(starts) - 1))
    print(f"beta2={b2}: v-pumping bursts : n={len(starts)}  spacing median {ds[len(ds)//2]}  "
          f"quartiles {ds[len(ds)//4]}-{ds[3*len(ds)//4]}")
S = torch.tensor(S[5000:], dtype=torch.float64)
vals = vals[5000:]
up = (((S[1:] > 38) & (S[:-1] <= 38)).nonzero().flatten() + 1)
sp = (up[1:] - up[:-1]).double()
print(f"beta2={b2}: S-crossing : n={len(up)}  spacing median {sp.median().item():.1f}  mean {sp.mean().item():.1f}  "
      f"<S> {S.mean().item():.2f}  S range [{S.min().item():.2f},{S.max().item():.2f}]")
base = sum(v for _, v in vals) / len(vals)
devs = torch.tensor([abs(v - base) for _, v in vals])
for seuil, fus in ((0.0015, 50), (0.0015, 10), (0.0005, 10), (0.0005, 25), (0.0003, 25), (0.0002, 5), (0.0001, 3)):
    pics = detecter_evenements(vals, base, seuil=seuil, gap_fusion=fus, pas_min=0)
    if len(pics) > 2:
        es = sorted(pics[i + 1][0] - pics[i][0] for i in range(len(pics) - 1))
        med = es[len(es) // 2]
    else:
        med = float("nan")
    print(f"   detector seuil={seuil} gap_fusion={fus}: n={len(pics)}  spacing median {med}")
print(f"   |R4-base| max {devs.max().item():.2e}  99th pct {devs.quantile(0.99).item():.2e}")
