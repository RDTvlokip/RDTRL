"""Is the near-linear ramp of r4 in the K=8, frac=0.03% window
(pas 245-365, dr4/dpas approx constant approx 6.9e-5) caused by Adam's
normalized update saturating (m/(sqrt(v)+eps) -> +-1, i.e. an
optimizer artifact), or is the raw objective gradient itself roughly
constant there (a property of the toy's landscape, not of Adam)?
Log Adam's internal exp_avg (m), exp_avg_sq (v), and the raw gradient
on q (the r4 logit) at every step in the window, plus the *effective*
normalized step actually taken.
"""
import sys
sys.path.insert(0, 'src/test3_communication')
import torch
from verifier_jouet_k_emetteur_variable import construire_toy, objectif_toy, N

K = 8
DELTA_C = 0.015098
frac = 0.0003  # 0.03%
delta = DELTA_C * (1 - frac)
PAS_MAX = 400

poids3 = torch.tensor((1.0 - delta) / N, dtype=torch.float64)
poids4 = torch.tensor((1.0 + delta) / N, dtype=torch.float64)
p3, p4, q = construire_toy(K, s3_init=0.999, s4_init=0.999, r_init=0.5)
lr = 0.05
eps = 1e-10
opt = torch.optim.Adam([p3, p4, q], lr=lr, eps=eps)

print(f"delta={delta:.8f}")
print(f"{'pas':>5} {'r4':>14} {'grad_q':>14} {'m(q)':>14} {'v(q)':>16} {'m/(sqrt(v)+eps)':>18} {'dq_effective':>16}")
for pas in range(PAS_MAX):
    q_before = q.item()
    j = objectif_toy(p3, p4, q, poids3, poids4, K)
    opt.zero_grad()
    (-j).backward()
    grad_q = q.grad.item()
    opt.step()
    q_after = q.item()
    if 240 <= pas <= 370:
        state = opt.state[q]
        m = state['exp_avg'].item()
        v = state['exp_avg_sq'].item()
        t = state['step'].item() if hasattr(state['step'], 'item') else state['step']
        bc1 = 1 - 0.9 ** t
        bc2 = 1 - 0.999 ** t
        m_hat = m / bc1
        v_hat = v / bc2
        norm_update = m_hat / (v_hat ** 0.5 + eps)
        with torch.no_grad():
            r4 = torch.sigmoid(q).item()
        dq = q_after - q_before
        if pas % 5 == 0 or pas < 250:
            print(f"{pas:5d} {r4:14.10f} {grad_q:14.6e} {m:14.6e} {v:16.6e} {norm_update:18.6f} {dq:16.10f}")
