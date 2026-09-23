"""agent (dipankar role), 23/09 : per-step trace of rows 5, 8, 12 from the
mur23 checkpoint (delta real), with the EXACT preconditioned Hessian of
each row (27x27, entropy part analytic, checked once against autograd).
Saves D:/tmp/agent_t58_trace_rows_5_8_12.pt. Read-only on the repo.

Usage: python agent_t58_c_trace.py <n_steps>
"""
import sys
sys.path.insert(0, '.')
import math
import torch
from replay_mur23_referent3 import BETA
from verifier_prior_asymetrique import objectif_pondere
from sauver_checkpoints_mur23_tour58 import reprendre, LR, ADAM_EPS

torch.set_num_threads(1)
N = 27
ROWS = (5, 8, 12)


def hess_negent(z):
    """Hessian wrt z of (beta/N) * sum s ln s (the entropy part of -J for one row)."""
    s = torch.softmax(z, 0)
    L = (s * s.log()).sum()
    a = s.log() - L + 1.0            # ln s_j - L + 1
    b = s * (s.log() - L)            # s_i (ln s_i - L)
    # d2L/dz_j dz_i = s_j (delta_ij - s_i) a_j - s_j b_i
    H = torch.diag(s * a) - torch.outer(s * a, s) - torch.outer(s, b)
    H = 0.5 * (H + H.T)
    return BETA / N * H


def main(T):
    e, r, opt, poids = reprendre(0.013026615)
    p_e = e.p[0]
    # one-off check of the analytic Hessian against autograd, full objective, row 5
    def f5(z5):
        pe = p_e.detach().clone()
        pe = torch.cat([pe[:5], z5.unsqueeze(0), pe[6:]], 0)
        s = torch.softmax(pe, 1)
        R = r.loi().detach()
        rew = (poids.unsqueeze(1) * s * R.t()).sum()
        ent = -(s * torch.log(s)).sum() / N
        return -(rew + BETA * ent)
    Hauto = torch.autograd.functional.hessian(f5, p_e[5].detach().clone())
    Han = hess_negent(p_e[5].detach())
    print(f"analytic vs autograd (full -J, row 5): max abs diff = {(Han - Hauto).abs().max().item():.3e}, "
          f"max abs entry = {Han.abs().max().item():.3e}")
    h = BETA / (N * N)
    rec = {k: [] for k in ("t", "def5", "sv5", "S5max", "S5mean", "S5min_sv", "z5", "g5", "m5",
                           "pm8", "sv8", "S8max", "pm12", "sv12", "S12max", "z8", "z12")}
    for t in range(T):
        with torch.no_grad():
            st = opt.state[p_e]
            V = st["exp_avg_sq"]
            for i in ROWS:
                z = p_e[i].detach()
                P = V[i].sqrt() + ADAM_EPS
                H = hess_negent(z)
                Dm = P.rsqrt()
                lam = torch.linalg.eigvalsh(Dm[:, None] * H * Dm[None, :]).max().item()
                if i == 5:
                    rec["S5max"].append(LR * lam)
                    rec["S5mean"].append(LR * h / (V[5].sqrt().mean().item() + ADAM_EPS))
                    rec["S5min_sv"].append(LR * h / (V[5].sqrt().min().item() + ADAM_EPS))
                    rec["sv5"].append(V[5].sqrt().clone())
                    s = torch.softmax(z, 0)
                    rec["def5"].append(math.log(N) + (s * s.log()).sum().item())
                    rec["z5"].append((z - z.mean()).clone())
                    rec["m5"].append(st["exp_avg"][5].clone())
                elif i == 8:
                    rec["S8max"].append(LR * lam)
                    s = torch.softmax(z, 0)
                    rec["pm8"].append((s[23] / (s[23] + s[19]) - 0.5).item())
                    rec["sv8"].append(((V[8, 23].sqrt() + V[8, 19].sqrt()) / 2).item())
                    rec["z8"].append((z[23] - z[19]).item())
                else:
                    rec["S12max"].append(LR * lam)
                    s = torch.softmax(z, 0)
                    rec["pm12"].append((s[1] / (s[1] + s[21]) - 0.5).item())
                    rec["sv12"].append(((V[12, 1].sqrt() + V[12, 21].sqrt()) / 2).item())
                    rec["z12"].append((z[1] - z[21]).item())
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        rec["g5"].append(p_e.grad[5].detach().clone())
        opt.step()
        rec["t"].append(54000 + t)
    out = {k: (torch.stack(v) if isinstance(v[0], torch.Tensor) else torch.tensor(v, dtype=torch.float64))
           for k, v in rec.items()}
    torch.save(out, "D:/tmp/agent_t58_trace_rows_5_8_12.pt")
    for k in ("S5max", "S5mean", "S5min_sv", "S8max", "S12max"):
        x = out[k]
        print(f"{k:9s} mean {x.mean().item():.3f}  median {x.median().item():.3f}  min {x.min().item():.3f}  "
              f"max {x.max().item():.3f}  frac>38 {(x > 38).double().mean().item():.3f}")
    print(f"<def5> = {out['def5'].mean().item():.4e}")


if __name__ == "__main__":
    main(int(sys.argv[1]))
