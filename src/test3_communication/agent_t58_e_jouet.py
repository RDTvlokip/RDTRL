"""agent (dipankar role), 23/09 : pure-quadratic toys under Adam(0.9, 0.999),
exactly PyTorch's update, float64, to see which numbers of the orphan and
of the synonyms are universal constants of Adam on a quadratic.

  1d      : f = lam/2 x^2 (synonym split; eps ~ 0 -> scale free)
  1d_off  : same with x stored as 25 + x (rounding floor of a real logit)
  orphan  : f = h/2 sum (z_j - zbar)^2, K = 27 (coupled through the mean)
  diag    : f = h(1-1/K)/2 sum z_j^2 (same diagonal, NO mean coupling)

Usage: python agent_t58_e_jouet.py <mode> [steps] [eps] [noise]
"""
import sys
import math
import numpy as np

LR, B1, B2 = 0.05, 0.9, 0.999
import os
BETA, N = 0.02, 27
K = int(os.environ.get("AGENT_K", "27"))


def run(mode, T, eps, noise, seed=1):
    rng = np.random.default_rng(seed)
    if mode.startswith("1d"):
        lam = BETA / (2 * N)
        dim = 1
    else:
        lam = BETA / (N * K)
        dim = K
    off = 25.0 if mode == "1d_off" else 0.0
    x = off + 1e-3 * rng.standard_normal(dim)
    m = np.zeros(dim)
    v = np.zeros(dim)
    rec_S, rec_x2, rec_sv, rec_x = [], [], [], []
    for t in range(1, T + 1):
        y = x - off
        if mode.startswith("1d"):
            g = lam * y
        elif mode == "orphan":
            g = lam * (y - y.mean())
        else:
            g = lam * (1 - 1 / K) * y
        if noise > 0:
            g = g + noise * rng.standard_normal(dim)
        m = B1 * m + (1 - B1) * g
        v = B2 * v + (1 - B2) * g * g
        bc1 = 1 - B1 ** t
        bc2 = 1 - B2 ** t
        denom = np.sqrt(v) / math.sqrt(bc2) + eps
        x = x - (LR / bc1) * m / denom
        if t > T // 2:
            P = np.sqrt(v) + eps
            if mode.startswith("1d"):
                S = LR * lam / P[0]
                xs = y[0]
            elif mode == "orphan":
                Dm = 1 / np.sqrt(P)
                H = lam * (np.eye(K) - np.ones((K, K)) / K)
                S = LR * np.linalg.eigvalsh(Dm[:, None] * H * Dm[None, :]).max()
                xs = y - y.mean()
            else:
                S = LR * lam * (1 - 1 / K) / P.min()
                xs = y
            rec_S.append(S)
            rec_x2.append(np.mean(np.atleast_1d(xs) ** 2))
            rec_sv.append(np.sqrt(v).mean())
            rec_x.append(np.atleast_1d(xs)[0])
    S = np.array(rec_S)
    x2 = np.array(rec_x2)
    sv = np.array(rec_sv)
    xs = np.array(rec_x)
    if mode.startswith("1d"):
        lam_eff = lam
    elif mode == "orphan":
        lam_eff = lam
    else:
        lam_eff = lam * (1 - 1 / K)
    rms_ratio = math.sqrt(x2.mean()) / (LR / 38 - eps / lam_eff)
    sv_ratio = sv.mean() / (LR * lam_eff / 38 - eps)
    s_eff = LR * lam_eff / (sv.mean() + eps)
    up = np.nonzero((S[1:] > 38) & (S[:-1] <= 38))[0] + 1
    per = np.diff(up).mean() if len(up) > 2 else float("nan")
    ax = np.abs(xs[xs != 0])
    print(f"{mode:7s} eps={eps:.0e} noise={noise:.0e}: rms/(lr/38-eps/lam) = {rms_ratio:.4f}  "
          f"<sqrt v>/pred = {sv_ratio:.4f}  S_eff(mean sqrt v) = {s_eff:.3f}  <S_max> = {S.mean():.3f}  "
          f"frac S>38 = {(S > 38).mean():.3f}  S range [{S.min():.2f},{S.max():.2f}]  "
          f"period {per:.1f}  log10|x| range [{np.log10(ax.min()):.2f},{np.log10(ax.max()):.2f}]")


if __name__ == "__main__":
    mode = sys.argv[1]
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
    eps = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-10
    noise = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    run(mode, T, eps, noise)
