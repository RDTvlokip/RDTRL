"""agent (dipankar role), 23/09 : is the receiver 'kick' of the mur 23 (period
~460-480 steps, open questions: period drops at lr=0.2/0.3, non-monotone in
beta2) just the universal relaxation oscillation of Adam on ONE isolated mode?

Toy = two receiver logits (rho3, rho4) of message 10 only, per-coordinate Adam
exactly as PyTorch, float64, with the EXACT 1-D objective along the gap
  J(y) = (1/N)[(1-d) s3 (1-sig(y)) + (1+d) s4 sig(y)] + (beta/N) H2(sig(y)),
y = rho4 - rho3, s3, s4 frozen at their mur23 values (delta real), or the
pure quadratic with the same curvature ('quad'), or the symmetric
delta=0 case (s3=s4=1, d=0).
Period = median spacing between upward crossings of S = lr*lam/(sqrt v+eps)
through 38 (lam = local curvature per coordinate).
Usage: python agent_t58_h_jouet_kick.py <case> <lr> [beta2] [steps]
case in {real, quad, sym}
"""
import sys
import math
import numpy as np

N, BETA, B1 = 27, 0.02, 0.9
EPS = 1e-10
RNG = np.random.default_rng(0)


def sig(y):
    return 1.0 / (1.0 + math.exp(-y))


def main(case, lr, b2, T):
    if case == "sym":
        d, s3, s4 = 0.0, 1.0, 1.0
    else:
        d, s3, s4 = 0.013026615, 0.9989630570, 0.99998
    a3, a4 = (1 - d) * s3 / N, (1 + d) * s4 / N
    ystar = (a4 - a3) / (BETA / N)
    # stationary point of J(y): dJ/dy = sig'(y)[(a4 - a3) - (beta/N) y] = 0  ->  y* = (a4-a3) N / beta
    p = sig(ystar)
    lam_y = (BETA / N) * p * (1 - p)          # curvature of -J in y at y*
    # per coordinate: rho4 = +y/2, rho3 = -y/2 ; g4 = -dJ/dy, g3 = +dJ/dy ; curvature per coord = 2 lam_y
    lam = 2 * lam_y

    def grad_y(y):
        if case == "quad":
            return lam_y * (y - ystar)       # d(-J)/dy
        q = sig(y)
        return -(q * (1 - q)) * ((a4 - a3) - (BETA / N) * y)

    x = np.array([-ystar / 2 + 1e-4, ystar / 2])   # rho3, rho4
    m = np.zeros(2)
    v = np.zeros(2)
    ups, Ss, amp = [], [], []
    S_prev = 0.0
    for t in range(1, T + 1):
        y = x[1] - x[0]
        gy = grad_y(y)
        g = np.array([-gy, gy]) + 1e-20 * RNG.standard_normal(2)
        m = B1 * m + (1 - B1) * g
        v = b2 * v + (1 - b2) * g * g
        bc1, bc2 = 1 - B1 ** t, 1 - b2 ** t
        x = x - (lr / bc1) * m / (np.sqrt(v) / math.sqrt(bc2) + EPS)
        S = lr * lam / (math.sqrt(v[1]) + EPS)
        if t > T // 3:
            if S > 38 and S_prev <= 38:
                ups.append(t)
            Ss.append(S)
            amp.append(abs(y - ystar))
        S_prev = S
    per = np.diff(ups)
    amp = np.array(amp)
    print(f"case={case:4s} lr={lr:<5} beta2={b2}: y*={ystar:.4f} p*={p:.4f}  period median {np.median(per):.1f} "
          f"(n={len(per)})  <S> {np.mean(Ss):.2f}  S range [{min(Ss):.2f},{max(Ss):.2f}]  "
          f"max|y-y*| {amp.max():.3e}  max|y-y*|/lr {amp.max()/lr:.4f}")


if __name__ == "__main__":
    case = sys.argv[1]
    lr = float(sys.argv[2])
    b2 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.999
    T = int(sys.argv[4]) if len(sys.argv) > 4 else 60000
    main(case, lr, b2, T)
