"""Independent equilibrium (mean-field) cross-check of the K-variable toy's
delta_c(K), NOT using Adam dynamics -- solves the stationary equations
directly and looks for the turning point (fold) of the graded branch as
delta increases, via continuation with a Newton corrector.

This is a DIFFERENT quantity than the Adam-measured delta_c (which is a
finite-step, fixed-initial-condition dynamical threshold -- possibly a
boundary crisis rather than a fold, per CARNET.md 7.63: "no curvature, no
critical slowing" was found for the real system). Used only to check
whether the equilibrium theory's K-sensitivity is broadly consistent with
the Adam-measured numbers, and to get delta_c(K) cheaply for many K.
"""
import numpy as np
from scipy.optimize import brentq

BETA = 0.02


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def residual(x, delta, K):
    s3, s4, r4 = x
    r3 = 1 - r4
    e1 = r4 - sigmoid((1.0 / BETA) * ((1 + delta) * s4 - (1 - delta) * s3))
    e2 = s3 - 1.0 / (1.0 + K * np.exp(-(1 - delta) * r3 / BETA))
    e3 = s4 - 1.0 / (1.0 + K * np.exp(-(1 + delta) * r4 / BETA))
    return np.array([e1, e2, e3])


def jacobian_fd(x, delta, K, eps=1e-7):
    J = np.zeros((3, 3))
    f0 = residual(x, delta, K)
    for i in range(3):
        xp = x.copy()
        xp[i] += eps
        J[:, i] = (residual(xp, delta, K) - f0) / eps
    return J


def newton_solve(x0, delta, K, tol=1e-13, maxit=200):
    x = np.array(x0, dtype=float)
    for _ in range(maxit):
        f = residual(x, delta, K)
        if np.max(np.abs(f)) < tol:
            return x, True
        J = jacobian_fd(x, delta, K)
        try:
            dx = np.linalg.solve(J, f)
        except np.linalg.LinAlgError:
            return x, False
        x = x - dx
        x[0] = min(max(x[0], 1e-12), 1 - 1e-12)
        x[1] = min(max(x[1], 1e-12), 1 - 1e-12)
        x[2] = min(max(x[2], 1e-12), 1 - 1e-12)
    return x, np.max(np.abs(residual(x, delta, K))) < 1e-8


def trace_graded_branch(K, delta_max=0.05, ddelta=0.0002):
    """Continuation along the graded branch (s3 near 1) starting at
    delta=0. Returns (deltas, s3s, folded_at) where folded_at is the delta
    where Newton fails to converge near the previous point (fold /
    branch disappearance), or None if it survives to delta_max."""
    x = np.array([1 - 1e-6, 1e-6, 1e-6])
    # warm up to delta=0 exactly (symmetric point s3=s4 isn't graded;
    # start at a tiny positive delta so the graded branch is well defined)
    delta = ddelta
    x, ok = newton_solve([0.999999, 0.5, 0.5], delta, K)
    deltas = [delta]
    s3s = [x[0]]
    prev_x = x.copy()
    while delta < delta_max:
        delta_next = delta + ddelta
        x_guess = prev_x  # warm start
        x_new, ok = newton_solve(x_guess, delta_next, K)
        # detect fold: residual explodes, or s3 jumps by a large amount
        if not ok or abs(x_new[0] - prev_x[0]) > 0.05:
            return deltas, s3s, delta_next
        deltas.append(delta_next)
        s3s.append(x_new[0])
        prev_x = x_new
        delta = delta_next
    return deltas, s3s, None


if __name__ == "__main__":
    print("=== fine mesh around K=20,26,30 (ddelta=1e-5) ===")
    for K in (20, 26, 30):
        deltas, s3s, fold = trace_graded_branch(K, delta_max=0.06, ddelta=0.00001)
        print(f"K={K:3d}  last graded delta={deltas[-1]:.6f}  s3={s3s[-1]:.6f}  fold~={fold}")
    print("=== K=1, extended delta_max ===")
    deltas, s3s, fold = trace_graded_branch(1, delta_max=0.5, ddelta=0.001)
    print(f"K=1  last graded delta={deltas[-1]:.6f}  s3={s3s[-1]:.6f}  fold~={fold}")
    print("=== K=8, cross-check against Adam ===")
    deltas, s3s, fold = trace_graded_branch(8, delta_max=0.06, ddelta=0.00001)
    print(f"K=8  last graded delta={deltas[-1]:.6f}  s3={s3s[-1]:.6f}  fold~={fold}")
