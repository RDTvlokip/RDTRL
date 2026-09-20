"""Diagnostic: is the meanfield_fold.py 'fold' for K=26 a genuine
saddle-node (Jacobian determinant -> 0, two solution branches merging)
or a numerical artifact of fixed-step continuation (Newton fails from a
poor warm start on a steeply-bending but still single-valued branch)?

Also checks bistability directly: does a solve started from a
COLLAPSED initial guess (s3 small) find a second solution coexisting
with the graded one, at deltas below the fold?
"""
import numpy as np

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


def newton_solve(x0, delta, K, tol=1e-13, maxit=300):
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
        # damped Newton to help convergence near sharp bends
        step = 1.0
        x_new = x - step * dx
        x_new = np.clip(x_new, 1e-14, 1 - 1e-14)
        x = x_new
    return x, np.max(np.abs(residual(x, delta, K))) < 1e-8


K = 26
print("=== det(J) along the graded branch as delta -> fold (K=26) ===")
x = np.array([0.999999, 1e-6, 1e-6])
delta = 0.0005
ddelta = 0.00002
prev_x = None
x, ok = newton_solve([0.999999, 0.5, 0.5], delta, K)
prev_x = x.copy()
while delta < 0.0145:
    delta_next = delta + ddelta
    x_new, ok = newton_solve(prev_x, delta_next, K)
    J = jacobian_fd(x_new, delta_next, K)
    detJ = np.linalg.det(J)
    if not ok or abs(x_new[0] - prev_x[0]) > 0.2:
        print(f"  delta={delta_next:.6f}  NEWTON FAILED/JUMP  prev_s3={prev_x[0]:.6f}")
        break
    if delta_next > 0.0133:
        print(f"  delta={delta_next:.6f}  s3={x_new[0]:.6f}  r3={1-x_new[2]:.6f}  det(J)={detJ:.6e}")
    prev_x = x_new
    delta = delta_next

print()
print("=== bistability check: solve from a COLLAPSED guess at deltas below the fold ===")
for delta in (0.005, 0.010, 0.012, 0.0133, 0.0134):
    x_collapsed, ok = newton_solve([1.0 / 27, 0.9, 0.9], delta, K)
    x_graded, ok2 = newton_solve([0.999999, 1e-6, 1e-6], delta, K)
    print(f"  delta={delta:<8} collapsed-start -> s3={x_collapsed[0]:.6f} r3={1-x_collapsed[2]:.6f} ok={ok}"
          f"   |  graded-start -> s3={x_graded[0]:.6f} ok={ok2}")
