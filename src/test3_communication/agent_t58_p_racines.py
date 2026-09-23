"""agent (dipankar role), 23/09 : closed forms of Adam's frozen-v linear map on a
quadratic mode, S = lr * lambda / (sqrt v + eps) :
  mu^2 + ((1-b1) S - (1+b1)) mu + b1 = 0
threshold mu=-1 at S = 2(1+b1)/(1-b1) ; double root at S = (1 +- sqrt b1)^2/(1-b1) ;
|mu| = sqrt(b1) exactly in between. Plus the quiet-phase bookkeeping for the
synonym cycle (S_min, S_peak measured on the 20000-step trace)."""
import numpy as np
b1, b2 = 0.9, 0.999
for S in (1, 10, 30, 37.9, 37.97, 37.974, 38, 38.1, 38.43, 39, 39.6):
    r = np.roots([1, (1 - b1) * S - (1 + b1), b1])
    print(f"S={S:7.3f} |mu|max={abs(r).max():.5f} ln|mu|={np.log(abs(r).max()):+.4f}")
print("double roots at S =", (1 + np.sqrt(b1)) ** 2 / (1 - b1), (1 - np.sqrt(b1)) ** 2 / (1 - b1),
      "; threshold", 2 * (1 + b1) / (1 - b1))
print("quiet decay per step ln(1/sqrt b1) =", 0.5 * np.log(1 / b1), "e-folds =", 0.5 * np.log10(1 / b1), "decades")
g = np.log(1 / b1) / (-np.log(b2))
print("gain ln(1/b1)/(-ln b2) =", g)
Smin, Speak = 31.94, 39.57
print("t_q =", 2 * np.log(38 / Smin) / (-np.log(b2)), " decay e-folds =", g * np.log(38 / Smin),
      " decades =", g * np.log(38 / Smin) / np.log(10))
print("growth time 38 -> S_peak =", 2 * np.log(Speak / 38) / (-np.log(b2)),
      " ; rise time S_min -> S_peak =", 2 * np.log(Speak / Smin) / (-np.log(b2)))
rho = Speak / Smin
print("S_peak ln(rho)/(rho-1) =", Speak * np.log(rho) / (rho - 1))
