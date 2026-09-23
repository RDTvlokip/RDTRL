"""agent (dipankar role), 23/09 : analysis of D:/tmp/agent_t58_trace_rows_5_8_12.pt
(20000 steps from the mur23 checkpoint, delta real)."""
import math
import torch

LR, EPS, BETA, N = 0.05, 1e-10, 0.02, 27
h = BETA / (N * N)
hu = BETA / (2 * N)
d = torch.load("D:/tmp/agent_t58_trace_rows_5_8_12.pt")
T = len(d["t"])
print("T =", T)

# (d) window artefact : block means of the deficit and of mean sqrt v, row 5
def5 = d["def5"]
sv5m = d["sv5"].mean(1)
v5m = (d["sv5"] ** 2).mean(1)
pred_def = 0.5 * (LR / 38 - EPS / h) ** 2
print(f"row 5 : <def> over all = {def5.mean():.4e}  pred {pred_def:.4e}  ratio {def5.mean()/pred_def:.4f}")
print(f"        identity check 1/2 <v>/h^2 = {0.5*v5m.mean()/h**2:.4e}   1/2 <sqrt v>^2/h^2 = {0.5*sv5m.mean()**2/h**2:.4e}")
for W in (200, 500, 1000, 2000, 4000, 5000):
    nb = T // W
    bm = def5[: nb * W].reshape(nb, W).mean(1) / pred_def
    bs = sv5m[: nb * W].reshape(nb, W).mean(1) / (LR * h / 38 - EPS)
    print(f"  window {W:5d}: {nb:3d} blocks  <def>/pred min {bm.min():.4f} max {bm.max():.4f} sd {bm.std():.4f} | "
          f"<sqrt v>/pred min {bs.min():.4f} max {bs.max():.4f} sd {bs.std():.4f}")
# integrated autocorrelation time of def5
x = def5 - def5.mean()
var = (x * x).mean()
tau = 1.0
for lag in range(1, 3000):
    c = (x[:-lag] * x[lag:]).mean() / var
    if c < 0.02:
        break
    tau += 2 * c
print(f"  integrated autocorr time of def5 ~ {tau:.1f} steps (cut at lag {lag}); "
      f"s.e. of the 20000-step mean ~ {math.sqrt(var*tau/T)/pred_def*100:.2f} % of pred")
print(f"  instantaneous def5 / pred : min {def5.min()/pred_def:.3f} max {def5.max()/pred_def:.3f}")

# per-coordinate spread of sqrt v, and where the top eigenvalue sits
sv = d["sv5"]
print(f"row 5 sqrt v spread : mean over time of (max/min) = {(sv.max(1).values/sv.min(1).values).mean():.4f}, "
      f"mean/min = {(sv.mean(1)/sv.min(1).values).mean():.4f}")
S5max, S5min = d["S5max"], d["S5min_sv"]
print(f"S5max / S(min sqrt v) = {(S5max/S5min).mean():.5f}   (h exact -> 1 ; h_diag -> {26/27:.4f})")

# count of coordinates in 'burst' : per-step |dz| alternating and above a level
z5 = d["z5"]
dz = z5[1:] - z5[:-1]
alt = (dz[1:] * dz[:-1] < 0)
big = dz[1:].abs() > 2e-4
nburst = (alt & big).sum(1).double()
print(f"coords simultaneously with alternating |dz|>2e-4 : mean {nburst.mean():.2f}, frac of steps with >=1 : {(nburst>=1).double().mean():.3f}")

# synonyms : burst detection and cycle statistics
for name in ("8", "12"):
    pm = d["pm" + name]
    svs = d["sv" + name]
    S = d["S" + name + "max"]
    rms = pm.pow(2).mean().sqrt().item()
    print(f"ref {name}: (p-1/2)_rms over {T} = {rms:.4e}  pred lr/76 = {LR/76:.4e}  ratio {rms/(LR/76):.4f}  <S> = {S.mean():.3f}  "
          f"<sqrt v> = {svs.mean():.4e}  pred {LR*hu/38-EPS:.4e}")
    for W in (2000, 4000, 5000):
        nb = T // W
        bm = pm[: nb * W].reshape(nb, W).pow(2).mean(1).sqrt() / (LR / 76)
        print(f"   window {W}: rms/pred min {bm.min():.3f} max {bm.max():.3f}")
    # bursts : S crosses 38 upward
    up = ((S[1:] > 38) & (S[:-1] <= 38)).nonzero().flatten() + 1
    per = (up[1:] - up[:-1]).double()
    print(f"   upward crossings of 38 : {len(up)}  period mean {per.mean():.1f} sd {per.std():.1f}")
    # per cycle : S peak, S min after, log amplitude range
    lz = pm.abs().clamp_min(1e-300).log10()
    rows = []
    for k in range(len(up) - 1):
        a, b = up[k].item(), up[k + 1].item()
        seg = S[a:b]
        lzs = lz[a:b]
        rows.append((seg.max().item(), seg.min().item(), lzs.max().item(), lzs.min().item(),
                     (seg > 38).sum().item(), b - a))
    rows = torch.tensor(rows)
    print(f"   per cycle (mean): S_peak {rows[:,0].mean():.3f}  S_min {rows[:,1].mean():.3f}  "
          f"log10|p-1/2| max {rows[:,2].mean():.2f} min {rows[:,3].mean():.2f}  steps above 38 {rows[:,4].mean():.1f}  length {rows[:,5].mean():.1f}")
    rho = rows[:, 0] / rows[:, 1]
    print(f"   rho = S_peak/S_min mean {rho.mean():.4f} ; 38*ln(rho)/(rho-1) = {(38*rho.log()/(rho-1)).mean():.3f}")
