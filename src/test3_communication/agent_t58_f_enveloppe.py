"""agent (dipankar role), 23/09 : is the orphan ever quiet? envelope statistics
of row 5 coordinates vs the synonym split, from agent_t58_trace_rows_5_8_12.pt."""
import torch
LR = 0.05
A = LR / 38
d = torch.load("D:/tmp/agent_t58_trace_rows_5_8_12.pt")
z = d["z5"]                                   # (T, 27), z - zbar
env = torch.maximum(z[1:].abs(), z[:-1].abs())
for thr in (1.0, 0.3, 0.1, 0.01, 1e-3):
    print(f"row 5 : frac (t,j) with envelope < {thr:g} lr/38 : {(env < thr*A).double().mean():.4f}   "
          f"frac of steps where ALL 27 < {thr:g} lr/38 : {(env < thr*A).all(1).double().mean():.4f}")
print(f"row 5 : min over (t,j) of envelope / (lr/38) = {(env.min()/A).item():.2e};  "
      f"per-coordinate min: median {(env.min(0).values/A).median().item():.2e}")
rms_t = z.pow(2).mean(1).sqrt()
print(f"row 5 : row-rms(t)/(lr/38) min {(rms_t.min()/A).item():.3f}  max {(rms_t.max()/A).item():.3f}")
for name, key in (("8", "z8"), ("12", "z12")):
    x = d[key] / 2                               # per-coordinate displacement = d/2
    e = torch.maximum(x[1:].abs(), x[:-1].abs())
    for thr in (1.0, 0.1, 0.01, 1e-3):
        print(f"ref {name}: frac steps with envelope < {thr:g} lr/38 : {(e < thr*A).double().mean():.4f}")
    print(f"ref {name}: min envelope/(lr/38) = {(e.min()/A).item():.2e}")
