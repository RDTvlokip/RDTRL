"""agent (dipankar role), 23/09 : closed form check. For 0.0263 < S < 37.974 the
frozen-v Adam map has complex roots of modulus sqrt(beta1) exactly, so in the
quiet phase every mode decays at ln(1/sqrt 0.9) = 0.05268 e-folds/step
(0.022879 decades/step), whatever S. Fit on the synonym split of rows 8/12
(20000-step trace), steps with S < 37.5 and at least 30 steps after the
burst peak."""
import torch
d = torch.load("D:/tmp/agent_t58_trace_rows_5_8_12.pt")
for name in ("8", "12"):
    x = d["z" + name]
    S = d["S" + name + "max"]
    env = torch.maximum(x[1:].abs(), x[:-1].abs())
    le = env.clamp_min(1e-300).log10()
    Ss = S[1:]
    # quiet segments: S < 37.5 continuously ; fit slope on each segment after skipping 30 steps
    slopes = []
    t = 0
    T = len(Ss)
    while t < T:
        if Ss[t] < 37.5:
            u = t
            while u < T and Ss[u] < 37.5:
                u += 1
            a, b = t + 30, u
            if b - a > 60:
                tt = torch.arange(a, b, dtype=torch.float64)
                yy = le[a:b]
                A = torch.stack([tt, torch.ones_like(tt)], 1)
                coef = torch.linalg.lstsq(A, yy.unsqueeze(1)).solution.flatten()
                slopes.append(coef[0].item())
            t = u
        else:
            t += 1
    s = torch.tensor(slopes)
    print(f"ref {name}: {len(s)} quiet segments, decay slope (decades/step) median {s.median().item():.6f} "
          f"[{s.quantile(0.1).item():.6f}, {s.quantile(0.9).item():.6f}]   closed form -0.022879")
