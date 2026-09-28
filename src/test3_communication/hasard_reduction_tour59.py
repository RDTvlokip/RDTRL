"""Hazard-rate sampling in the reduced model.  Protocol P-A (mirror of the full-network grid): reduction started from the
full-network phase-1 state (20 000 Adam steps at pli-1e-6), then 20 000 + k reduced steps at pli-1e-6 (k random), a random
relative perturbation of size PERT on the 6 coordinates, then delta = pli + off, up to T steps, escape = R_b > 0.9.
After a crossing we continue POST steps and record R_b at the end (irreversibility check)."""
import sys, json, time
sys.path.insert(0, ".")
import numpy as np
import torch
import reduction_numba_tour59 as redlib  # port numba de reduction_deux_lignes_tour59.py (agent style dipankar, 29/09/2026)

PHASE1 = "D:/tmp/rdtrl_tour59_phase1_mur23_3_4_20000.pt"
POST = 3000


def base_state():
    c = torch.load(PHASE1)
    st = c["opt"]["state"]
    s = redlib.reduire(c["params"], [st[0]["exp_avg"], st[1]["exp_avg"]], [st[0]["exp_avg_sq"], st[1]["exp_avg_sq"]],
                       float(st[0]["step"]))
    return s


def warmed(s1, k, rng, pert):
    w = redlib.copie(s1)
    redlib.avancer(w, redlib.PLI - 1e-6, 20000 + k, cross=2.0, stop=False)
    if pert > 0:
        w["x"] = w["x"] * (1.0 + pert * rng.standard_normal(6))
    return w


def replicas(off, n_rep, T, seed, kmax=4444, pert=1e-9, eps=1e-10, cross=0.9, s1=None, post=POST):
    rng = np.random.default_rng(seed)
    s1 = s1 or base_state()
    out = []
    for j in range(n_rep):
        k = int(rng.integers(0, kmax))
        w = warmed(s1, k, rng, pert)
        r = redlib.avancer(w, redlib.PLI + off, T, eps=eps, cross=cross, stop=True)
        first, maxrb, rb, done = r
        fin = -1.0
        if first >= 0:
            r2 = redlib.avancer(w, redlib.PLI + off, post, eps=eps, cross=2.0, stop=False)
            fin = r2[2]
        out.append((k, first, done, fin))
    return out


def resume(events, expo):
    return events, expo


if __name__ == "__main__":
    # usage: hazard.py tag n_rep T seed off1 off2 ...   (offsets in units of 1e-9)
    tag, n_rep, T, seed = sys.argv[1], int(sys.argv[2]), int(float(sys.argv[3])), int(sys.argv[4])
    offs = [float(x) for x in sys.argv[5:]]
    s1 = base_state()
    redlib.avancer(redlib.copie(s1), 0.0, 1)
    for o in offs:
        t0 = time.time()
        res = replicas(o * 1e-9, n_rep, T, seed, s1=s1)
        ev = [(k, f, d, fin) for (k, f, d, fin) in res if f >= 0]
        expo = sum((f + 1) if f >= 0 else d for (k, f, d, fin) in res)
        lam = len(ev) / expo if expo else float("nan")
        print(f"off={o:g}e-9 n_rep={n_rep} T={T} events={len(ev)} exposure={expo} lambda={lam:.3e} "
              f"times={[f for (_, f, _, _) in ev][:12]} fin={[round(x, 4) for (_, _, _, x) in ev][:12]} wall={time.time()-t0:.0f}s",
              flush=True)
        with open(f"D:/tmp/rdtrl_t59_hazred_{tag}_{o:g}.json", "w") as fh:
            json.dump(dict(off=o, n_rep=n_rep, T=T, seed=seed, res=res), fh)
