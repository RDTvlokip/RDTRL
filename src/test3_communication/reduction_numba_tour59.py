"""Reduced Adam simulator (numba) for tour-59 falaise hazard. Same equations as D:/tmp/agent_dk60/red2.py.
State vector x[0:6] = (e3,o3,e4,o4,l3,l4), m[0:6], v[0:6].  Coordinates: sender row-3 logit of message 10 (e3) and mean of
its 26 competitors (o3), same for row 4 (e4,o4), receiver logits l3,l4 of row 10.  Tail (25 receiver refs) frozen
relative to l4 (Se, Sed)."""
import math
import numpy as np
import torch
import numba as nb

B, C, LR, N = 0.02, 26.0, 0.05, 27.0
PLI = 0.0134372100660973
B1, B2 = 0.9, 0.999
NAMES = ["e3", "o3", "e4", "o4", "l3", "l4"]


def _sep(mat, i):
    return float(mat[i, 10]), float(np.delete(mat[i].numpy(), 10).mean())


def reduire(params, m_list, v_list, t):
    """params=(E,R) 27x27 tensors, m_list/v_list = [Em,Rm]/[Ev,Rv] tensors."""
    e, r = params
    x = np.zeros(6); m = np.zeros(6); v = np.zeros(6)
    for j, i in ((0, 3), (2, 4)):
        x[j], x[j + 1] = _sep(e, i)
        m[j], m[j + 1] = _sep(m_list[0], i)
        v[j], v[j + 1] = _sep(v_list[0], i)
    x[4], x[5] = float(r[10, 3]), float(r[10, 4])
    m[4], m[5] = float(m_list[1][10, 3]), float(m_list[1][10, 4])
    v[4], v[5] = float(v_list[1][10, 3]), float(v_list[1][10, 4])
    tail = np.delete(r[10].numpy().astype(float), [3, 4]) - x[5]
    et = np.exp(tail)
    return dict(x=x, m=m, v=v, Se=float(et.sum()), Sed=float((et * tail).sum()), t=float(t))


def charger_chaud(fichier):
    c = torch.load(fichier)
    st = c["opt"]["state"]
    return reduire(c["params"], [st[0]["exp_avg"], st[1]["exp_avg"]], [st[0]["exp_avg_sq"], st[1]["exp_avg_sq"]],
                   float(st[0]["step"]))


def charger_masque(fichier):
    c = torch.load(fichier)
    return reduire(c["params"], [c["m"][0], c["m"][1]], [c["v"][0], c["v"][1]], float(c["t"]))


@nb.njit(cache=True, fastmath=False)
def run(x, m, v, Se, Sed, t0, delta, eps, nsteps, cross, stop_at_cross, rec):
    """Advance nsteps (t >= 40000 assumed: bias corrections = 1).  Returns (first_cross_step or -1, maxRb, last Rb, steps done).
    x,m,v modified in place.  rec: array (nrec,) receives Rb every step for the first nrec steps (for diagnostics)."""
    first = -1
    maxrb = 0.0
    rb = 0.0
    e3 = x[0]; o3 = x[1]; e4 = x[2]; o4 = x[3]; l3 = x[4]; l4 = x[5]
    g = np.empty(6)
    nrec = rec.shape[0]
    done = 0
    for i in range(nsteps):
        z3 = e3 - o3
        z4 = e4 - o4
        ex3 = C * math.exp(-z3)
        s3 = 1.0 / (1.0 + ex3)
        d3 = ex3 * s3
        ex4 = C * math.exp(-z4)
        d4 = ex4 / (1.0 + ex4)
        s4 = 1.0 - d4
        u = l3 - l4
        w3 = math.exp(u)
        Z = w3 + 1.0 + Se
        r3 = w3 / Z
        r4 = 1.0 / Z
        logZ = math.log(Z)
        Hr = -(r3 * (u - logZ) + r4 * (-logZ) + (Sed - logZ * Se) / Z)
        a3 = (1.0 - delta) * s3
        a4 = (1.0 + delta) * s4
        abar = r3 * a3 + r4 * a4
        g[4] = r3 * ((a3 - abar) - B * ((u - logZ) + Hr)) / N
        g[5] = r4 * ((a4 - abar) - B * ((-logZ) + Hr)) / N
        X3 = (1.0 - delta) * r3 / B
        X4 = (1.0 + delta) * r4 / B
        ge3 = s3 * d3 * B * (X3 - z3) / N
        ge4 = s4 * d4 * B * (X4 - z4) / N
        g[0] = ge3
        g[1] = -ge3 / C
        g[2] = ge4
        g[3] = -ge4 / C
        for k in range(6):
            m[k] = B1 * m[k] + (1.0 - B1) * g[k]
            v[k] = B2 * v[k] + (1.0 - B2) * g[k] * g[k]
        tt = t0 + i + 1.0
        if tt < 60000.0:
            bc1 = 1.0 - B1 ** tt
            sq = math.sqrt(1.0 - B2 ** tt)
        else:
            bc1 = 1.0
            sq = 1.0
        e3 += LR * (m[0] / bc1) / (math.sqrt(v[0]) / sq + eps[0])
        o3 += LR * (m[1] / bc1) / (math.sqrt(v[1]) / sq + eps[1])
        e4 += LR * (m[2] / bc1) / (math.sqrt(v[2]) / sq + eps[2])
        o4 += LR * (m[3] / bc1) / (math.sqrt(v[3]) / sq + eps[3])
        l3 += LR * (m[4] / bc1) / (math.sqrt(v[4]) / sq + eps[4])
        l4 += LR * (m[5] / bc1) / (math.sqrt(v[5]) / sq + eps[5])
        uu = l3 - l4
        rr4 = 1.0 / (1.0 + math.exp(uu) + Se)
        rb = (1.0 - d4) * rr4
        if rb > maxrb:
            maxrb = rb
        if i < nrec:
            rec[i] = rb
        done = i + 1
        if first < 0 and rb > cross:
            first = i
            if stop_at_cross:
                break
    x[0] = e3; x[1] = o3; x[2] = e4; x[3] = o4; x[4] = l3; x[5] = l4
    return first, maxrb, rb, done




@nb.njit(cache=True, fastmath=False)
def run_obs(x, m, v, Se, Sed, t0, delta, eps, nsteps, cross, stop_at_cross, rec):
    """Advance nsteps (t >= 40000 assumed: bias corrections = 1).  Returns (first_cross_step or -1, maxRb, last Rb, steps done).
    x,m,v modified in place.  rec: array (nrec,) receives Rb every step for the first nrec steps (for diagnostics)."""
    first = -1
    maxrb = 0.0
    rb = 0.0
    e3 = x[0]; o3 = x[1]; e4 = x[2]; o4 = x[3]; l3 = x[4]; l4 = x[5]
    g = np.empty(6)
    nrec = rec.shape[0]
    done = 0
    for i in range(nsteps):
        z3 = e3 - o3
        z4 = e4 - o4
        ex3 = C * math.exp(-z3)
        s3 = 1.0 / (1.0 + ex3)
        d3 = ex3 * s3
        ex4 = C * math.exp(-z4)
        d4 = ex4 / (1.0 + ex4)
        s4 = 1.0 - d4
        u = l3 - l4
        w3 = math.exp(u)
        Z = w3 + 1.0 + Se
        r3 = w3 / Z
        r4 = 1.0 / Z
        logZ = math.log(Z)
        Hr = -(r3 * (u - logZ) + r4 * (-logZ) + (Sed - logZ * Se) / Z)
        a3 = (1.0 - delta) * s3
        a4 = (1.0 + delta) * s4
        abar = r3 * a3 + r4 * a4
        g[4] = r3 * ((a3 - abar) - B * ((u - logZ) + Hr)) / N
        g[5] = r4 * ((a4 - abar) - B * ((-logZ) + Hr)) / N
        X3 = (1.0 - delta) * r3 / B
        X4 = (1.0 + delta) * r4 / B
        ge3 = s3 * d3 * B * (X3 - z3) / N
        ge4 = s4 * d4 * B * (X4 - z4) / N
        g[0] = ge3
        g[1] = -ge3 / C
        g[2] = ge4
        g[3] = -ge4 / C
        for k in range(6):
            m[k] = B1 * m[k] + (1.0 - B1) * g[k]
            v[k] = B2 * v[k] + (1.0 - B2) * g[k] * g[k]
        tt = t0 + i + 1.0
        if tt < 60000.0:
            bc1 = 1.0 - B1 ** tt
            sq = math.sqrt(1.0 - B2 ** tt)
        else:
            bc1 = 1.0
            sq = 1.0
        e3 += LR * (m[0] / bc1) / (math.sqrt(v[0]) / sq + eps[0])
        o3 += LR * (m[1] / bc1) / (math.sqrt(v[1]) / sq + eps[1])
        e4 += LR * (m[2] / bc1) / (math.sqrt(v[2]) / sq + eps[2])
        o4 += LR * (m[3] / bc1) / (math.sqrt(v[3]) / sq + eps[3])
        l3 += LR * (m[4] / bc1) / (math.sqrt(v[4]) / sq + eps[4])
        l4 += LR * (m[5] / bc1) / (math.sqrt(v[5]) / sq + eps[5])
        uu = l3 - l4
        rr4 = 1.0 / (1.0 + math.exp(uu) + Se)
        rb = (1.0 - d4) * rr4
        if rb > maxrb:
            maxrb = rb
        if i < nrec:
            rec[i, 0] = rb
            rec[i, 1] = e3 - o3
            rec[i, 2] = l3 - l4
            rec[i, 3] = math.sqrt(v[0])
            rec[i, 4] = math.sqrt(v[4])
            rec[i, 5] = m[0]
            rec[i, 6] = m[4]
        done = i + 1
        if first < 0 and rb > cross:
            first = i
            if stop_at_cross:
                break
    x[0] = e3; x[1] = o3; x[2] = e4; x[3] = o4; x[4] = l3; x[5] = l4
    return first, maxrb, rb, done



@nb.njit(cache=True, fastmath=False)
def run_until_up(x, m, v, Se, Sed, t0, delta, eps, nsteps, level, prev_rb):
    """Advance nsteps (t >= 40000 assumed: bias corrections = 1).  Returns (first_cross_step or -1, maxRb, last Rb, steps done).
    x,m,v modified in place.  rec: array (nrec,) receives Rb every step for the first nrec steps (for diagnostics)."""
    first = -1
    maxrb = 0.0
    rb = 0.0
    e3 = x[0]; o3 = x[1]; e4 = x[2]; o4 = x[3]; l3 = x[4]; l4 = x[5]
    g = np.empty(6)
    cross = 2.0
    done = 0
    for i in range(nsteps):
        z3 = e3 - o3
        z4 = e4 - o4
        ex3 = C * math.exp(-z3)
        s3 = 1.0 / (1.0 + ex3)
        d3 = ex3 * s3
        ex4 = C * math.exp(-z4)
        d4 = ex4 / (1.0 + ex4)
        s4 = 1.0 - d4
        u = l3 - l4
        w3 = math.exp(u)
        Z = w3 + 1.0 + Se
        r3 = w3 / Z
        r4 = 1.0 / Z
        logZ = math.log(Z)
        Hr = -(r3 * (u - logZ) + r4 * (-logZ) + (Sed - logZ * Se) / Z)
        a3 = (1.0 - delta) * s3
        a4 = (1.0 + delta) * s4
        abar = r3 * a3 + r4 * a4
        g[4] = r3 * ((a3 - abar) - B * ((u - logZ) + Hr)) / N
        g[5] = r4 * ((a4 - abar) - B * ((-logZ) + Hr)) / N
        X3 = (1.0 - delta) * r3 / B
        X4 = (1.0 + delta) * r4 / B
        ge3 = s3 * d3 * B * (X3 - z3) / N
        ge4 = s4 * d4 * B * (X4 - z4) / N
        g[0] = ge3
        g[1] = -ge3 / C
        g[2] = ge4
        g[3] = -ge4 / C
        for k in range(6):
            m[k] = B1 * m[k] + (1.0 - B1) * g[k]
            v[k] = B2 * v[k] + (1.0 - B2) * g[k] * g[k]
        tt = t0 + i + 1.0
        if tt < 60000.0:
            bc1 = 1.0 - B1 ** tt
            sq = math.sqrt(1.0 - B2 ** tt)
        else:
            bc1 = 1.0
            sq = 1.0
        e3 += LR * (m[0] / bc1) / (math.sqrt(v[0]) / sq + eps[0])
        o3 += LR * (m[1] / bc1) / (math.sqrt(v[1]) / sq + eps[1])
        e4 += LR * (m[2] / bc1) / (math.sqrt(v[2]) / sq + eps[2])
        o4 += LR * (m[3] / bc1) / (math.sqrt(v[3]) / sq + eps[3])
        l3 += LR * (m[4] / bc1) / (math.sqrt(v[4]) / sq + eps[4])
        l4 += LR * (m[5] / bc1) / (math.sqrt(v[5]) / sq + eps[5])
        uu = l3 - l4
        rr4 = 1.0 / (1.0 + math.exp(uu) + Se)
        rb = (1.0 - d4) * rr4
        if rb > maxrb:
            maxrb = rb
        done = i + 1
        if prev_rb < level and rb >= level:
            first = i
            prev_rb = rb
            break
        prev_rb = rb
    x[0] = e3; x[1] = o3; x[2] = e4; x[3] = o4; x[4] = l3; x[5] = l4
    return first, maxrb, rb, done



@nb.njit(cache=True)
def crit_delta(x, m, v, Se, Sed, t0, lo, hi, H, nbis, eps):
    """smallest delta in [lo,hi] such that the state escapes (R_b>0.9) within H steps; bisection on the indicator
    (assumed monotone). returns np.inf if no escape at hi, -np.inf if escape already at lo."""
    rec = np.zeros(0)
    xs = x.copy(); ms = m.copy(); vs = v.copy()
    r = run(xs, ms, vs, Se, Sed, t0, hi, eps, H, 0.9, True, rec)
    if r[0] < 0:
        return np.inf
    xs = x.copy(); ms = m.copy(); vs = v.copy()
    r = run(xs, ms, vs, Se, Sed, t0, lo, eps, H, 0.9, True, rec)
    if r[0] >= 0:
        return -np.inf
    a = lo; b = hi
    for _ in range(nbis):
        mid = 0.5 * (a + b)
        xs = x.copy(); ms = m.copy(); vs = v.copy()
        r = run(xs, ms, vs, Se, Sed, t0, mid, eps, H, 0.9, True, rec)
        if r[0] >= 0:
            b = mid
        else:
            a = mid
    return 0.5 * (a + b)



_REC0 = np.zeros(0)


def avancer(s, delta, nsteps, eps=1e-10, cross=0.9, stop=True, rec=_REC0):
    """Advance state dict s IN PLACE (x,m,v). Returns (first, maxRb, lastRb, done)."""
    e = np.full(6, eps) if np.isscalar(eps) else np.asarray(eps, float)
    r = run(s["x"], s["m"], s["v"], s["Se"], s["Sed"], s["t"], float(delta), e, int(nsteps), float(cross), bool(stop), rec)
    s["t"] += r[3]
    return r


def avancer_obs(s, delta, nsteps, rec, eps=1e-10, cross=2.0, stop=False):
    e = np.full(6, eps) if np.isscalar(eps) else np.asarray(eps, float)
    r = run_obs(s["x"], s["m"], s["v"], s["Se"], s["Sed"], s["t"], float(delta), e, int(nsteps), float(cross), bool(stop), rec)
    s["t"] += r[3]
    return r


def jusqua_montee(s, delta, nmax, level, prev_rb=0.0, eps=1e-10):
    """advance until R_b crosses `level` upward. returns (steps used or -1, rb)."""
    e = np.full(6, eps) if np.isscalar(eps) else np.asarray(eps, float)
    first, _, rb, done = run_until_up(s["x"], s["m"], s["v"], s["Se"], s["Sed"], s["t"], float(delta), e, int(nmax), float(level), float(prev_rb))
    s["t"] += done
    return first, rb, done


def copie(s):
    return dict(x=s["x"].copy(), m=s["m"].copy(), v=s["v"].copy(), Se=s["Se"], Sed=s["Sed"], t=s["t"])
