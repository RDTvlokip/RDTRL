"""Reduced Adam simulator v2 (reviewer's, for checking): rows 3 and 4 of the sender (message-10 logit and the
26 symmetric competitors), receiver row 10 with its 25-ref tail frozen. Exact float64 gradients."""
import math
import numpy as np
import torch

B, C, LR, N = 0.02, 26, 0.05, 27
PLI = 0.0134372100660973
B1, B2 = 0.9, 0.999


def charger(fichier):
    c = torch.load(fichier)
    e, r = c["params"]
    st = c["opt"]["state"]
    ge, gr = st[0], st[1]

    def sep(mat, i):
        return float(mat[i, 10]), float(np.delete(mat[i].numpy(), 10).mean())
    s = {}
    for nom, i in (("3", 3), ("4", 4)):
        (s["e10_" + nom], s["eo_" + nom]) = sep(e, i)
        (s["m_e10_" + nom], s["m_eo_" + nom]) = sep(ge["exp_avg"], i)
        (s["v_e10_" + nom], s["v_eo_" + nom]) = sep(ge["exp_avg_sq"], i)
    s["l3"], s["l4"] = float(r[10, 3]), float(r[10, 4])
    s["m_l3"], s["m_l4"] = float(gr["exp_avg"][10, 3]), float(gr["exp_avg"][10, 4])
    s["v_l3"], s["v_l4"] = float(gr["exp_avg_sq"][10, 3]), float(gr["exp_avg_sq"][10, 4])
    tail = np.delete(r[10].numpy().astype(float), [3, 4]) - s["l4"]
    s["tail_d"] = tail
    s["step"] = float(ge["step"])
    return s


def simuler(s0, delta, eps, pas, mode="full", pre=None, trace=True, cross=0.9, fixe_v=None, seuil_stop=True):
    e3, o3 = s0["e10_3"], s0["eo_3"]; e4, o4 = s0["e10_4"], s0["eo_4"]
    l3, l4 = s0["l3"], s0["l4"]
    m = dict(e3=s0["m_e10_3"], o3=s0["m_eo_3"], e4=s0["m_e10_4"], o4=s0["m_eo_4"], l3=s0["m_l3"], l4=s0["m_l4"])
    v = dict(e3=s0["v_e10_3"], o3=s0["v_eo_3"], e4=s0["v_e10_4"], o4=s0["v_eo_4"], l3=s0["v_l3"], l4=s0["v_l4"])
    td = s0["tail_d"]
    et = np.exp(td); Se = et.sum(); Sed = (et * td).sum()
    t = s0["step"]
    out = np.empty((pas, 6)) if trace else None
    bascule = -1
    sw = 1.0
    for i in range(pas):
        z3 = e3 - o3; z4 = e4 - o4
        s3 = 1.0 / (1.0 + C * math.exp(-z3)); d3 = C * math.exp(-z3) * s3
        d4 = C * math.exp(-z4) / (1.0 + C * math.exp(-z4)); s4 = 1.0 - d4
        u = l3 - l4
        w3 = math.exp(u)
        Z = w3 + 1.0 + Se
        r3 = w3 / Z; r4 = 1.0 / Z
        # receiver entropy pieces
        logZ = math.log(Z)
        sum_rlogr = r3 * (u - logZ) + r4 * (-logZ) + (Sed - logZ * Se) / Z
        Hr = -sum_rlogr
        a3 = (1 - delta) * s3; a4 = (1 + delta) * s4
        abar = r3 * a3 + r4 * a4
        g_l3 = r3 * ((a3 - abar) - B * ((u - logZ) + Hr)) / N
        g_l4 = r4 * ((a4 - abar) - B * ((-logZ) + Hr)) / N
        X3 = (1 - delta) * r3 / B; X4 = (1 + delta) * r4 / B
        g_e3 = s3 * d3 * B * (X3 - z3) / N
        g_e4 = s4 * d4 * B * (X4 - z4) / N
        g = dict(e3=g_e3, o3=-g_e3 / C, e4=g_e4, o4=-g_e4 / C, l3=g_l3, l4=g_l4)
        t += 1
        bc1 = 1 - B1 ** t; sq = math.sqrt(1 - B2 ** t)
        for k in g:
            m[k] = B1 * m[k] + (1 - B1) * g[k]
            v[k] = B2 * v[k] + (1 - B2) * g[k] * g[k]
        if mode == "vfix":
            den = fixe_v
        else:
            den = {k: math.sqrt(v[k]) / sq + (eps[k] if isinstance(eps, dict) else eps) for k in v}
        if mode == "m_is_g":  # no first moment lag
            pass
        e3 += LR * (m["e3"] / bc1) / den["e3"]; o3 += LR * (m["o3"] / bc1) / den["o3"]
        e4 += LR * (m["e4"] / bc1) / den["e4"]; o4 += LR * (m["o4"] / bc1) / den["o4"]
        l3 += LR * (m["l3"] / bc1) / den["l3"]; l4 += LR * (m["l4"] / bc1) / den["l4"]
        if trace:
            out[i] = (d3, r4, math.sqrt(v["e3"]), math.sqrt(v["l3"]), d4, u)
        if bascule < 0:
            uu = l3 - l4
            rr4 = 1.0 / (1.0 + math.exp(uu) + Se * math.exp(-0.0))  # approx
            ee4 = 1.0 - d4
            if ee4 * rr4 > cross:
                bascule = i
                if seuil_stop and not trace:
                    break
    fin = {"e10_3": e3, "eo_3": o3, "e10_4": e4, "eo_4": o4, "l3": l3, "l4": l4,
           "m_e10_3": m["e3"], "m_eo_3": m["o3"], "m_e10_4": m["e4"], "m_eo_4": m["o4"], "m_l3": m["l3"], "m_l4": m["l4"],
           "v_e10_3": v["e3"], "v_eo_3": v["o3"], "v_e10_4": v["e4"], "v_eo_4": v["o4"], "v_l3": v["l3"], "v_l4": v["l4"],
           "tail_d": s0["tail_d"], "step": t}
    return out, bascule, fin
