"""Agent (role dipankarsarkar, tour 59 simule) : la ligne 4 de l'emetteur,
deja dans les traces du tour 58 (e4, g_e4, v_e4), jamais regardee.

Deuxieme ligne pour falsifier la formule epinglee sur la ligne 3 :
formule(u4) = 1/2 [u4/(u4+eps) + (u4/26)/(u4/26+eps)], u4 = sqrt(v e4[10]),
contre la loi de couplage sous le plancher (sans u) :
pente4 = - pente3 * s4(1-s4) / s3(1-s3)   (delta=0 : poids et r3 r4 communs).
Et la part salve / fond du gradient de e4[10].

Lit seulement les traces D:/tmp/rdtrl_tour58_trace_*.pt.
"""
import sys
sys.path.insert(0, '.')
import torch

from verifier_reponse_dipankar_tour58_evenements_alignes import (
    charger, series, evenements, residus, DEBUT, DELTAS)

EPS = 1e-10


def formule(u, eps=EPS):
    return 0.5 * (u / (u + eps) + (u / 26) / (u / 26 + eps))


if __name__ == "__main__":
    med = lambda v: sorted(v)[len(v) // 2]
    for delta in DELTAS:
        t = charger(delta)
        s = series(t)
        e4 = t["e4"]
        autres = [j for j in range(27) if j != 10]
        s["X4"] = e4[:, 10] - e4[:, autres].mean(dim=1)
        un4 = torch.exp(torch.logsumexp(e4[:, autres], 1) - torch.logsumexp(e4, 1))
        u4 = t["v_e4"][:, 10].sqrt()
        u3 = t["v_e3"][:, 10].sqrt()
        g4, g3 = t["g_e4"][:, 10], t["g_e3"][:, 10]
        i = 59000 - DEBUT
        print(f"=== delta={delta} === pas 59000 : X3={s['X'][i].item():.5f}  X4={s['X4'][i].item():.5f}  "
              f"1-s3={s['un_moins_s3'][i].item():.4e}  1-s4={un4[i].item():.4e}  "
              f"u3={u3[i].item():.4e}  u4={u4[i].item():.4e}")
        lignes = []
        for p, d in evenements(s):
            rg = residus(s["gap_r"], p)
            if rg is None:
                continue
            r4, r3 = residus(s["X4"], p), residus(s["X"], p)
            sl4 = ((r4 * rg).sum() / (rg * rg).sum()).item()
            sl3 = ((r3 * rg).sum() / (rg * rg).sum()).item()
            k = p - DEBUT
            q = (un4[k] * (1 - un4[k]) / (s["un_moins_s3"][k] * (1 - s["un_moins_s3"][k]))).item()
            gb4 = (g4[k - 15:k + 16] - g4[k - 60:k - 20].mean()).abs().max().item()
            gf4 = abs(g4[k - 60:k - 20].mean().item())
            lignes.append((p, sl3, sl4, -sl3 * q, u4[k + 1].item(), gb4 / gf4))
            print(f"  pas={p} pente3={sl3:+.4e} pente4={sl4:+.4e} predit(couplage)={-sl3*q:+.4e} "
                  f"formule(u4)={formule(u4[k+1].item()):+.4e}  salve/fond g_e4={gb4/gf4:.4f}")
        print(f"  MEDIANES : pente4={med([l[2] for l in lignes]):+.4e}  predit(couplage)={med([l[3] for l in lignes]):+.4e}  "
              f"formule(u4)={formule(med([l[4] for l in lignes])):+.4e}  salve/fond={med([l[5] for l in lignes]):.4f}  "
              f"negatives={sum(l[2] < 0 for l in lignes)}/{len(lignes)}")
