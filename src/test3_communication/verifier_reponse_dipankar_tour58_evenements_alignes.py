"""Tour 58 (23/09/2026, VRAIE critique de dipankarsarkar) : le co-timing
s3 <-> kick recepteur, mesure ALIGNE SUR LES EVENEMENTS.

Sa these : le "s3 plat a delta=0" du tour 54 peut n'etre que de la
saturation (s3(1-s3) ~1e-7 contre 1,04e-3) ; il faut comparer l'ecart
logit d3 - dbar (ligne 3 de l'emetteur) a chacun des 16 evenements
delta=0 et des 15 evenements delta reel, un test de magnitude ou le
signe n'entre pas.

Lit les traces de tracer_mur23_lignes3_10_complet.py (aucune
simulation ici). Partie A : reproduit les listes d'evenements
publiees au tour 57 et le chiffre 1,426e-10 du tour 54 (qui mesurait
sigmoid(logit brut), pas s3 -- bug trouve en verifiant sa premisse).
Partie B : vrai s3 (softmax sur 27) et vrai s3(1-s3) aux deux deltas.
"""

import sys
sys.path.insert(0, '.')
import math
import torch

from verifier_kicks_adam_grille_fine import detecter_evenements

DEBUT, PAS_MAX = 54000, 62000
DEBUT_DETECTION = 55000
DELTAS = (0.0, 0.013026615)


def charger(delta):
    return torch.load(f"D:/tmp/rdtrl_tour58_trace_mur23_g77777_k3_delta{delta}"
                      f"_pas{DEBUT}-{PAS_MAX}.pt")


def autres(ligne, k):
    idx = [j for j in range(ligne.shape[-1]) if j != k]
    return ligne[..., idx]


def series(t):
    """Series par pas, toutes en float64, indexees par (pas - DEBUT)."""
    e3, e4, r10 = t["e3"], t["e4"], t["r10"]
    o3 = autres(e3, 10)
    s = {}
    s["gap_r"] = r10[:, 4] - r10[:, 3]
    s["d3"] = e3[:, 10]
    s["dbar"] = o3.mean(dim=1)
    s["X"] = e3[:, 10] - o3.mean(dim=1)                  # ecart de dipankar
    s["G"] = e3[:, 10] - torch.logsumexp(o3, dim=1)       # ecart exact
    s["s3"] = torch.softmax(e3, dim=1)[:, 10]
    s["un_moins_s3"] = torch.exp(torch.logsumexp(o3, dim=1)
                                 - torch.logsumexp(e3, dim=1))
    s["sig_brut"] = torch.sigmoid(e3[:, 10])               # ce que mesurait le tour 54
    s["R4"] = torch.softmax(e4, dim=1)[:, 10] * torch.softmax(r10, dim=1)[:, 4]
    s["r3"] = torch.softmax(r10, dim=1)[:, 3]
    s["r4"] = torch.softmax(r10, dim=1)[:, 4]
    s["cv_autres"] = o3.std(dim=1) / o3.mean(dim=1).abs()
    return s


def evenements(s):
    vals = [(DEBUT + i, v) for i, v in enumerate(s["gap_r"].tolist())]
    fen = [v for p, v in vals if DEBUT_DETECTION <= p < PAS_MAX]
    base = sum(fen) / len(fen)
    return detecter_evenements(vals, base, seuil=0.0008, pas_min=DEBUT_DETECTION)


def partie_a(delta, s):
    pics = evenements(s)
    print(f"--- delta={delta} : {len(pics)} evenements sur le gap r4-r3 ---")
    print("  " + " ".join(f"{p}({'+' if d > 0 else '-'})" for p, d in pics))
    if delta == 0.0:
        i0, i1 = 59000 - DEBUT, 61000 - DEBUT
        sb = s["sig_brut"][i0:i1]
        base = sb[:100].mean()
        print(f"  reproduction tour 54 : sigmoid(logit brut) base={base.item():.8f}  "
              f"max|d|={(sb - base).abs().max().item():.3e}")
        print(f"  logit brut p_e[3,10] a pas=59000 : {s['d3'][i0].item():.6f}")
    return pics


def partie_b(delta, s):
    for pas in (59000, 59989, 61000):
        i = pas - DEBUT
        u = s["un_moins_s3"][i].item()
        print(f"  pas={pas}  vrai s3={s['s3'][i].item():.10f}  1-s3={u:.4e}  "
              f"s3(1-s3)={u*(1-u):.4e}  X=d3-dbar={s['X'][i].item():.6f}  "
              f"G={s['G'][i].item():.6f}  G-(X-ln26)={s['G'][i].item()-(s['X'][i].item()-math.log(26)):.2e}  "
              f"CV(26 autres)={s['cv_autres'][i].item():.2e}  "
              f"r3={s['r3'][i].item():.4e}  r4={s['r4'][i].item():.6f}  R4={s['R4'][i].item():.8f}")


PRE = (240, 60)      # fenetre calme avant le pic : [p-240, p-60]
POST = (150, 240)    # fenetre calme apres le pic : [p+150, p+240] (tronquee au bord)
BURST = 15           # la salve de periode 2 dure ~10 pas autour du pic


def residus(serie, p):
    """Residu de `serie` sur [p-BURST, p+BURST] par rapport a une droite
    ajustee sur les fenetres calmes avant/apres (retire la derive lente,
    qui est lineaire a delta=0). None si la fenetre apres est trop courte."""
    i = p - DEBUT
    fin = min(i + POST[1], len(serie) - 1)
    if fin - (i + POST[0]) < 30:
        return None
    idx = list(range(i - PRE[0], i - PRE[1] + 1)) + list(range(i + POST[0], fin + 1))
    x = torch.tensor(idx, dtype=torch.float64)
    y = serie[idx]
    A = torch.stack([x, torch.ones_like(x)], dim=1)
    coef = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.squeeze(1)
    xs = torch.arange(i - BURST, i + BURST + 1, dtype=torch.float64)
    return serie[i - BURST:i + BURST + 1] - (coef[0] * xs + coef[1])


def mesure(s, p, nom):
    rg, rx = residus(s["gap_r"], p), residus(s[nom], p)
    if rg is None:
        return None
    pente = (rx * rg).sum() / (rg * rg).sum()
    return rx[BURST].item(), rx.abs().max().item(), rg.abs().max().item(), pente.item()


def partie_c(delta, s, pics, nom="X"):
    """d3-dbar (ou G exact) a chaque evenement, a cote du meme calcul aux
    pseudo-evenements places au milieu entre deux evenements (controle :
    niveau "entre les kicks")."""
    print(f"--- delta={delta} : {nom} aligne sur les evenements (residu / derive lineaire) ---")
    res_ev, res_ctrl, sautes = [], [], []
    for k, (p, d) in enumerate(pics):
        m = mesure(s, p, nom)
        if m is None:
            sautes.append(p)
            continue
        res_ev.append(m)
        print(f"  pas={p}  gap au pic={d:+.4e}  {nom} au pic={m[0]:+.4e}  "
              f"max|{nom}| salve={m[1]:.4e}  max|gap| salve={m[2]:.4e}  pente {nom}/gap={m[3]:+.4e}")
        if k + 1 < len(pics):
            milieu = (p + pics[k + 1][0]) // 2
            mc = mesure(s, milieu, nom)
            if mc is not None:
                res_ctrl.append(mc)
    med = lambda v: sorted(v)[len(v) // 2]
    print(f"  evenements utilises : {len(res_ev)}  (sautes, fenetre apres hors trace : {sautes})")
    print(f"  mediane |{nom} au pic|        evenements = {med([abs(m[0]) for m in res_ev]):.4e}   "
          f"controle entre kicks = {med([abs(m[0]) for m in res_ctrl]):.4e}  (n={len(res_ctrl)})")
    print(f"  mediane max|{nom}| sur salve   evenements = {med([m[1] for m in res_ev]):.4e}   "
          f"controle entre kicks = {med([m[1] for m in res_ctrl]):.4e}")
    print(f"  mediane pente {nom}/gap        evenements = {med([m[3] for m in res_ev]):+.4e}   "
          f"min/max = {min(m[3] for m in res_ev):+.4e} / {max(m[3] for m in res_ev):+.4e}")
    return res_ev, res_ctrl


def partie_d(delta, s, t, pics):
    """Deux chiffres publies que dipankar met en cause :
    - |d_logit_r4| = 0,001504 a delta=0 (REPONSE_ORDRE56, "x7,35"), lu a
      pas=59989 contre l'interpolation 59000-61000 ;
    - "le kick de R4 survit a delta=0, dR4=+3,72e-6" (tour 54), calcule
      avec sigmoid(logit brut) au lieu du vrai R4 = s[4,10]*r[10,4]."""
    lr4 = t["r10"][:, 4]
    i0, i1, i = 59000 - DEBUT, 61000 - DEBUT, 59989 - DEBUT
    interp = lr4[i0] + (59989 - 59000) / 2000 * (lr4[i1] - lr4[i0])
    print(f"--- delta={delta} : logit r4 et vrai R4 ---")
    print(f"  reproduction tour 56 : d_logit_r4 a pas=59989 (interp 59000-61000) = {(lr4[i]-interp).item():+.6e}")
    s["lr4"] = lr4
    amp_lr4, amp_R4 = [], []
    for p, d in pics:
        a, b = residus(s["lr4"], p), residus(s["R4"], p)
        if a is None:
            continue
        amp_lr4.append(abs(a[BURST].item()))
        amp_R4.append(abs(b[BURST].item()))
    med = lambda v: sorted(v)[len(v) // 2]
    print(f"  aux {len(amp_lr4)} evenements : mediane |d_logit_r4| au pic = {med(amp_lr4):.4e}  "
          f"(min {min(amp_lr4):.4e}, max {max(amp_lr4):.4e})")
    print(f"  aux {len(amp_R4)} evenements : mediane |d vrai R4| au pic = {med(amp_R4):.4e}  "
          f"(vrai R4 de base ~{s['R4'][i0].item():.6f})")
    if delta == 0.0:
        sig = torch.sigmoid(t["e4"][:, 10]) * torch.sigmoid(t["r10"][:, 4])
        v = sig[59000 - DEBUT:61000 - DEBUT]
        base = v[:100].mean()
        print(f"  reproduction tour 54 : R4 'sigmoide brute' base={base.item():.8f}  "
              f"max|d|={(v - base).abs().max().item():.3e}   (vrai R4 base={s['R4'][i0].item():.8f})")


if __name__ == "__main__":
    torch.set_printoptions(precision=10)
    for delta in DELTAS:
        t = charger(delta)
        s = series(t)
        pics = partie_a(delta, s)
        partie_b(delta, s)
        partie_c(delta, s, pics, "X")
        partie_c(delta, s, pics, "G")
        partie_d(delta, s, t, pics)
