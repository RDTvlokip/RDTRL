"""Tour 75 (vraie critique de dipankarsarkar, 20/10/2026) : la dent de scie des salves et leur seuil de demarrage.

Il dit : S a 0,87-0,965 du seuil peut etre une dent de scie (S monte entre deux salves parce que sqrt(v) decroit, une salve remplit v et S retombe de ce remplissage
fois sqrt(v)/(sqrt(v)+eps)) ; lectures : S au pas ou chaque salve demarre (38 a un ou deux pas), profondeur de la dent contre eps/(sqrt(v)+eps) ; a b1 = 0,8 et 0,95 les
salves demarrent-elles a 18 et 78 ?

S_c(t) = - lr * (valeur propre raide de P J6), P = diag(1/(sqrt(v_i) + eps)), J6 = jacobien (6 coordonnees brutes) au point de la reduction ; pour v fige le momentum est le meme b1
pour toutes les coordonnees, donc (mu - b1)(mu - 1) = (1 - b1) lr lambda_i mu pour chaque valeur propre lambda_i de P J6 et le bord (mu = -1) est S_c = 2 (1 + b1)/(1 - b1).
S_66(t) = lr * 2 |J_uu| / (sqrt(v_l3) + eps) est la formule des tableaux du tour 66.

Usage : python verifier_tour75_salves_depart.py verifier
        python verifier_tour75_salves_depart.py salves <b1> <eps> <x> [<nrun> [<nrec>]]     (b1 = 0.9, 0.8 ou 0.95 ; lr 0,05 ; x = pli - delta)
"""

import importlib
import json
import sys

import numpy as np
from mpmath import mp, mpf

sys.path.insert(0, ".")
from verifier_tour74_valeur_propre import jac6, PLI

mp.dps = 50
LR = 0.05
J_UU = 1.1204e-4
MODULES = {0.9: "reduction_numba_tour59", 0.8: "reduction_numba_tour66_b1_08", 0.95: "reduction_numba_tour66_b1_095"}


def etat_base():
    d = json.load(open("etat_base_reduction_mur23.json", encoding="utf-8"))
    return dict(x=np.array(d["x"]), m=np.array(d["m"]), v=np.array(d["v"]), Se=d["Se"], Sed=d["Sed"], t=d["t"])


def mode_verifier():
    cj = 0.01945
    print("exact (racine de eps^2 + eps (1-b1)(1-cJ) - cJ (1-b1) = 0), premier ordre cJ, second ordre cJ (1 - b1 cJ/(1-b1)), rapports a cJ")
    for b1 in (0.8, 0.9, 0.95):
        e = 0.5 * (-(1 - b1) * (1 - cj) + np.sqrt(((1 - b1) * (1 - cj)) ** 2 + 4 * cj * (1 - b1)))
        e2 = cj * (1 - b1 * cj / (1 - b1))
        print(f"   b1 = {b1} : levier b1/(1-b1) = {b1 / (1 - b1):.0f} ; exact eps = {e:.6f} (rapport {e / cj:.4f}, ln mu = {np.log1p(e):.5f}) ; second ordre {e2:.6f} (rapport {e2 / cj:.4f}) ; ecart {100 * (e2 / e - 1):+.1f} %")
    print("seuils 2(1+b1)/(1-b1) :", {b1: 2 * (1 + b1) / (1 - b1) for b1 in (0.8, 0.9, 0.95)})


def courbe_Sc(x6, delta, lr, eps, ratio_e3, s_l3_ref, b1):
    """S_c en fonction de sqrt(v_l3) : grille log, sqrt(v_e3) = ratio_e3 sqrt(v_l3), sqrt(v_o3) = sqrt(v_e3)/26, sqrt(v_l4) = sqrt(v_l3), e4 et o4 morts (2e-14, 7,7e-16)."""
    J6 = jac6([mpf(float(a)) for a in x6], mpf(delta))
    J = np.array([[float(J6[i, j]) for j in range(6)] for i in range(6)])
    grille = s_l3_ref * np.exp(np.linspace(-0.6, 0.6, 121))
    sc = []
    for s in grille:
        se = ratio_e3 * s
        sv = np.array([se, se / 26.0, 2.0e-14, 7.7e-16, s, s])
        P = np.diag(1.0 / (sv + eps))
        ev = np.linalg.eigvals(P @ J)
        sc.append(-lr * float(np.min(ev.real)))
    return grille, np.array(sc), J


def mode_salves(b1, eps, xs, nrun, nrec):
    redlib = importlib.import_module(MODULES[b1])
    thr = 2 * (1 + b1) / (1 - b1)
    S1 = etat_base()
    delta = float(PLI - mpf(xs))
    ks = [int(a) for a in np.random.default_rng(7).integers(0, 4444, 16)]
    rng = np.random.default_rng(11)
    tous = dict(sc_creux=[], sc_vis=[], sc_pic=[], sc_apres=[], s66_creux=[], v_creux=[], v_apres=[], mediane_sc=[], mediane_s66=[], dt_vis=[], plateau=[], nb=0,
                sc_min=[], pente=[], replay=[], decal_min=[])
    esc = 0
    for k in ks[:nrun]:
        w = redlib.copie(S1)
        redlib.avancer(w, float(PLI - mpf("1e-6")), 20000 + k, eps=eps, cross=2.0, stop=False)
        w["x"] = w["x"] * (1.0 + 1e-9 * rng.standard_normal(6))
        redlib.avancer_obs(w, delta, 10000, np.zeros((1, 7)), eps=eps)
        x6 = w["x"].copy()
        w0 = redlib.copie(w)
        rec = np.zeros((nrec, 7))
        first, maxrb, rb, done = redlib.avancer_obs(w, delta, nrec, rec, eps=eps)
        if maxrb > 0.9:
            esc += 1
            continue
        u = rec[:, 2]; s_e3 = rec[:, 3]; s_l3 = rec[:, 4]
        ratio_e3 = float(np.median(s_e3 / s_l3))
        grille, sc_g, J = courbe_Sc(x6, delta, LR, eps, ratio_e3, float(np.median(s_l3)), b1)
        Sc = np.interp(np.log(s_l3), np.log(grille), sc_g)
        S66 = LR * 2 * J_UU / (s_l3 + eps)
        A = np.abs(u[2:] - 2 * u[1:-1] + u[:-2]) / 4.0
        E = np.maximum(A[:-1], A[1:])          # E[i] ~ pas i+1
        vis = 3e-4
        idx = np.nonzero(E > vis)[0]
        if len(idx) == 0:
            continue
        coupe = np.nonzero(np.diff(idx) > 30)[0]
        debuts = np.concatenate([[0], coupe + 1]); fins = np.concatenate([coupe, [len(idx) - 1]])
        groupes = [(idx[a], idx[b]) for a, b in zip(debuts, fins)]
        tous["mediane_sc"].append(float(np.median(Sc))); tous["mediane_s66"].append(float(np.median(S66)))
        for g, (a, b) in enumerate(groupes):
            if g == 0:
                continue
            prev_end = groupes[g - 1][1]
            seg = E[prev_end + 5:a]
            if len(seg) < 40:
                continue
            tp = a + int(np.argmax(E[a:b + 1]))
            lm = np.log10(E[prev_end + 5:a] + 1e-300)
            mn = lm.min()
            ok = np.nonzero(lm <= mn + np.log10(3.0))[0]
            t_creux = prev_end + 5 + int(np.median(ok))
            tous["plateau"].append(float(E[prev_end + 5:a].min() < 1e-14))
            t_min = prev_end + 5 + int(np.argmin(lm))
            tous["sc_min"].append(Sc[t_min + 1]); tous["decal_min"].append(t_min - t_creux)
            # estimateur par la croissance : pente observee s de ln E entre 0,08 et 0,35, racine reelle mu = -exp(s) => S inferre
            lE = np.log(E + 1e-300)
            tt = np.arange(max(t_min - 100, prev_end + 8), a)
            s_obs = (lE[tt + 2] - lE[tt - 2]) / 4.0
            bon = (s_obs > 0.08) & (s_obs < 0.35) & (E[tt] < vis)
            if np.any(bon):
                s_ok = s_obs[bon]
                s_inf = (1 + b1 + np.exp(s_ok) + b1 * np.exp(-s_ok)) / (1 - b1)
                tous.setdefault("dS_croiss", []).extend((s_inf - Sc[tt[bon] + 1]).tolist())
                tous.setdefault("Sinf_croiss", []).extend(s_inf.tolist())
            deb = prev_end + 30
            sel = np.arange(deb, t_min - 10)
            sel = sel[Sc[sel + 1] < thr - 1.0] if len(sel) else sel
            if len(sel) >= 20:
                tous["pente"].append(float(np.polyfit(sel, np.log(E[sel] + 1e-300), 1)[0]))
            if len(tous["replay"]) < 4 and k == ks[0]:
                w1 = redlib.copie(w0)
                redlib.avancer(w1, delta, t_min + 2, eps=eps, cross=2.0, stop=False)
                J6 = jac6([mpf(float(a)) for a in w1["x"]], mpf(delta))
                Jn = np.array([[float(J6[i, j]) for j in range(6)] for i in range(6)])
                sv = np.sqrt(w1["v"])
                evr = np.linalg.eigvals(np.diag(1.0 / (sv + eps)) @ Jn)
                tous["replay"].append((float(-LR * np.min(evr.real)), float(Sc[t_min + 1])))
            apres = min(b + 3, len(Sc) - 100)
            tous["sc_creux"].append(Sc[t_creux + 1]); tous["s66_creux"].append(S66[t_creux + 1])
            tous["sc_vis"].append(Sc[a + 1]); tous["sc_pic"].append(Sc[tp + 1])
            tous["sc_apres"].append(float(np.min(Sc[apres:apres + 60])))
            tous["v_creux"].append(s_l3[t_creux + 1]); tous["v_apres"].append(float(s_l3[apres + 30]))
            tous["dt_vis"].append(a - t_creux)
            tous["nb"] += 1
    if tous["nb"] == 0:
        print(f"b1 = {b1} eps = {eps:.1e} x = {xs} : 16/16 effondrees ou aucune salve ({esc} runs effondres)")
        return
    q = lambda key: np.array(tous[key])
    sccr = q("sc_creux")
    sc_med = float(np.mean(tous["mediane_sc"])); s66_med = float(np.mean(tous["mediane_s66"]))
    sc_apres = q("sc_apres")
    pred_med = float(np.sqrt(np.median(sccr) * np.median(sc_apres)))
    f_eps = float(np.median(eps / (q("v_creux") + eps)))
    print(f"b1 = {b1} eps = {eps:.1e} x = {xs} : seuil {thr:.0f} ; {tous['nb']} salves ({nrun} runs de {nrec} pas, {esc} effondres) ; plateau (E < 1e-14) : {100 * float(np.mean(tous['plateau'])):.0f} % des creux")
    print(f"   S_c au creux      : mediane {np.median(sccr):.3f} ({np.median(sccr) / thr:.4f} du seuil), 10-90 % [{np.percentile(sccr, 10):.3f} ; {np.percentile(sccr, 90):.3f}]")
    print(f"   S_66 au creux     : mediane {np.median(q('s66_creux')):.3f} ; rapport S_66/S_c au creux {np.median(q('s66_creux') / sccr):.4f}")
    print(f"   S_c a l'amplitude visible (> 3e-4) : mediane {np.median(q('sc_vis')):.3f} ; au pic : {np.median(q('sc_pic')):.3f} ; juste apres (min sur 60 pas) : {np.median(sc_apres):.3f}")
    print(f"   delai creux -> visible : mediane {np.median(q('dt_vis')):.0f} pas")
    print(f"   S_c mediane temporelle : {sc_med:.3f} ({sc_med / thr:.4f} du seuil) ; S_66 mediane (tour 66) : {s66_med:.3f} ({s66_med / thr:.4f}) ; moyenne geometrique creux x apres : {pred_med:.3f} ({pred_med / thr:.4f})")
    scm = q("sc_min")
    print(f"   S_c au minimum exact (argmin) : mediane {np.median(scm):.3f} ({np.median(scm) / thr:.4f} du seuil), 10-90 % [{np.percentile(scm, 10):.3f} ; {np.percentile(scm, 90):.3f}] ; argmin - creux (fenetre x3) : mediane {np.median(q('decal_min')):.0f} pas")
    print(f"   pente de decroissance de ln E entre les salves : mediane {np.median(q('pente')):.5f} ({len(tous['pente'])} salves) ; 1/2 ln b1 = {0.5 * np.log(b1):.5f} ; rapport {np.median(q('pente')) / (0.5 * np.log(b1)):.3f}")
    if tous.get("dS_croiss"):
        dS = np.array(tous["dS_croiss"])
        print(f"   estimateur par la croissance ({len(dS)} points) : S inferre - S_c : mediane {np.median(dS):+.3f} ({100 * np.median(dS) / thr:+.2f} % du seuil), 10-90 % [{np.percentile(dS, 10):+.3f} ; {np.percentile(dS, 90):+.3f}]")
    print("   rejeu exact (S_c complet, S_c interpole) : " + " ; ".join(f"{a:.3f} / {b_:.3f}" for a, b_ in tous["replay"]))
    print(f"   manque 1 - S_c mediane/seuil = {1 - sc_med / thr:.4f} ; eps/(sqrt(v_l3 creux) + eps) = {f_eps:.4f} ; sqrt(v) creux {np.median(q('v_creux')):.3e} -> apres {np.median(q('v_apres')):.3e} (remplissage {np.median(q('v_apres')) / np.median(q('v_creux')):.3f})")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "verifier":
        mode_verifier()
    else:
        mode_salves(float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 2, int(sys.argv[6]) if len(sys.argv) > 6 else 400000)
