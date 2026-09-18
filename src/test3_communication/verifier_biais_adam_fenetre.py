"""Etapes 4 et 5 du protocole agent-dipankar (18/09/2026, CARNET.md fin
de §7.65) pour trancher si le point selle rencontre a pas=600 sous
masse de fond est le MEME que H6 (0,994300/0,829390) ou un voisin.

Etape 4 : R derive-t-il pendant la fenetre 0-59 pas de H6-direct, ou
reste-t-il bien sur 0,829390 comme suppose implicitement par le fit
1D (v(x) = f(s3) seul) ?

Etape 5, et surtout NOUVELLE HYPOTHESE non listee dans le protocole
d'origine mais trouvee en y reflechissant (regle CLAUDE.md : ajouter
ses propres pistes) : la correction de biais d'Adam (m_hat, v_hat)
n'est PAS un phenomene de "premier pas seulement" (deja identifie et
corrige pour les tentatives (ii)/(iii)) -- c'est une ENVELOPPE
MULTIPLICATIVE continue g(t) = bias1(t)/sqrt(bias2(t)) qui varie
fortement sur ~50-100 pas avant de saturer a 1. Elle est IDENTIQUE
pour tout parametre (ne depend que de t, beta1=0.9, beta2=0.999, pas
du gradient reel) puisque m_hat=m/(1-beta1^t) et v_hat=v/(1-beta2^t).

Hypothese standard/academique : la fenetre H6-direct (pas<=24) est
presque entierement dans la zone de forte variation de g(t) (g passe
de 0,316 a t=1 a un minimum vers t~15 puis remonte), alors que la
fenetre du cas retarde (pas=400-700) est deja loin dans la zone de
saturation lente -- ce qui suffirait a expliquer l'instabilite de
a_H6direct (x5,6 selon sous-fenetre) contre la stabilite relative de
a_delayed (x1,56), SANS invoquer une geometrie locale differente.

Test : calculer g(t) exactement (verifie contre l'etat reel de
l'optimiseur, pas seulement la formule), deflater v(x) par g(t) dans
les deux fenetres, refaire le fit quadratique sur v corrige, comparer
la stabilite et la magnitude de 'a' AVANT/APRES correction.
"""

import sys
sys.path.insert(0, '.')
import math
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import activer, parametres
from verifier_prior_asymetrique import objectif_pondere
from verifier_sonde_bassin import fixer_s3, fixer_r4, poids_delta
from verifier_masse_fond_systeme_reel import fixer_masse_fond
from verifier_forme_normale_pli import moindres_carres_quadratique

DELTA = 0.013
ADAM_EPS = 1e-10
BETA1 = 0.9
BETA2 = 0.999
REFERENTS_FOND = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


def g_biais(t, beta1=BETA1, beta2=BETA2):
    """Enveloppe multiplicative de la correction de biais d'Adam,
    independante du parametre (ne depend que de t)."""
    bias1 = 1.0 / (1.0 - beta1 ** t)
    bias2 = 1.0 / (1.0 - beta2 ** t)
    return bias1 / math.sqrt(bias2)


def verifier_formule_contre_etat_reel():
    """Verifie g(t) (formule) contre l'etat REEL de l'optimiseur Adam
    apres t pas, sur le parametre e.p[0] (emetteur) -- pas de confiance
    aveugle en ma propre derivation (regle 5bis s'applique a moi aussi)."""
    poids = poids_delta(DELTA)
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    fixer_s3(e, 0.994295)
    fixer_r4(r, 0.829390)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=0.05, eps=ADAM_EPS,
                            betas=(BETA1, BETA2))
    print("=== Verification g(t) formule vs etat reel de l'optimiseur ===")
    for cible_t in (1, 5, 10, 24, 50):
        while True:
            st = opt.state.get(e.p[0])
            t_actuel = 0 if st is None else st.get('step', 0)
            if isinstance(t_actuel, torch.Tensor):
                t_actuel = int(t_actuel.item())
            if t_actuel >= cible_t:
                break
            j, _ = objectif_pondere(e, r, BETA, poids)
            opt.zero_grad()
            (-j).backward()
            opt.step()
        st = opt.state[e.p[0]]
        exp_avg = st['exp_avg'][3, 10].item()
        exp_avg_sq = st['exp_avg_sq'][3, 10].item()
        t = st['step']
        if isinstance(t, torch.Tensor):
            t = int(t.item())
        m_hat = exp_avg / (1 - BETA1 ** t)
        v_hat = exp_avg_sq / (1 - BETA2 ** t)
        # ratio update_reel / (m/sqrt(v)) = m_hat/sqrt(v_hat) * sqrt(v)/m
        # = g(t) exactement (independant de m,v tant que eps negligeable)
        ratio_reel = (m_hat / (math.sqrt(v_hat) + ADAM_EPS)) / \
            (exp_avg / (math.sqrt(exp_avg_sq) + ADAM_EPS))
        g_formule = g_biais(t)
        print(f"  t={t:3d}  g_formule={g_formule:.6f}  "
              f"g_mesure_sur_etat_reel={ratio_reel:.6f}  "
              f"ecart={abs(g_formule-ratio_reel):.2e}")


def trace_complet(cible_s3, R_init, masse_fond, pas, check_tous=1, lr=0.05):
    """Comme entrainer_trace (verifier_trajectoire_renversement.py) mais
    renvoie aussi le pas Adam interne 't' pour pouvoir deflater par g(t)."""
    poids = poids_delta(DELTA)
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    fixer_s3(e, cible_s3)
    fixer_r4(r, R_init)
    if masse_fond > 0:
        fixer_masse_fond(r, REFERENTS_FOND, masse_fond)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=lr, eps=ADAM_EPS,
                            betas=(BETA1, BETA2))
    trace = []
    for i in range(pas):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        if i % check_tous == 0:
            with torch.no_grad():
                s_ = e.loi()
                r_ = r.loi()
                s3 = s_[3, 10].item()
                r4 = r_[10, 4].item()
            trace.append((i + 1, s3, r4))  # i+1 = pas Adam REEL (t) apres ce step
    return trace


def analyser_derive_r(label, trace):
    r4_vals = [r4 for (_, _, r4) in trace]
    print(f"=== derive de R (r4) -- {label} ===")
    print(f"  r4[0]={r4_vals[0]:.6f}  r4[fin]={r4_vals[-1]:.6f}  "
          f"min={min(r4_vals):.6f}  max={max(r4_vals):.6f}  "
          f"amplitude={max(r4_vals)-min(r4_vals):.2e}")
    return r4_vals


def fit_corrige(label, trace, i_min, i_max):
    """Extrait la sous-fenetre [i_min,i_max] par INDICE DE PAS, calcule
    v(x) brut par difference finie, deflate par g(t_milieu), refit."""
    sous = [(t, s3) for (t, s3, r4) in trace if i_min <= t <= i_max]
    xs_brut, vs_brut, vs_corr, ts_mid = [], [], [], []
    for k in range(len(sous) - 1):
        t0, s3_0 = sous[k]
        t1, s3_1 = sous[k + 1]
        dt = t1 - t0
        v = (s3_1 - s3_0) / dt
        t_mid = (t0 + t1) / 2
        x_mid = (s3_0 + s3_1) / 2
        xs_brut.append(x_mid)
        vs_brut.append(v)
        vs_corr.append(v / g_biais(t_mid))
        ts_mid.append(t_mid)
    A_brut, B_brut, C_brut = moindres_carres_quadratique(xs_brut, vs_brut)
    A_corr, B_corr, C_corr = moindres_carres_quadratique(xs_brut, vs_corr)
    print(f"=== {label} (pas {i_min}-{i_max}, n={len(xs_brut)}) ===")
    print(f"  a_BRUT  = {A_brut:.6e}")
    print(f"  a_CORRIGE (deflate g(t)) = {A_corr:.6e}")
    print(f"  g(t) sur cette fenetre : min={min(g_biais(t) for t in ts_mid):.4f} "
          f"max={max(g_biais(t) for t in ts_mid):.4f} "
          f"ratio={max(g_biais(t) for t in ts_mid)/min(g_biais(t) for t in ts_mid):.4f}")
    return A_brut, A_corr


if __name__ == "__main__":
    torch.set_printoptions(precision=6)

    verifier_formule_contre_etat_reel()

    print()
    print("=== g(t) sur les deux fenetres utilisees pour les fits precedents ===")
    g_h6 = [g_biais(t) for t in range(1, 25)]
    g_delayed = [g_biais(t) for t in range(400, 701)]
    print(f"  H6-direct (t=1..24)   : g(1)={g_h6[0]:.4f}  g(24)={g_h6[-1]:.4f}  "
          f"min={min(g_h6):.4f}  max={max(g_h6):.4f}  ratio max/min={max(g_h6)/min(g_h6):.4f}")
    print(f"  delayed   (t=400..700): g(400)={g_delayed[0]:.4f}  g(700)={g_delayed[-1]:.4f}  "
          f"min={min(g_delayed):.4f}  max={max(g_delayed):.4f}  ratio max/min={max(g_delayed)/min(g_delayed):.4f}")

    print()
    print("### Etape 4 : derive de R pendant la fenetre H6-direct (pas 0-59) ###")
    trace_h6 = trace_complet(0.994295, R_init=0.829390, masse_fond=0.0, pas=60, check_tous=1)
    analyser_derive_r("H6-direct, pas 0-59", trace_h6)

    print()
    print("### Derive de R pendant la fenetre du cas retarde (pas 400-700), pour comparaison ###")
    trace_delayed = trace_complet(0.999455, R_init=0.60, masse_fond=0.25, pas=800, check_tous=2)
    analyser_derive_r("cas retarde, pas 400-700 (extrait de la trace 0-800)",
                       [(t, s3, r4) for (t, s3, r4) in trace_delayed if 400 <= t <= 700])

    print()
    print("### Etape 5 : fit 'a' BRUT vs CORRIGE (deflate par g(t)) ###")
    fit_corrige("H6-direct", trace_h6, 1, 24)
    fit_corrige("cas retarde", trace_delayed, 400, 700)

    print()
    print("=== Stabilite du a_CORRIGE : sous-fenetres de H6-direct ===")
    for (i_min, i_max) in [(1, 12), (13, 24), (1, 24)]:
        fit_corrige(f"H6-direct sous-fenetre pas={i_min}-{i_max}", trace_h6, i_min, i_max)
