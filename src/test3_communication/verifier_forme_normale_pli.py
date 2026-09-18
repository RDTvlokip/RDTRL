"""Protocole precis (agent-dipankar, 18/09/2026) pour trancher si le
point selle rencontre a pas=600 sous masse de fond est le MEME que H6
(0,994300 / 0,829390) ou un voisin avec sa propre geometrie locale.

Le "x17" initial etait un artefact de definition de tau (corrige a
x5,94-6,20, CARNET.md). Ici : etape 2 du protocole -- ajuster le
coefficient quadratique `a` de la forme normale du pli
(dx/dt = mu + a*(x-x0)^2) DIRECTEMENT sur les donnees brutes de vitesse
`ds3/dt` en fonction de `x=s3` (pas de gradient, pas de supposer a=1),
separement pour H6-direct et pour le cas retarde. Si `a_H6direct` et
`a_delayed` different significativement, c'est une preuve DIRECTE
(indpendante de tout probleme d'unites tau/mu) que les deux points
selles ont une geometrie locale differente.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import activer, parametres
from verifier_prior_asymetrique import objectif_pondere
from verifier_sonde_bassin import fixer_s3, fixer_r4, poids_delta
from verifier_masse_fond_systeme_reel import fixer_masse_fond

DELTA = 0.013
ADAM_EPS = 1e-10
REFERENTS_FOND = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


def trace_s3(cible_s3, R_init, masse_fond, pas, check_tous=1, lr=0.05):
    poids = poids_delta(DELTA)
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    fixer_s3(e, cible_s3)
    fixer_r4(r, R_init)
    if masse_fond > 0:
        fixer_masse_fond(r, REFERENTS_FOND, masse_fond)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=lr, eps=ADAM_EPS)
    s3_trace = []
    for i in range(pas):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        if i % check_tous == 0:
            with torch.no_grad():
                s3 = e.loi()[3, 10].item()
            s3_trace.append((i, s3))
    return s3_trace


def moindres_carres_quadratique(xs, ys):
    """Ajuste y = A*x^2 + B*x + C par moindres carres (3 inconnues,
    equations normales resolues a la main, pas de numpy)."""
    n = len(xs)
    Sx0, Sx1, Sx2, Sx3, Sx4 = n, sum(xs), sum(x**2 for x in xs), \
        sum(x**3 for x in xs), sum(x**4 for x in xs)
    Sy0 = sum(ys)
    Sxy = sum(x * y for x, y in zip(xs, ys))
    Sx2y = sum(x**2 * y for x, y in zip(xs, ys))
    # systeme 3x3 : [Sx4 Sx3 Sx2][A]   [Sx2y]
    #               [Sx3 Sx2 Sx1][B] = [Sxy ]
    #               [Sx2 Sx1 Sx0][C]   [Sy0 ]
    M = [[Sx4, Sx3, Sx2], [Sx3, Sx2, Sx1], [Sx2, Sx1, Sx0]]
    b = [Sx2y, Sxy, Sy0]
    # elimination de Gauss simple (3x3, pas de pivot -- suffisant ici)
    for i in range(3):
        piv = M[i][i]
        for k in range(i, 3):
            M[i][k] /= piv
        b[i] /= piv
        for j in range(3):
            if j != i:
                factor = M[j][i]
                for k in range(i, 3):
                    M[j][k] -= factor * M[i][k]
                b[j] -= factor * b[i]
    A, B, C = b
    return A, B, C


def mapper_v_de_x(cibles_s3, R_init=0.829390, masse_fond=0.0, lr=0.05):
    """Mesure directement v(x)=ds3/dpas en prenant UN SEUL pas d'Adam
    depuis chaque cible_s3 (etat frais a chaque fois) -- beaucoup plus
    dense et propre que de tracer une seule trajectoire, puisque chaque
    point est une mesure INDEPENDANTE de v a cet x precis, sans se
    soucier de savoir si la trajectoire y passe vraiment ou pas."""
    poids = poids_delta(DELTA)
    xs, vs = [], []
    for cible in cibles_s3:
        e, r = construire_mur23(adam_eps=ADAM_EPS)
        fixer_s3(e, cible)
        fixer_r4(r, R_init)
        if masse_fond > 0:
            fixer_masse_fond(r, REFERENTS_FOND, masse_fond)
        activer(e, r)
        opt = torch.optim.Adam(parametres(e, r), lr=lr, eps=ADAM_EPS)
        with torch.no_grad():
            s3_avant = e.loi()[3, 10].item()
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        with torch.no_grad():
            s3_apres = e.loi()[3, 10].item()
        v = s3_apres - s3_avant
        xs.append((s3_avant + s3_apres) / 2)
        vs.append(v)
    return xs, vs


def trace_vers_xv(trace):
    """Convertit une trace (pas,s3) en paires (x,v) par difference finie."""
    xs, vs = [], []
    for k in range(len(trace) - 1):
        pas0, s3_0 = trace[k]
        pas1, s3_1 = trace[k + 1]
        dpas = pas1 - pas0
        v = (s3_1 - s3_0) / dpas
        x_mid = (s3_0 + s3_1) / 2
        xs.append(x_mid)
        vs.append(v)
    return xs, vs


def analyser_xv(label, xs, vs):
    """Ajuste une parabole vitesse = A*(x-x0)^2 + mu sur des paires (x,v)
    deja calculees (par difference finie sur trajectoire OU par mesure
    directe d'un seul pas depuis plusieurs points de depart)."""
    A, B, C = moindres_carres_quadratique(xs, vs)
    if A != 0:
        x0 = -B / (2 * A)
        mu = C - B**2 / (4 * A)
    else:
        x0, mu = None, None
    print(f"=== {label} ===")
    print(f"  n_points={len(xs)}  x range=[{min(xs):.6f}, {max(xs):.6f}]")
    print(f"  fit: v = {A:.6e}*(x)^2 + {B:.6e}*x + {C:.6e}")
    print(f"  vertex x0={x0}  mu(=v au vertex)={mu}")
    return A, B, C, x0, mu


def analyser(label, trace, x0_guess=None):
    xs, vs = trace_vers_xv(trace)
    return analyser_xv(label, xs, vs)


if __name__ == "__main__":
    torch.set_printoptions(precision=6)

    print("### H6-direct (sans masse de fond, s3/R places sur le point instable) ###")
    trace_h6_effondre = trace_s3(0.994295, R_init=0.829390, masse_fond=0.0, pas=60, check_tous=1)
    # restreindre par INDICE DE PAS (pas par valeur de s3, qui melange des
    # regimes differents) -- la zone de ralentissement "lente" est pas=0-24
    # (avant que le regime rapide/runaway prenne clairement le dessus)
    trace_h6_local = [(i, s3) for (i, s3) in trace_h6_effondre if i <= 24]
    A_h6, B_h6, C_h6, x0_h6, mu_h6 = analyser(
        "H6-direct, fenetre LOCALE (pas<=24)", trace_h6_local)

    print()
    print("### Cas retarde (masse de fond 25%, R_init=0,60) ###")
    trace_delayed_effondre = trace_s3(0.999455, R_init=0.60, masse_fond=0.25, pas=800, check_tous=2)
    # restreindre par INDICE DE PAS : exclure la phase d'evacuation precoce
    # (pas<400, sans rapport) ET l'effondrement catastrophique (pas>700) --
    # ne garder QUE la zone de ralentissement pres du point selle
    trace_delayed_local = [(i, s3) for (i, s3) in trace_delayed_effondre if 400 <= i <= 700]
    A_d, B_d, C_d, x0_d, mu_d = analyser(
        "Cas retarde, fenetre LOCALE (pas=400-700)", trace_delayed_local)

    print()
    print("=== COMPARAISON DIRECTE DU COEFFICIENT QUADRATIQUE 'a' (fenetres locales) ===")
    print(f"  a_H6direct = {A_h6:.6e}  (x0={x0_h6:.6f}, attendu pres de 0,994300)")
    print(f"  a_delayed  = {A_d:.6e}  (x0={x0_d:.6f})")
    if A_h6 != 0 and A_d != 0:
        print(f"  ratio a_delayed/a_H6direct = {A_d/A_h6:.4f}")

    print()
    print("### H6-direct : POOL de plusieurs trajectoires courtes (pas la 1ere mesure,")
    print("    biaisee par le premier pas Adam a etat frais -- corrige en sautant pas<3) ###")
    cibles_pool = [0.994300 + d for d in
                   (-0.0020, -0.0014, -0.0008, -0.0004, -0.0002,
                    0.0002, 0.0004, 0.0008, 0.0014, 0.0020)]
    xs_pool, vs_pool = [], []
    for cible in cibles_pool:
        tr = trace_s3(cible, R_init=0.829390, masse_fond=0.0, pas=25, check_tous=1)
        # ne garder que pas>=3 (evite le biais du tout premier pas Adam)
        # et s3 pas encore parti en effondrement/saturation complete
        tr_utile = [(i, s3) for (i, s3) in tr if i >= 3 and 0.98 <= s3 <= 0.9999]
        x_part, v_part = trace_vers_xv(tr_utile)
        xs_pool.extend(x_part)
        vs_pool.extend(v_part)
    A_pool, B_pool, C_pool, x0_pool, mu_pool = analyser_xv(
        "H6-direct POOL (10 trajectoires courtes, pas>=3)", xs_pool, vs_pool)

    print()
    print("=== COMPARAISON FINALE (pool vs cas retarde) ===")
    print(f"  a_H6direct_pool = {A_pool:.6e}  (x0={x0_pool:.6f}, attendu pres de 0,994300)")
    print(f"  a_delayed       = {A_d:.6e}  (x0={x0_d:.6f})")
    if A_pool != 0 and A_d != 0:
        print(f"  ratio a_delayed/a_H6direct_pool = {A_d/A_pool:.4f}")
