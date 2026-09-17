"""Suite de verifier_refit_gap_hybride.py : ce dernier montrait que le
residu (R mesure - R_molle=sigmoid(2*delta/beta)) ne suit PAS une loi qui
s'annule vers delta_c -- il CROIT (exposant log-log mesure -0.94, pas
+0.5), ce qui contredit la lecture attendue d'un coefficient qui
« s'annule au pli ». Diagnostic : `R_molle` (sigmoid(2*delta/beta), sans
le terme de couplage (1-delta)*d3) n'est PAS la bonne reference -- ce
n'est pas une prediction qui vaut R_fold a delta_c, donc rien n'oblige le
residu par rapport a elle a s'annuler au pli. Le terme neglige,
(1-delta)*d3, grandit lui-meme a mesure que d3 (le deficit du referent 3)
grandit en approchant delta_c -- ce qui explique MECANIQUEMENT pourquoi
le residu contre cette reference-la grandit au lieu de retrecir.

La vraie signature de noeud-col a tester, algebrique et independante de
tout optimiseur : l'ECART ENTRE LES DEUX RACINES (stable et instable) du
systeme couple. Pres d'un pli, la forme normale standard
(dx/dt = mu - x^2) donne un ecart entre racines ~ sqrt(mu) -- ici
mu ~ (delta_c - delta). Test purement algebrique (racines(), deja
verifiees a 5-6 chiffres contre l'entrainement reel), sans entrainer quoi
que ce soit -- rapide, et decouple la question « le pli lui-meme suit-il
sqrt() » de la question separee (deja traitee en §7.65/hybride) de
savoir si un OPTIMISEUR donne suit cette meme loi en temps de convergence.
"""

import math

from verifier_refit_gap_hybride import BETA, f_pli, racines, localiser_delta_c, sigmoid


def stable_instable(delta):
    r = racines(delta)
    grad = sorted(x for x in r if x < 0.5)
    assert len(grad) == 2, f"attendu 2 racines graduees a delta={delta}, trouve {len(grad)}"
    return grad[0], grad[1]  # (stable, instable)


def R_de_d3(d3, delta):
    return sigmoid((2 * delta + (1 - delta) * d3) / BETA)


if __name__ == "__main__":
    DELTA_C = localiser_delta_c()
    print(f"=== ecart racine stable / racine instable, en d3 et en R, delta_c={DELTA_C:.9f} ===\n")

    fracs = (0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001)
    lignes = []
    for frac in fracs:
        delta = DELTA_C * (1 - frac)
        gap_dc = DELTA_C - delta
        d3_s, d3_i = stable_instable(delta)
        R_s, R_i = R_de_d3(d3_s, delta), R_de_d3(d3_i, delta)
        gap_d3 = d3_i - d3_s
        gap_R = R_i - R_s
        lignes.append((delta, gap_dc, d3_s, d3_i, gap_d3, R_s, R_i, gap_R))
        print(f"  delta={delta:.8f}  dc-delta={gap_dc:.4e}  d3_stable={d3_s:.8e}  "
              f"d3_instable={d3_i:.6e}  gap_d3={gap_d3:.6e}  R_stable={R_s:.6f}  "
              f"R_instable={R_i:.6f}  gap_R={gap_R:.6e}")

    def ajuster_exposant(xs_gap, ys_gap):
        xs = [math.log(g) for g in xs_gap]
        ys = [math.log(g) for g in ys_gap]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        pente = sxy / sxx
        intercept = my - pente * mx
        return pente, math.exp(intercept)

    gap_dcs = [l[1] for l in lignes]
    gap_d3s = [l[4] for l in lignes]
    gap_Rs = [l[7] for l in lignes]

    p_d3, c_d3 = ajuster_exposant(gap_dcs, gap_d3s)
    p_R, c_R = ajuster_exposant(gap_dcs, gap_Rs)

    print(f"\n=== exposant log-log, gap_d3 ~ C*(dc-delta)^p (predit p=0.5) ===")
    print(f"  p_mesure = {p_d3:.4f}   C = {c_d3:.6f}")

    print(f"\n=== exposant log-log, gap_R ~ C*(dc-delta)^p (predit p=0.5) ===")
    print(f"  p_mesure = {p_R:.4f}   C = {c_R:.6f}")

    print("\n=== coefficients individuels gap_d3/sqrt(dc-delta) ===")
    for delta, gap_dc, d3_s, d3_i, gap_d3, R_s, R_i, gap_R in lignes:
        print(f"  delta={delta:.8f}  gap_d3/sqrt(dc-delta)={gap_d3/math.sqrt(gap_dc):.6f}  "
              f"gap_R/sqrt(dc-delta)={gap_R/math.sqrt(gap_dc):.6f}")

    print("\n=== POURQUOI la derive : ajustement a deux termes,")
    print("    gap_d3 = C1*sqrt(dc-delta) + C2*(dc-delta)  (forme normale du pli + 1er correctif) ===")
    # moindres carres, base [sqrt(gap_dc), gap_dc]
    xs1 = [math.sqrt(g) for g in gap_dcs]
    xs2 = list(gap_dcs)
    ys = list(gap_d3s)
    n = len(ys)
    s11 = sum(a * a for a in xs1)
    s12 = sum(a * b for a, b in zip(xs1, xs2))
    s22 = sum(b * b for b in xs2)
    s1y = sum(a * y for a, y in zip(xs1, ys))
    s2y = sum(b * y for b, y in zip(xs2, ys))
    det = s11 * s22 - s12 * s12
    C1 = (s1y * s22 - s2y * s12) / det
    C2 = (s11 * s2y - s12 * s1y) / det
    print(f"  C1 (coefficient sqrt, attendu ~0.2212) = {C1:.6f}")
    print(f"  C2 (correctif lineaire en dc-delta)    = {C2:.6f}")
    print("  residus (gap_d3 mesure - modele a deux termes) :")
    for (delta, gap_dc, *_), g_d3 in zip(lignes, gap_d3s):
        modele = C1 * math.sqrt(gap_dc) + C2 * gap_dc
        print(f"    delta={delta:.8f}  gap_d3_mesure={g_d3:.6e}  modele={modele:.6e}  "
              f"residu_relatif={(g_d3-modele)/g_d3*100:+.3f}%")
