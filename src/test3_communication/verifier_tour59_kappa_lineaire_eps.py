"""Tour 59 (28/09/2026) : le coefficient du fantome kappa(eps) de
t = kappa / sqrt(delta - delta_c') est-il lineaire en eps d'Adam ?

Motif : la vitesse d'un pas d'Adam est lr m / (sqrt(v) + eps). Si le
passage par le goulot du pli est gouverne par cette vitesse, le temps
de passage doit etre multiplie par (sqrt(v) + eps) / sqrt(v), donc
kappa(eps) = kappa0 (1 + eps / sqrt(v_lent)) = a + b eps, avec
sqrt(v_lent) = a / b.

Deux points d'ajustement (eps 1e-8 et 1e-7, kappa issus de
verifier_tour59_ajustement_fantome.py, 0,085 et 0,210), puis deux
verifications HORS echantillon :
  - eps 3e-8 (6 points de grille, kappa ajuste 0,111) ;
  - eps 1e-6, temps de bascule des runs du 27/09 (pli+3e-10 : 91 264
    pas ; pli+1e-9 : 50 845 pas), protocole chauffe seule.
"""

if __name__ == "__main__":
    k1, k7 = 0.085, 0.210
    b = (k7 - k1) / (1e-7 - 1e-8)
    a = k1 - b * 1e-8
    print(f"kappa(eps) = {a:.4f} + {b:.3e} * eps    sqrt(v_lent) = a/b = {a / b:.3e}")
    print(f"eps 3e-8 : kappa predit {a + b * 3e-8:.4f}   ajuste 0,111")
    # 2e-8 et 5e-8 : ajustes apres coup (grille du 28/09), jamais utilises pour a et b
    for eps, ajuste in ((2e-8, 0.0995), (5e-8, 0.1379)):
        p = a + b * eps
        print(f"eps {eps:.0e} : kappa predit {p:.4f}   ajuste {ajuste:.4f}   ecart {100 * (ajuste / p - 1):+.1f} %")
    k = a + b * 1e-6
    # E2 (29/09, reseau complet, eps 3e-6, chauffe 20 000 + 20 000, precommise par l'agent) :
    # t = kappa (1/2 + arctan(sqrt(delta0/D))/pi) / sqrt(D), delta0 = 1e-6, D = delta - delta_c' (Delta ~ 0
    # sans salves). Ma droite lineaire predit t(+1e-7) = 12 096 ; la sienne (asymptote 1,637e6) 13 846.
    import math
    print("--- E2, eps 3e-6 : temps de bascule mesures (13 846 / 9 343 / 5 421)")
    for D, t in ((1e-7, 13846), (2e-7, 9343), (5e-7, 5421)):
        facteur = 0.5 + math.atan(math.sqrt(1e-6 / D)) / math.pi
        kappa_mesure = t * math.sqrt(D) / facteur
        print(f"  D={D:.0e} : kappa mesure {kappa_mesure:.3f}   ma droite {a + b * 3e-6:.3f} (t predit {(a + b * 3e-6) * facteur / math.sqrt(D):.0f})   "
              f"sa formule 4,851 (t predit {4.851 * facteur / math.sqrt(D):.0f})   pente kappa/eps = {kappa_mesure / 3e-6:.3e}")
    print("--- fin E2")
    for off, obs in ((3e-10, 91264), (1e-9, 50845)):
        print(f"eps 1e-6, pli+{off:.0e} : t predit {k / off ** 0.5:.0f}   observe {obs}   "
              f"delta - delta_c' implique {(k / obs) ** 2:.2e} (Delta(1e-6) petit si ~ 0)")
