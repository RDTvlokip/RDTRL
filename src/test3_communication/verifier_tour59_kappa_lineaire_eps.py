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
    k = a + b * 1e-6
    for off, obs in ((3e-10, 91264), (1e-9, 50845)):
        print(f"eps 1e-6, pli+{off:.0e} : t predit {k / off ** 0.5:.0f}   observe {obs}   "
              f"delta - delta_c' implique {(k / obs) ** 2:.2e} (Delta(1e-6) petit si ~ 0)")
