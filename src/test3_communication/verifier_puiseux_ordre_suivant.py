"""Ferme la question laissee ouverte par l'agent-dipankar (17/09/2026,
verifier_puiseux_gap.py) : le coefficient D~8,0021 du terme eps^(3/2)
dans gap = C0*sqrt(eps) + D*eps^(3/2) + ... n'avait ete que mesure
numeriquement ("I have not derived D analytically"), jamais ferme.

Developpement de Lyapunov-Schmidt/Puiseux a l'ordre suivant. Avec
eps=t^2 et u=x-x0=a1*t+a2*t^2+a3*t^3+..., substitution dans
F(x0+u,eps)=0 (F=h, F_x=0 au pli par construction) :

  ordre t^2 : (1/2)F_xx*a1^2 + F_eps = 0            => a1^2 = A = -2F_eps/F_xx
  ordre t^3 : F_xx*a1*a2 + (1/6)F_xxx*a1^2 + F_xeps = 0   => a2 (independant du signe de a1)
  ordre t^4 : F_xx*(a1*a3 + a2^2/2) + (1/2)F_xxx*a1^2*a2
              + (1/24)F_xxxx*a1^4 + F_xeps*a2
              + (1/2)F_xxeps*a1^2 + (1/2)F_epseps = 0   => a3

Le gap entre les deux branches (s=+1/-1, a1 change de signe, a2 non) :
  gap = u_+ - u_- = 2*a1*t + 2*a3*t^3 + O(t^5)
      = 2*a1*sqrt(eps) + 2*a3*eps^(3/2) + O(eps^(5/2))
donc D_analytique = 2*a3 (a comparer au D~8,0021 mesure numeriquement).
"""

import sys
sys.path.insert(0, '.')
import mpmath as mp

from verifier_puiseux_gap import h, localiser_pli

mp.mp.dps = 50


if __name__ == "__main__":
    d3_c, delta_c = localiser_pli('0.003', '0.0134')
    print(f"d3_c = {d3_c}")
    print(f"delta_c = {delta_c}")

    F_xx = mp.diff(lambda x: h(x, delta_c), d3_c, 2)
    F_xxx = mp.diff(lambda x: h(x, delta_c), d3_c, 3)
    F_xxxx = mp.diff(lambda x: h(x, delta_c), d3_c, 4)
    F_eps = -mp.diff(lambda de: h(d3_c, de), delta_c)
    F_xeps = -mp.diff(lambda x: mp.diff(lambda de: h(x, de), delta_c), d3_c)
    F_xxeps = -mp.diff(lambda x: mp.diff(lambda de: h(x, de), delta_c), d3_c, 2)
    F_epseps = mp.diff(lambda de: h(d3_c, de), delta_c, 2)

    print(f"\nF_xx     = {F_xx}")
    print(f"F_xxx    = {F_xxx}")
    print(f"F_xxxx   = {F_xxxx}")
    print(f"F_eps    = {F_eps}")
    print(f"F_xeps   = {F_xeps}")
    print(f"F_xxeps  = {F_xxeps}")
    print(f"F_epseps = {F_epseps}")

    A = -2 * F_eps / F_xx
    a1 = mp.sqrt(A)
    C0 = 2 * a1
    print(f"\nA  = {A}")
    print(f"a1 = sqrt(A) = {a1}")
    print(f"C0 = 2*a1 = {C0}  (attendu ~0,2212603871)")

    a2 = -((F_xxx / 6) * a1**2 + F_xeps) / F_xx
    print(f"\na2 = {a2}  (= b de verifier_puiseux_gap.py, attendu ~1,4408268274)")

    # ordre t^4 : F_xx*(a1*a3 + a2^2/2) + (1/2)F_xxx*a1^2*a2 + (1/24)F_xxxx*a1^4
    #             + F_xeps*a2 + (1/2)F_xxeps*a1^2 + (1/2)F_epseps = 0
    reste = ((F_xxx / 2) * a1**2 * a2
             + (F_xxxx / 24) * a1**4
             + F_xeps * a2
             + (F_xxeps / 2) * a1**2
             + F_epseps / 2)
    a3 = -(reste + F_xx * (a2**2 / 2)) / (F_xx * a1)
    D_analytique = 2 * a3

    print(f"\na3 = {a3}")
    print(f"D_analytique = 2*a3 = {D_analytique}")
    print(f"D_empirique (verifier_puiseux_gap.py, extrapole a eps->0) = 8,0021...")
    print(f"ecart relatif = {abs(D_analytique - mp.mpf('8.0021'))/mp.mpf('8.0021')*100} %")
