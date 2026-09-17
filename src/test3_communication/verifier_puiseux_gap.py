"""Verification independante (haute precision, mpmath) de la critique de
l'agent-dipankar du 17/09/2026 sur verifier_gap_racines.py :

1. Mon tableau a 11 points (float64) rebrousse chemin au dernier point
   (0,221305 -> 0,221309) -- pretendu artefact de precision, pas physique.
2. Mon modele a 2 termes (gap = C1*sqrt(eps) + C2*eps) utiliserait une
   base fausse : le developpement de Lyapunov-Schmidt d'un pli generique
   donne gap = 2*sqrt(A*eps) + O(eps^(3/2)) -- le terme lineaire en eps
   serait ANALYTIQUEMENT NUL (annule entre les deux branches), le vrai
   terme suivant est eps^(3/2), donc ratio=gap/sqrt(eps) = C0 + D*eps
   (lineaire en eps), pas C1 + C2*sqrt(eps) comme mon fit l'impliquait.
3. Valeur revendiquee par calcul analytique local (derivees partielles de
   h au pli) : C0 = 0,2212603871...

Rien de tout ca n'est pris pour acquis (regle 5bis, meme venant d'un
agent qui imite dipankar) -- refait ici en mpmath (50 chiffres) de zero,
independamment de son code.
"""

import sys
sys.path.insert(0, '.')
import mpmath as mp

mp.mp.dps = 50

BETA = mp.mpf('0.02')


def h(d3, delta, beta=BETA):
    R = 1 / (1 + mp.e ** (-(2 * delta + (1 - delta) * d3) / beta))
    return d3 / (26 * (1 - d3)) - mp.e ** (-(1 - delta) * (1 - R) / beta)


def h_d3(d3, delta):
    return mp.diff(lambda x: h(x, delta), d3)


def localiser_pli(d3_guess, delta_guess):
    """Newton 2D sur (h=0, h_d3=0)."""
    d3, delta = mp.mpf(d3_guess), mp.mpf(delta_guess)
    for _ in range(60):
        F = mp.matrix([h(d3, delta), h_d3(d3, delta)])
        # jacobien par differences finies mpmath (diff multivarie)
        Jd3 = mp.matrix([
            mp.diff(lambda x: h(x, delta), d3),
            mp.diff(lambda x: h_d3(x, delta), d3),
        ])
        Jdelta = mp.matrix([
            mp.diff(lambda de: h(d3, de), delta),
            mp.diff(lambda de: h_d3(d3, de), delta),
        ])
        J = mp.matrix([[Jd3[0], Jdelta[0]], [Jd3[1], Jdelta[1]]])
        delta_vec = mp.lu_solve(J, -F)
        d3 += delta_vec[0]
        delta += delta_vec[1]
        if max(abs(delta_vec[0]), abs(delta_vec[1])) < mp.mpf('1e-45'):
            break
    return d3, delta


def racine_dans(delta, lo, hi):
    """Bissection haute precision de h(., delta) sur (lo, hi)."""
    flo, fhi = h(lo, delta), h(hi, delta)
    assert (flo < 0) != (fhi < 0), f"pas de racine entre {lo} et {hi} a delta={delta}"
    for _ in range(200):
        mid = (lo + hi) / 2
        fm = h(mid, delta)
        if (fm < 0) == (flo < 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return (lo + hi) / 2


if __name__ == "__main__":
    print("=== 1) localisation du pli (Newton 2D, mpmath 50 chiffres) ===")
    d3_c, delta_c = localiser_pli('0.003', '0.0134')
    print(f"  d3_c    = {d3_c}")
    print(f"  delta_c = {delta_c}")
    print(f"  (publie/verifie 3x en float64 : 0.013437210)")

    print("\n=== 2) derivees partielles de h au pli (mpmath.diff, mixte) ===")
    F_xx = mp.diff(lambda x: h(x, delta_c), d3_c, 2)
    F_xxx = mp.diff(lambda x: h(x, delta_c), d3_c, 3)
    F_eps = -mp.diff(lambda de: h(d3_c, de), delta_c)  # eps = delta_c - delta
    F_xeps = -mp.diff(lambda x: mp.diff(lambda de: h(x, de), delta_c), d3_c)
    print(f"  F_xx   = {F_xx}")
    print(f"  F_xxx  = {F_xxx}")
    print(f"  F_eps  = {F_eps}")
    print(f"  F_xeps = {F_xeps}")

    A = -2 * F_eps / F_xx
    C0 = 2 * mp.sqrt(A)
    b = -((F_xxx / 6) * A + F_xeps) / F_xx
    print(f"\n  A  = {A}")
    print(f"  C0 = 2*sqrt(A) = {C0}")
    print(f"  b  = {b}")
    print(f"  (revendique par l'agent : C0=0.2212603871451519857998869119023433392467)")

    print("\n=== 3) retracage des racines a haute precision, jusqu'a eps=1.3437e-8 ===")
    fracs_eps = [mp.mpf(x) for x in
                 ('6.7186e-03', '4.0312e-03', '2.6874e-03', '1.3437e-03', '6.7186e-04',
                  '4.0312e-04', '1.3437e-04', '4.0312e-05', '1.3437e-05', '4.0312e-06',
                  '1.3437e-06', '4.0312e-07', '1.3437e-07', '1.3437e-08')]
    lignes = []
    for eps in fracs_eps:
        delta = delta_c - eps
        d3_stable = racine_dans(delta, mp.mpf('1e-12'), d3_c)
        d3_instable = racine_dans(delta, d3_c, mp.mpf('0.5'))
        gap = d3_instable - d3_stable
        ratio = gap / mp.sqrt(eps)
        lignes.append((eps, gap, ratio))
        print(f"  eps={float(eps):.6e}  gap={float(gap):.8e}  gap/sqrt(eps)={float(ratio):.8f}")

    print("\n=== 4) la table remonte-t-elle vraiment a la fin (comme mon float64), ou continue-t-elle a descendre ? ===")
    ratios = [float(r) for _, _, r in lignes]
    monotone = all(ratios[i] >= ratios[i + 1] - 1e-9 for i in range(len(ratios) - 1))
    print(f"  suite strictement decroissante (a 1e-9 pres) : {monotone}")
    print(f"  dernier point (eps={float(fracs_eps[-1]):.4e}) : ratio = {ratios[-1]:.8f}")
    print(f"  C0 analytique                                : {float(C0):.8f}")
    print(f"  ecart dernier point / C0 analytique           : {ratios[-1]-float(C0):+.2e}")

    print("\n=== 5) refit correct : ratio = C0_fit + D*eps (lineaire), vs mon ancien modele ===")
    xs = [float(e) for e, _, _ in lignes]
    ys = ratios
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    D_fit = sxy / sxx
    C0_fit = my - D_fit * mx
    print(f"  C0_fit (ordonnee a l'origine) = {C0_fit:.8f}")
    print(f"  D_fit  (pente en eps)         = {D_fit:.6f}")
    print("  residus relatifs (ratio_mesure - modele)/mesure :")
    for x, y in zip(xs, ys):
        modele = C0_fit + D_fit * x
        print(f"    eps={x:.4e}  ratio={y:.8f}  modele={modele:.8f}  residu={((y-modele)/y)*100:+.4f}%")

    print("\n=== 6) l'agent n'a pas ferme D analytiquement -- D empirique converge-t-il ? ===")
    print("    D_empirique(eps) := (gap - C0_analytique*sqrt(eps)) / eps^1.5")
    for eps, gap, ratio in lignes:
        residu_ordre_suivant = gap - C0 * mp.sqrt(eps)
        D_emp = residu_ordre_suivant / eps ** mp.mpf('1.5')
        print(f"    eps={float(eps):.4e}  D_empirique={float(D_emp):.6f}")
