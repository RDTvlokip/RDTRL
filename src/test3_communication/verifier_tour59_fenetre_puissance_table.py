"""Tour 59 (29/09/2026) : verification (V3) de l'affirmation du second agent
style dipankar : la loi de puissance lambda = c (x - x*)^gamma que j'ai
ajustee sur le reseau complet est un artefact de la FENETRE d'ajustement.

Entrees NON verifiees par moi : le tableau (x, evenements, exposition) de
sa reduction (port numba de red2.py), recopie de son rapport du 29/09.
Ce que je verifie ici, avec mon propre code : (1) les parametres (gamma,
x*) selon la fenetre, par maximum de vraisemblance de Poisson
(n_i ~ Poisson(lambda_i E_i)) ; (2) ce que MA loi publiee (1,809e-5,
x*=5,72, gamma=3,48) prevoit en evenements dans ses lignes a 5,25-6,0e-9.
x en unites de 1e-9 au-dessus du pli.
"""

import numpy as np
from scipy.optimize import minimize

TABLE = [  # x, evenements, exposition (pas)
    (5.00, 41, 2.45e10), (5.25, 96, 1.03e10), (5.50, 100, 2.5e9), (5.75, 100, 5.0e8),
    (6.00, 136, 2.3e8), (6.10, 184, 1.5e8), (6.30, 200, 7.1e7), (6.50, 200, 3.0e7),
    (6.80, 200, 1.0e7), (7.00, 200, 5.4e6), (7.50, 200, 1.8e6), (8.00, 200, 1.2e6),
    (10.0, 200, 4.5e5),
]


def loglik_poisson(lam_fn, pts):
    ll = 0.0
    for x, n, E in pts:
        lam = lam_fn(x)
        if lam < 0 or not np.isfinite(lam):
            return -1e18
        if lam == 0:
            if n > 0:
                return -1e18
            continue
        ll += n * np.log(lam * E) - lam * E
    return ll


def ajuster(pts):
    xmin = min(x for x, _, _ in pts)
    def f(p):
        logc, g, xs = p
        if g <= 0 or xs < 0.0 or xs >= xmin:
            return 1e18
        return -loglik_poisson(lambda x: np.exp(logc) * (x - xs) ** g if x > xs else 0.0, pts)
    best = None
    for xs0 in (max(0.5, xmin - 3.0), xmin - 1.0, xmin - 0.3):
        for g0 in (2.0, 4.0, 10.0):
            r = minimize(f, [-11.0, g0, xs0], method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-9, "maxiter": 20000})
            if best is None or r.fun < best.fun:
                best = r
    return best


def deviance(lam_fn, pts):
    d = 0.0
    for x, n, E in pts:
        mu = lam_fn(x) * E
        d += 2 * ((n * np.log(n / mu) if n > 0 else 0.0) - (n - mu))
    return d


def seuil_selon_horizon():
    """delta au-dessus du pli (1e-9) pour lequel lambda(delta) = 1/T, interpolation log-log de la table :
    la 'probabilite de casser dans T pas' vaut 1 - exp(-1) = 63 % en ce point."""
    xs = np.array([x for x, _, _ in TABLE])
    lam = np.array([n / E for _, n, E in TABLE])
    print("horizon T (pas)   delta_1/T - pli  (1e-9)   [lambda = 1/T ; 63 % de casser dans T]")
    for T in (1e4, 6e4, 1e5, 3e5, 1e6, 1e7, 1e8, 1e9):
        cible = np.log(1.0 / T)
        x = np.interp(cible, np.log(lam), xs)
        print(f"   {T:9.0e}         {x:6.2f}")


if __name__ == "__main__":
    fenetres =[("[6,1 ; 8,0]  (ma fenetre)", 6.1, 8.0), ("[6,1 ; 10]", 6.1, 10.0), ("[5,25 ; 10]", 5.25, 10.0), ("[5,0 ; 7,0]", 5.0, 7.0)]
    print("fenetre                       points   gamma    x* (1e-9)   deviance/ddl        son gamma / x*")
    ses = {"[6,1 ; 8,0]  (ma fenetre)": "3,50 / 5,56", "[6,1 ; 10]": "2,26 / 5,87", "[5,25 ; 10]": "4,16 / 5,00", "[5,0 ; 7,0]": "12,4 / 3,43"}
    for nom, a, b in fenetres:
        pts = [t for t in TABLE if a - 1e-9 <= t[0] <= b + 1e-9]
        r = ajuster(pts)
        logc, g, xs = r.x
        dev = deviance(lambda x: np.exp(logc) * (x - xs) ** g if x > xs else 0.0, pts)
        print(f"{nom:28s}  {len(pts):4d}   {g:6.2f}   {xs:7.3f}     {dev:8.1f} / {len(pts) - 3:2d}       {ses[nom]}")
    print("\nMA loi publiee (c=1,809e-5, x*=5,72, gamma=3,48) : evenements attendus dans SES lignes basses")
    for x, n, E in TABLE:
        if x <= 6.0:
            mu = 1.809e-5 * (x - 5.72) ** 3.48 * E if x > 5.72 else 0.0
            print(f"   x={x:5.2f}   observes {n:4d}   attendus {mu:8.3f}")
    print()
    seuil_selon_horizon()
