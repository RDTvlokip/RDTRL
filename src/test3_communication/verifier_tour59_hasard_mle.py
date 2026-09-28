"""Tour 59 (29/09/2026) : le taux d'echappement lambda(delta) de la falaise
a eps 1e-10 suit-il une exponentielle ou une loi de puissance en
(delta - delta*) ? EXPLORATOIRE (post hoc : l'idee vient de la lecture de
verifier_tour59_hasard_echappement.py) ; la mise a l'epreuve est le test
hors echantillon precommis dans le carnet.

Donnees : runs de lancer_tour59_hasard.sh (10 phases x 6 delta, censure a
30 000 pas). Chaque run : taux constant lambda(delta) sur toute sa duree ;
log-vraisemblance  log(lambda) - lambda t  (echappement a t)  ou
-lambda T (censure a T). Deux modeles :
  exp   lambda = exp(a + b x)                        (2 parametres)
  puis  lambda = c x^g pour x = delta - delta* > 0   (3 parametres)
x = offset - pli. Maximisation (Nelder-Mead depuis plusieurs departs),
AIC, et prediction des taux hors echantillon (6,3e-9 ; 9e-9 ; 12e-9),
plus le nombre d'echappements attendu sur une exposition donnee.
"""

import glob
import re
import numpy as np
from scipy.optimize import minimize


def lire(prefixe="hz"):
    donnees = []
    for f in glob.glob(f"D:/tmp/rdtrl_t59_{prefixe}_k*_1e-10_*.txt"):
        m = re.search(r"_1e-10_([0-9.e+-]+)\.txt$", f)
        txt = open(f).read()
        if not m or "bascule" not in txt:
            continue
        off = float(m.group(1))
        pas = int(re.search(r"pas=(\d+)", txt).group(1))
        b = int(re.search(r"bascule=(-?\d+)", txt).group(1))
        donnees.append((off, b if b >= 0 else None, pas))
    return donnees


def loglik(lam_fonction, donnees):
    ll = 0.0
    for off, b, pas in donnees:
        lam = lam_fonction(off)
        if lam <= 0 or not np.isfinite(lam):
            return -1e18
        ll += (np.log(lam) - lam * b) if b is not None else (-lam * pas)
    return ll


def ajuster_exp(donnees):
    f = lambda p: -loglik(lambda x: np.exp(p[0] + p[1] * x * 1e9), donnees)
    meilleur = None
    for a0 in (-12, -9):
        for b0 in (0.5, 1.5, 3):
            r = minimize(f, [a0, b0], method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000})
            if meilleur is None or r.fun < meilleur.fun:
                meilleur = r
    return meilleur


def ajuster_puissance(donnees):
    xmin = min(o for o, _, _ in donnees)

    def f(p):
        logc, g, dstar = p  # dstar en unites 1e-9
        if dstar >= xmin * 1e9 - 1e-6 or g <= 0:
            return 1e18
        return -loglik(lambda x: np.exp(logc) * ((x * 1e9 - dstar)) ** g, donnees)

    meilleur = None
    for d0 in (4.0, 5.0, 5.8):
        for g0 in (1.0, 2.5, 4.0):
            r = minimize(f, [-9.0, g0, d0], method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 8000})
            if meilleur is None or r.fun < meilleur.fun:
                meilleur = r
    return meilleur


if __name__ == "__main__":
    import sys
    # argument optionnel : prefixe des fichiers (hz par defaut) ; plusieurs prefixes separes par des virgules
    D = [x for p in (sys.argv[1] if len(sys.argv) > 1 else "hz").split(",") for x in lire(p)]
    print(f"{len(D)} runs, {sum(1 for _, b, _ in D if b is not None)} echappements")
    re_ = ajuster_exp(D)
    rp = ajuster_puissance(D)
    ll_e, ll_p = -re_.fun, -rp.fun
    a, b = re_.x
    logc, g, dstar = rp.x
    print(f"exponentielle : lambda = exp({a:.3f} + {b:.3f} * x/1e-9)   pente d ln(lambda)/d delta = {b * 1e9:.3e}   "
          f"logL={ll_e:.2f}  AIC={2 * 2 - 2 * ll_e:.2f}")
    print(f"puissance     : lambda = {np.exp(logc):.3e} * (x/1e-9 - {dstar:.3f})^{g:.3f}   "
          f"delta* = pli + {dstar:.3f}e-9   exposant {g:.2f}   logL={ll_p:.2f}  AIC={2 * 3 - 2 * ll_p:.2f}")
    print(f"difference d'AIC (exp - puissance) = {(4 - 2 * ll_e) - (6 - 2 * ll_p):+.2f}   (>0 : la puissance est preferee ; >6 : nettement)")
    print("\nprediction hors echantillon : offset   exp        puissance")
    for x in (6.0e-9, 6.3e-9, 6.5e-9, 9e-9, 1.2e-8):
        le = np.exp(a + b * x * 1e9)
        lp = np.exp(logc) * (x * 1e9 - dstar) ** g if x * 1e9 > dstar else 0.0
        print(f"                          {x:.2e}   {le:.3e}   {lp:.3e}")
