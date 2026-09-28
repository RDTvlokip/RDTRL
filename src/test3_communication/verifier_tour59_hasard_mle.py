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


PLI = 0.0134372100660973


def lire(prefixe="hz"):
    """prefixe : hz / hzo (lancer_tour59_hasard.sh), e5 (E5 : 10 phases a 6,5e-9, fichiers
    e5k<k>), e8 (E8 : un seul run de 300 000 pas a 6,0e-9)."""
    donnees = []
    if prefixe == "e8":
        f = "D:/tmp/rdtrl_t59_e8_300k.txt"
        txt = open(f).read()
        off = float(re.search(r"delta=([0-9.]+)", txt).group(1)) - PLI
        pas = int(re.search(r"pas=(\d+)", txt).group(1))
        b = int(re.search(r"bascule=(-?\d+)", txt).group(1))
        return [(round(off, 12), b if b >= 0 else None, pas)]
    motif = "D:/tmp/rdtrl_t59_e5k*_1e-10_*.txt" if prefixe == "e5" else f"D:/tmp/rdtrl_t59_{prefixe}_k*_1e-10_*.txt"
    for f in glob.glob(motif):
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
        if not np.isfinite(lam) or lam < 0:
            return -1e18
        if lam == 0:
            # taux nul (sous delta*) : un run qui tient est parfaitement legal, un echappement est impossible
            if b is not None:
                return -1e18
            continue
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
        if dstar >= 20.0 or dstar < 0.0 or g <= 0:
            return 1e18
        return -loglik(lambda x: np.exp(logc) * (x * 1e9 - dstar) ** g if x * 1e9 > dstar else 0.0, donnees)

    meilleur = None
    for d0 in (4.0, 5.0, 5.8):
        for g0 in (1.0, 2.5, 4.0):
            r = minimize(f, [-9.0, g0, d0], method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 8000})
            if meilleur is None or r.fun < meilleur.fun:
                meilleur = r
    return meilleur


def profil_puissance(donnees, dstars):
    """Vraisemblance profilee : pour chaque delta* fixe (unites 1e-9), maximum sur (c, gamma)."""
    sortie = []
    for ds in dstars:
        f = lambda p: -loglik(lambda x: np.exp(p[0]) * (x * 1e9 - ds) ** p[1] if x * 1e9 > ds else 0.0, donnees) if p[1] > 0 else 1e18
        meilleur = None
        for g0 in (1.5, 3.5, 6.0):
            r = minimize(f, [-9.0, g0], method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 6000})
            if meilleur is None or r.fun < meilleur.fun:
                meilleur = r
        sortie.append((ds, -meilleur.fun, meilleur.x[1]))
    return sortie


def ajuster_plancher_puissance(donnees):
    """lambda = l0 + c (x - x*)^g pour x > x*, l0 sinon (5 parametres -> 4 : l0, c, g, x*)."""
    xmin = min(o for o, _, _ in donnees)

    def f(p):
        logl0, logc, g, dstar = p
        if g <= 0 or dstar >= 20.0 or dstar < 0.0:
            return 1e18
        def lam(x):
            xx = x * 1e9 - dstar
            return np.exp(logl0) + (np.exp(logc) * xx ** g if xx > 0 else 0.0)
        return -loglik(lam, donnees)

    meilleur = None
    for l0 in (-14.0, -13.0):
        for d0 in (5.5, 6.2):
            for g0 in (1.5, 3.0):
                r = minimize(f, [l0, -9.0, g0, d0], method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 12000})
                if meilleur is None or r.fun < meilleur.fun:
                    meilleur = r
    return meilleur


def ajuster_plancher_exp(donnees):
    """lambda = l0 + exp(a + b x) (3 parametres)."""
    def f(p):
        logl0, a, b = p
        return -loglik(lambda x: np.exp(logl0) + np.exp(a + b * x * 1e9), donnees)
    meilleur = None
    for l0 in (-14.0, -13.0):
        for a0 in (-14.0, -12.0):
            for b0 in (0.5, 1.5):
                r = minimize(f, [l0, a0, b0], method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 8000})
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
    rfe = ajuster_plancher_exp(D)
    rfp = ajuster_plancher_puissance(D)
    ll_fe, ll_fp = -rfe.fun, -rfp.fun
    l0, a2, b2 = rfe.x
    print(f"plancher+exp  : lambda = {np.exp(l0):.2e} + exp({a2:.2f} + {b2:.3f} x/1e-9)   logL={ll_fe:.2f}  AIC={2 * 3 - 2 * ll_fe:.2f}")
    print(f"plancher+puiss: lambda = {np.exp(rfp.x[0]):.2e} + {np.exp(rfp.x[1]):.2e} (x/1e-9 - {rfp.x[3]:.2f})^{rfp.x[2]:.2f}   "
          f"logL={ll_fp:.2f}  AIC={2 * 4 - 2 * ll_fp:.2f}")
    print("\nprofil de vraisemblance de la loi de puissance (delta* fixe, max sur c et gamma) ; IC 95 % : logL >= max - 1,92")
    prof = profil_puissance(D, [4.0, 4.5, 5.0, 5.3, 5.5, 5.7, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5])
    lmax = max(l for _, l, _ in prof)
    for ds, l, gg in prof:
        print(f"   delta* = pli + {ds:.1f}e-9   logL = {l:8.2f}   (max - {lmax - l:5.2f})   gamma = {gg:.2f}   {'dans l IC' if lmax - l <= 1.92 else ''}")
    print("\nprediction hors echantillon : offset   exp        puissance")
    for x in (6.0e-9, 6.3e-9, 6.5e-9, 9e-9, 1.2e-8):
        le = np.exp(a + b * x * 1e9)
        lp = np.exp(logc) * (x * 1e9 - dstar) ** g if x * 1e9 > dstar else 0.0
        print(f"                          {x:.2e}   {le:.3e}   {lp:.3e}")
