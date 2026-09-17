"""Piste 1 de ETAT.md : refitter le coefficient de ralentissement avec de
vraies donnees mesurees, pas seulement les 3 points cites par dipankar
(verifier_coefficient_ralentissement.py, tour 51/52).

Motif de mefiance (regle 5bis / « ne jamais suivre le standard ») : le
point le plus proche de delta_c dans ses 3 points a un residu (5,943e-4
en unites s3) plus PETIT que la taille typique d'une excursion sous Adam
complet pres du pli (~2e-3 en unites R, cf. CARNET §7.64 -- l'episode ou
un instantane a pas=60 000 avait justement mal lu la branche a cause
d'une excursion). Si ses 3 points venaient d'instantanes uniques a un pas
rond plutot que d'une valeur stabilisee, le point le plus proche du pli
-- celui dont le residu attendu est justement le plus petit, donc le plus
sensible au bruit -- pourrait etre domine par du bruit d'excursion, pas
par le vrai ecart loi-molle/branche.

Ici : (1) delta_c re-localise par recherche de racines pure (fusion des
deux racines graduee/instable de son systeme ferme), independamment du
`DELTA_C=0.0134295` code en dur dans les anciens scripts (bissection
grossiere) ; (2) a chaque delta, mesure de R par l'optimiseur HYBRIDE
(Adam emetteur seul + SGD recepteur seul, verifie entrainer reellement
le referent 3 -- H15/tour 52) sur une FENETRE de fin de run, valeur
retenue = mediane (robuste a une excursion isolee), pas un instantane ;
(3) residu = R_mesure - R_molle(delta), R_molle = sigmoid(2*delta/beta)
(la meme « loi molle » que verifier_jouet_2_referents.py) ; (4) ajustement
log-log de l'exposant reel, compare a 0,5 (noeud-col 1D pur) et au
coefficient moyen ~0,2212-0,2322 des 3 points publies.
"""

import sys
sys.path.insert(0, '.')
import math
import statistics
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import N, activer
from verifier_prior_asymetrique import objectif_pondere

ADAM_EPS = 1e-10
LR_ADAM_E = 0.05
LR_SGD_R = 50.0
CHECK_TOUS = 40
FENETRE_FRACTION = 0.2


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def f_pli(d3, delta):
    """Systeme couple de dipankar (verifie independamment au tour 51,
    CARNET §7.64) : difference entre les deux membres de l'equation
    emetteur, R substitue via l'equation recepteur."""
    R = sigmoid((2 * delta + (1 - delta) * d3) / BETA)
    lhs = d3 / (26 * (1 - d3))
    rhs = math.exp(-(1 - delta) * (1 - R) / BETA)
    return lhs - rhs


def racines(delta, d3_lo=1e-9, d3_hi=0.5, n_scan=200_000):
    """Balaye d3 et renvoie TOUTES les racines trouvees (par changement de
    signe), triees par d3 croissant."""
    pas = (d3_hi - d3_lo) / n_scan
    prev_d3 = d3_lo
    prev_f = f_pli(d3_lo, delta)
    trouvees = []
    for i in range(1, n_scan + 1):
        d3 = d3_lo + i * pas
        fv = f_pli(d3, delta)
        if prev_f == 0:
            trouvees.append(prev_d3)
        elif (fv < 0) != (prev_f < 0):
            lo, hi = prev_d3, d3
            flo = prev_f
            for _ in range(200):
                mid = (lo + hi) / 2
                fm = f_pli(mid, delta)
                if (fm < 0) == (flo < 0):
                    lo, flo = mid, fm
                else:
                    hi = mid
            trouvees.append((lo + hi) / 2)
        prev_d3, prev_f = d3, fv
    return trouvees


def localiser_delta_c(lo=0.0130, hi=0.0140, tol=1e-9):
    """delta_c = frontiere ou le nombre de racines graduees/instables passe
    de 2 (en dessous, stable+instable distinctes) a 0 (au-dessus, fusion
    +effondrement seul). Bissection independante du 0,0134295 code en dur
    dans les anciens scripts."""
    def n_racines_graduees(delta):
        r = racines(delta)
        # on exclut la racine d'effondrement pres de 26/27=0.962963
        return len([x for x in r if x < 0.5])

    assert n_racines_graduees(lo) == 2, "lo devrait etre sous delta_c (2 racines)"
    assert n_racines_graduees(hi) == 0, "hi devrait etre au-dessus de delta_c (0 racine)"
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if n_racines_graduees(mid) >= 1:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def mesurer_hybride(delta, pas_max):
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + delta) / N
    poids[3] = (1.0 - delta) / N
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    opt_e = torch.optim.Adam(e.p, lr=LR_ADAM_E, eps=ADAM_EPS)
    opt_r = torch.optim.SGD(r.p, lr=LR_SGD_R)
    R_trace = []
    for pas in range(pas_max):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt_e.zero_grad()
        opt_r.zero_grad()
        (-j).backward()
        opt_e.step()
        opt_r.step()
        if pas % CHECK_TOUS == 0:
            with torch.no_grad():
                s_, rr_ = e.loi(), r.loi()
                R_trace.append((s_[4, 10] * rr_[10, 4]).item())
    n_fenetre = max(3, int(len(R_trace) * FENETRE_FRACTION))
    fenetre = R_trace[-n_fenetre:]
    return statistics.median(fenetre), statistics.pstdev(fenetre), min(fenetre), max(fenetre)


if __name__ == "__main__":
    torch.set_printoptions(precision=6)

    print("=== relocalisation independante de delta_c (fusion des racines, pas de training) ===")
    DELTA_C = localiser_delta_c()
    print(f"  delta_c = {DELTA_C:.9f}  (ancien code en dur dans les scripts precedents : 0.0134295 / 0.013437210)")

    print("\n=== mesure R via optimiseur hybride, mediane sur fenetre de fin de run ===")
    fracs = (0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01, 0.003, 0.001)
    PAS_MAX = 80_000
    lignes = []
    for frac in fracs:
        delta = DELTA_C * (1 - frac)
        gap_dc = DELTA_C - delta
        R_molle = sigmoid(2 * delta / BETA)
        R_med, R_std, R_min, R_max = mesurer_hybride(delta, PAS_MAX)
        gap_R = R_med - R_molle
        lignes.append((delta, gap_dc, R_molle, R_med, gap_R, R_std, R_max - R_min))
        print(f"  delta={delta:.7f}  dc-delta={gap_dc:.4e}  R_molle={R_molle:.6f}  "
              f"R_mediane={R_med:.6f}  gap_R={gap_R:.4e}  std_fenetre={R_std:.2e}  "
              f"amplitude_fenetre={R_max-R_min:.2e}")

    print("\n=== coefficients individuels gap_R / sqrt(dc-delta) ===")
    coeffs = []
    for delta, gap_dc, R_molle, R_med, gap_R, R_std, ampl in lignes:
        c = gap_R / math.sqrt(gap_dc)
        coeffs.append(c)
        print(f"  delta={delta:.7f}  coefficient={c:.6f}")

    print(f"\n  moyenne des {len(coeffs)} points : {sum(coeffs)/len(coeffs):.6f}")
    print(f"  min={min(coeffs):.6f}  max={max(coeffs):.6f}  "
          f"derive (max-min)/min = {(max(coeffs)-min(coeffs))/min(coeffs)*100:.1f} %")
    print("  (a comparer, MEME UNITE R, pas s3 -- pas de comparaison directe possible avec le")
    print("   0,2212 de dipankar/verifier_coefficient_ralentissement.py qui est en unites s3)")

    print("\n=== ajustement log-log de l'exposant reel (gap_R ~ C * (dc-delta)^p) ===")
    xs = [math.log(gap_dc) for _, gap_dc, *_ in lignes]
    ys = [math.log(abs(gap_R)) for *_, gap_R, _, _ in lignes]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    pente = sxy / sxx
    intercept = my - pente * mx
    print(f"  exposant mesure (pente log-log) = {pente:.4f}  (predit par un noeud-col 1D pur : 0.5)")
    print(f"  coefficient implique (exp(intercept)) = {math.exp(intercept):.6f}")
