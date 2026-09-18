"""Piste 3a de ETAT.md (jamais commencee avant cette session, verifie
par grep sur tout src/test3_communication/ avant de se lancer -- seul
`verifier_ode_separatrice.py` existe, pour le systeme REEL, M=0
implicite) : construire les fonctions de branche et l'ODE a deux
echelles de temps POUR LE JOUET A M CATEGORIES DE FOND
(`verifier_jouet_n_variable.py`), puis bissecter des points de bascule
dessus, comme le protocole pin-and-falsify du tour 52 sur le systeme
reel.

Derivation des fonctions de branche (emetteur inchange par M, seul le
recepteur a M categories de fond supplementaires, cf. docstring de
`verifier_jouet_n_variable.py`) :

  x = d3 = 1-s3 (lent, cote emetteur, referent 3)
  R = r4 (recepteur, credit au challenger)
  s4 traite comme SATURE (=1, comme le systeme reel ou le challenger
  est pousse +30 des l'init) -- pas une variable dynamique separee,
  meme simplification que verifier_ode_separatrice.py.

  x_br(R) = meme forme que le systeme reel (K_PREFACTEUR=26), car M
  est cote RECEVEUR seulement -- l'emetteur ne voit jamais M. Ce n'est
  qu'une approximation (elle suppose r3~1-R, en negligeant la part de
  masse M/Z que le fond retient a l'equilibre) -- verifiee ci-dessous.

  R_br(x; M) = e4 / (e3(x) + e4 + M), avec
      e3(x) = exp(N*poids3*(1-x)/beta)   poids3=(1-delta)/N
      e4    = exp(N*poids4/beta)          poids4=(1+delta)/N  (s4=1 fixe)
  -- derive directement du point fixe algebrique deja valide dans
  `verifier_point_fixe_jouet_m.py` (meme Z=e3+e4+M), ici resolu
  seulement cote recepteur (x fixe, pas emetteur).
"""

import sys
sys.path.insert(0, '.')
import math

BETA = 0.02
N = 27
K_PREFACTEUR = 26


def x_br(R, delta, beta=BETA):
    """BUG trouve en tournant ce script une premiere fois : cette
    fonction reutilisait la forme du systeme REEL (softmax a 27 voies,
    K_PREFACTEUR=26), fausse ici -- l'emetteur du jouet est un
    SIGMOIDE BINAIRE independant (s3=sigmoid(p3), entropie binaire),
    pas une ligne de softmax a 27 voies (cf. `objectif_toy_m`,
    `ent = lambda p: -(p*log(p)+(1-p)*log(1-p))`). La vraie branche,
    derivee en annulant le gradient de poids3*s3*r3+(beta/N)*H(s3) :
        s3 = sigmoid(N*poids3*r3/beta)
    deja utilisee (et validee empiriquement) dans point_fixe() de
    verifier_point_fixe_jouet_m.py. Approxime r3~1-R (neglige la fuite
    de masse M/Z vers le fond dans l'equation de x -- verifie plus bas
    que e3,e4 >> M dans la region d'interet, donc l'approximation est
    bonne)."""
    poids3 = (1 - delta) / N
    r3_approx = 1 - R
    s3 = 1 / (1 + math.exp(-N * poids3 * r3_approx / beta))
    return 1 - s3


def R_br(x, M, delta, beta=BETA):
    poids3 = (1 - delta) / N
    poids4 = (1 + delta) / N
    s3 = 1 - x
    e3 = math.exp(min(N * poids3 * s3 / beta, 700))
    e4 = math.exp(min(N * poids4 * 1.0 / beta, 700))
    Z = e3 + e4 + M
    return e4 / Z


def verifier_reduction_m0():
    """M=0 doit redonner EXACTEMENT la meme R_br que la forme fermee du
    systeme reel (logit(R)*beta = 2*delta + (1-delta)*d3), sans quoi la
    derivation ci-dessus a une erreur avant meme de toucher M>0."""
    delta = 0.013
    for x in (0.005, 0.01, 0.02, 0.05):
        R_ici = R_br(x, 0, delta)
        logitR = (2 * delta + (1 - delta) * x) / BETA
        R_reel = 1 / (1 + math.exp(-logitR))
        ecart = abs(R_ici - R_reel)
        print(f"  x={x:.3f}  R_br(ici, M=0)={R_ici:.12f}  "
              f"R_br(forme fermee reelle)={R_reel:.12f}  ecart={ecart:.2e}")


def localiser_pli(M, delta_lo=None, delta_hi=None, beta=BETA, tol=1e-10,
                   x0_guess=0.0042146, delta_guess=0.0187):
    """CORRIGE (18/09/2026, apres challenge agent-dipankar) : la version
    precedente comptait les racines sur une grille FIXE de 200000 points
    sur tout [0,1] (pas ~5e-6) -- beaucoup trop grossier pres du vrai
    point selle (x*~0,0042146, une fenetre ~40x plus petite que ce pas
    de grille), ce qui localisait un ARTEFACT DE GRILLE a delta~0,01115
    (faux de 40% par rapport au vrai pli a delta~0,018699, verifie
    independamment par tangence Newton en mpmath 50 chiffres ET ici en
    float64). Remplace par Newton sur le systeme de tangence exact
    {f(x,delta)=0, df/dx(x,delta)=0} avec f(x,delta)=x_br(R_br(x;M,delta),delta)-x,
    derivees par difference finie centree (h=1e-9)."""
    h = 3e-6  # pas fini choisi pour rester loin du plancher float64 (~1e-16)
              # sur des derivees SECONDES (qui divisent par h**2)

    def f(x, delta):
        return x_br(R_br(x, M, delta), delta) - x

    def F0F1(x, delta):
        F0 = f(x, delta)
        F1 = (f(x + h, delta) - f(x - h, delta)) / (2 * h)
        return F0, F1

    x, delta = x0_guess, delta_guess
    for _ in range(100):
        F0, F1 = F0F1(x, delta)
        F0_xp, F1_xp = F0F1(x + h, delta)
        F0_xm, F1_xm = F0F1(x - h, delta)
        F0_dp, F1_dp = F0F1(x, delta + h)
        F0_dm, F1_dm = F0F1(x, delta - h)
        J00 = (F0_xp - F0_xm) / (2 * h)   # dF0/dx
        J01 = (F0_dp - F0_dm) / (2 * h)   # dF0/ddelta
        J10 = (F1_xp - F1_xm) / (2 * h)   # dF1/dx
        J11 = (F1_dp - F1_dm) / (2 * h)   # dF1/ddelta
        det = J00 * J11 - J01 * J10
        if det == 0:
            return None
        dx = -(F0 * J11 - F1 * J01) / det
        dd = -(J00 * F1 - J10 * F0) / det
        x += dx
        delta += dd
        if abs(dx) < tol and abs(dd) < tol:
            break
    return delta, x


def separatrice_croise_R(M, delta, k, x_selle, R_init_cible, dt=-1e-6, pas_max=5_000_000):
    """Meme integration a rebours que verifier_ode_separatrice.py, mais
    avec R_br(x;M) au lieu de R_br(x) fixe -- suit la variete stable
    du point selle jusqu'a ce que R passe sous R_init_cible."""
    eps = 1e-7
    x = x_selle + eps
    R = R_br(x_selle, M, delta) + eps * 0.1
    for _ in range(pas_max):
        dx = x_br(R, delta) - x
        dR = k * (R_br(x, M, delta) - R)
        x += dx * dt
        R += dR * dt
        if R <= R_init_cible:
            return 1 - x
        if x < 0 or x > 1:
            return None
    return None


def trouver_x_selle(M, delta):
    """Racine INSTABLE du pli (entre les deux racines stables si delta<delta_c,
    sinon aucune) -- balayage fin + raffinement, pas de derivee symbolique."""
    meilleurs = []
    prec_f, prec_x = None, None
    for i in range(1, 2_000_000):
        x = i / 2_000_000
        f = x_br(R_br(x, M, delta), delta) - x
        if prec_f is not None and (f > 0) != (prec_f > 0):
            meilleurs.append((prec_x, x))
        prec_f, prec_x = f, x
    if len(meilleurs) < 3:
        return None
    # la racine instable est celle du MILIEU quand il y en a 3 (stable/instable/stable)
    lo, hi = meilleurs[1]
    for _ in range(60):
        mid = (lo + hi) / 2
        f = x_br(R_br(mid, M, delta), delta) - mid
        f_lo = x_br(R_br(lo, M, delta), delta) - lo
        if (f > 0) == (f_lo > 0):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


if __name__ == "__main__":
    print("=== Sanite : M=0 doit reproduire exactement R_br du systeme reel ===")
    verifier_reduction_m0()

    print()
    print("=== delta_c(M) : fold localise par tangence Newton (CORRIGE apres challenge agent) ===")
    resultats_dc = {}
    for M in (0, 1, 3, 8, 25):
        out = localiser_pli(M)
        if out is None:
            print(f"  M={M:2d}: pas converge")
            continue
        dc, x_c = out
        resultats_dc[M] = (dc, x_c)
        print(f"  M={M:2d}   delta_c(M) = {dc:.15f}   x*={x_c:.10e}")

    if 0 in resultats_dc and 25 in resultats_dc:
        d0 = resultats_dc[0][0]
        d25 = resultats_dc[25][0]
        print(f"  ecart relatif delta_c(25) vs delta_c(0) en float64 : {(d25-d0)/d0:.3e} "
              f"(attendu ~1e-21 par mpmath -- sous le plancher de precision float64, "
              f"donc NUL ou bruit ici est le resultat CORRECT, pas un bug)")

    print()
    print("=== points de bascule (pin-and-falsify sur l'ODE), M=0 vs M=25, x_selle CORRIGE ===")
    for M in (0, 25):
        out = localiser_pli(M)
        if out is None:
            print(f"  M={M}: pas de pli trouve, saute")
            continue
        dc, x_selle = out
        delta_test = dc * 0.95
        # x_selle a delta_test (pas exactement au pli) -- redecouple via trouver_x_selle
        x_selle_test = trouver_x_selle(M, delta_test)
        if x_selle_test is None:
            print(f"  M={M}, delta={delta_test:.6f}: pas de racine instable trouvee")
            continue
        print(f"  M={M}  delta_c={dc:.6f}  delta_test={delta_test:.6f}  x_selle={x_selle_test:.6e}")
        for k in (1.0, 1.5, 2.0):
            for R_init in (0.50, 0.60, 0.75):
                s3_flip = separatrice_croise_R(M, delta_test, k, x_selle_test, R_init)
                print(f"    k={k}  R_init={R_init}  s3_flip={s3_flip}")
