"""Tour 52 de dipankarsarkar : sa forme fermee (les points fixes) ne
dessine pas la separatrice -- elle depend de k, le rapport des vitesses
de relaxation recepteur/emetteur, absent de ses deux equations
d'origine. Il fournit :
  dx/dt = x_br(R) - x          (x = d3 = 1-s3)
  dR/dt = k * (R_br(x) - R)
et une table de points de bascule a R_init=0,60 pour k=0.5..2.0, plus
un k*=0,428762 sous lequel la separatrice n'atteint plus jamais R=0.

Ce script integre cette ODE independamment (Euler explicite a rebours
depuis le point selle, le long de la variete stable) pour verifier sa
table SANS la prendre pour acquise -- regle CLAUDE.md 5bis.
"""

import math

BETA = 0.02
DELTA = 0.013
K_PREFACTEUR = 26
D3_SELLE = 5.70024e-3  # racine instable a delta=0.013, cf. tour 51


def R_br(d3, delta=DELTA, beta=BETA):
    logitR = (2 * delta + (1 - delta) * d3) / beta
    return 1 / (1 + math.exp(-logitR))


def x_br(R, delta=DELTA, beta=BETA):
    A = math.exp(-(1 - delta) * (1 - R) / beta)
    return K_PREFACTEUR * A / (1 + K_PREFACTEUR * A)


def separatrice_croise_R(k, R_cible, dt=-1e-5, pas_max=3_000_000):
    """Suit la variete stable du point selle a rebours (dt<0) jusqu'a
    ce que R passe sous R_cible ; renvoie s3=1-x a cet instant."""
    eps = 1e-7
    x = D3_SELLE + eps
    R = R_br(D3_SELLE) + eps * 0.1
    for _ in range(pas_max):
        dx = x_br(R) - x
        dR = k * (R_br(x) - R)
        x += dx * dt
        R += dR * dt
        if R <= R_cible:
            return 1 - x
        if x < 0 or x > 1:
            return None
    return None


def separatrice_R_min(k, dt=-1e-5, pas_max=3_000_000):
    """Le R le plus bas atteint par la variete stable -- sous k*, elle
    ne redescend jamais jusqu'a R=0."""
    eps = 1e-7
    x = D3_SELLE + eps
    R = R_br(D3_SELLE) + eps * 0.1
    R_min = R
    for _ in range(pas_max):
        dx = x_br(R) - x
        dR = k * (R_br(x) - R)
        x += dx * dt
        R += dR * dt
        R_min = min(R_min, R)
        if x < 0 or x > 1 or R < 0 or R > 1:
            break
    return R_min


if __name__ == "__main__":
    print("=== table k -> point de bascule a R_init=0.60 (verification independante) ===")
    for k in (0.5, 0.75, 1.0, 1.5, 2.0):
        s3 = separatrice_croise_R(k, 0.60)
        print(f"  k={k:<5}  s3_flip@0.60={s3:.6f}")

    print("\n=== k* : seuil sous lequel la separatrice ne touche jamais R=0 ===")
    for k in (0.25, 0.30, 0.40, 0.42, 0.428762, 0.43, 0.50):
        print(f"  k={k:<9}  R_min={separatrice_R_min(k):.6e}")
