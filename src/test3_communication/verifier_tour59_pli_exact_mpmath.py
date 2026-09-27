"""Tour 59 : le pli par resolution EXACTE du systeme G(u, delta) = 0 et
dG/du = 0 (mpmath, 40 chiffres), au lieu de la bissection sur un
comptage de racines en grille (verifier_tour59_pli_concurrents.py), qui
peut perdre deux racines proches AVANT leur fusion et donc sous-estimer
le pli. Motif : les seuils dynamiques des deux collisions tombent tous
deux dans (pli_grille + 6e-10, pli_grille + 1e-9)."""

from mpmath import mp, mpf, exp, diff, findroot

mp.dps = 40
B = mpf("0.02")


def G(u, d, C=26):
    r3 = 1 / (1 + exp(-u)); r4 = 1 / (1 + exp(u))
    s3 = 1 / (1 + C * exp(-(1 - d) * r3 / B)); s4 = 1 / (1 + C * exp(-(1 + d) * r4 / B))
    return -u - ((1 + d) * s4 - (1 - d) * s3) / B


if __name__ == "__main__":
    F_GRILLE = mpf("0.013437210041901196")
    for C in (25, 26, 27, 28, 29):
        sol = findroot(lambda u, d: [G(u, d, C), diff(lambda x: G(x, d, C), u)],
                       [mpf("-1.47"), mpf("0.01344")])
        u, d = sol
        r3 = 1 / (1 + exp(-u))
        print(f"C={C}  pli exact delta_c = {mp.nstr(d, 16)}   u = {mp.nstr(u, 12)}   X3 = {mp.nstr((1 - d) * r3 / B, 10)}"
              + (f"   (pli_grille + {mp.nstr(d - F_GRILLE, 4)})" if C == 26 else ""))
