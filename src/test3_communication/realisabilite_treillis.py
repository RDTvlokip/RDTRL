"""Les 3123 candidats du treillis sont-ils realisables ? Et son certificat hors-ligne.

L'enumeration de `treillis_inflation.py` laisse 3123 triples au-dessus de la valeur
trouvee par recherche. Le treillis et les contraintes de marge ne suffisent donc pas :
ce qui mord est la realisabilite conjointe par une bijection.

Ce script :

  A. cherche directement a realiser le sommet du treillis (a = b = c)
  B. verifie son certificat : masse hors ligne d'argmax = perte pure
  C. mesure la masse hors ligne des optima locaux, dont ma borne publiee
"""

import sys
import numpy as np

sys.path.insert(0, "src/test3_communication")
from loi_nulle_longue import N, matrices_information, statistiques

PAIRES = [(i, j) for i in range(N) for j in range(i + 1, N)]


def voisins(code):
    v = np.repeat(code[None, :], len(PAIRES), axis=0)
    for n, (i, j) in enumerate(PAIRES):
        v[n, i], v[n, j] = code[j], code[i]
    return v


def monter(code, objectif, pas=300):
    valeur = float(objectif(code[None, :])[0])
    for _ in range(pas):
        cand = voisins(code)
        vals = objectif(cand)
        k = int(vals.argmax())
        if vals[k] <= valeur + 1e-12:
            break
        code, valeur = cand[k].copy(), float(vals[k])
    return valeur, code


def inflation(lot):
    cm, ca, _ = statistiques(matrices_information(lot, verifier_bijectivite=False))
    return cm - ca


def main():
    print("=" * 74)
    print("A. LE SOMMET DU TREILLIS EST-IL REALISABLE ?")
    print("=" * 74)
    print("   Le meilleur triple enumere est (0,521362, 0,521362, 0,521362), qui")
    print("   donnerait 0,219295. Il faut donc une bijection ou les TROIS positions")
    print("   portent 0,5214 bit sur le MEME attribut.")
    print("   Objectif direct : maximiser le minimum des trois, pour chaque attribut.\n")

    for attribut in range(3):
        def objectif(lot, a=attribut):
            M = matrices_information(lot, verifier_bijectivite=False)
            return M[:, a, :].min(axis=1)          # min_j I(A_a ; M_j)

        generateur = np.random.default_rng(4242)
        meilleur, code = -1.0, None
        for _ in range(400):
            v, c = monter(generateur.permutation(N), objectif)
            if v > meilleur:
                meilleur, code = v, c.copy()
        M = matrices_information(code[None, :], verifier_bijectivite=False)[0]
        ligne = M[attribut, :]
        print(f"   attribut {attribut} : min_j I = {meilleur:.9f}   "
              f"ligne = ({ligne[0]:.6f}, {ligne[1]:.6f}, {ligne[2]:.6f})")
        print(f"               somme = {ligne.sum():.9f}   inflation de ce code = "
              f"{float(inflation(code[None, :])[0]):.9f}")
    print("\n   cible a atteindre pour 0,219295 : min_j I = 0,521362144")

    print()
    print("=" * 74)
    print("B. SON CERTIFICAT : LA MASSE HORS LIGNE D'ARGMAX EST UNE PERTE PURE")
    print("=" * 74)
    print("   conc est la somme des maxima de colonne : elle ne voit pas le hors-ligne.")
    print("   apparie est un maximum sur assignations : le hors-ligne ne peut que")
    print("   l'augmenter. Donc un optimum charge hors ligne est battu.\n")

    generateur = np.random.default_rng(2026)
    optima = {}
    for _ in range(600):
        v, c = monter(generateur.permutation(N), inflation)
        cle = round(v, 10)
        if cle not in optima:
            M = matrices_information(c[None, :], verifier_bijectivite=False)[0]
            gagnants = M.argmax(axis=0)
            une_ligne = len(set(gagnants.tolist())) == 1
            hors = float(M.sum() - M[gagnants, range(3)].sum()) if une_ligne else None
            if une_ligne:
                r = gagnants[0]
                hors = float(M.sum() - M[r, :].sum())
            optima[cle] = [0, une_ligne, hors, M]
        optima[cle][0] += 1

    print(f"   {len(optima)} optima locaux distincts sur 600 montees. Les six premiers :\n")
    print(f"   {'inflation':>16}{'montees':>9}{'une ligne':>11}{'masse hors ligne':>19}")
    for v in sorted(optima, reverse=True)[:6]:
        n, une, hors, _ = optima[v]
        marque = "   <- ma borne publiee" if abs(v - 0.144297209128) < 1e-9 else ""
        h = f"{hors:.12f}" if hors is not None else "n/a"
        print(f"   {v:>16.12f}{n:>9}{str(une):>11}{h:>19}{marque}")

    print()
    print("=" * 74)
    print("C. CE QUE LE CERTIFICAT PROUVE ET CE QU'IL NE PROUVE PAS")
    print("=" * 74)
    sommets = sorted(optima, reverse=True)
    propres = [v for v in sommets if optima[v][2] is not None
               and optima[v][2] < 1e-9]
    print(f"   optima a masse hors ligne nulle : {len(propres)} sur {len(optima)}")
    print(f"   le plus grand : {max(propres):.12f}")
    if len(propres) > 1:
        print(f"   le second     : {sorted(propres, reverse=True)[1]:.12f}")
        print("   -> une matrice propre ne prouve pas qu'on a fini. Une matrice")
        print("      chargee prouve qu'on n'a pas fini. Necessaire, pas suffisant,")
        print("      exactement comme il l'ecrit.")


if __name__ == "__main__":
    main()
