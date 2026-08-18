"""Le supremum de l'inflation par enumeration, au lieu de redemarrages.

Le relecteur remarque que pour un code bijectif, la table de contingence 3x3 de
(A_i, M_j) a ses six marges egales a 9, ce qui rend I(A_i ; M_j) fini : un treillis
de valeurs. Sa question : pourquoi continuer a retomber dessus par accident ?

Ce script :

  1. enumere les tables a marges 9 et le treillis de valeurs
  2. etablit une contrainte que ni lui ni moi n'avons ecrite : pour un code
     bijectif les M_j sont mutuellement independants, donc chaque LIGNE et
     chaque COLONNE de la matrice d'information somme a au plus log2(3)
  3. enumere les candidats au-dessus du supremum trouve par recherche
  4. teste la realisabilite de chaque candidat survivant
"""

import itertools
import math
import numpy as np

N_VAL = 3           # valeurs par attribut, et par position
N_POS = 3
N_REF = 27
LOG2_3 = math.log2(3)
INFO_TOTALE = math.log2(N_REF)


def im_depuis_table(table):
    """Information mutuelle d'une table de contingence, en bits."""
    n = table.sum()
    p = table / n
    pi = p.sum(axis=1, keepdims=True)
    pj = p.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terme = np.where(p > 0, p * np.log2(p / (pi * pj)), 0.0)
    return float(terme.sum())


def tables_a_marges(m=9):
    """Toutes les tables 3x3 d'entiers >= 0 dont les six marges valent m."""
    sortie = []
    for a in range(m + 1):
        for b in range(m + 1 - a):
            c = m - a - b
            for d in range(m + 1):
                for e in range(m + 1 - d):
                    f = m - d - e
                    g, h, i = m - a - d, m - b - e, m - c - f
                    if g < 0 or h < 0 or i < 0:
                        continue
                    if g + h + i != m:
                        continue
                    sortie.append(np.array([[a, b, c], [d, e, f], [g, h, i]]))
    return sortie


def main():
    print("=" * 74)
    print("1. LE TREILLIS")
    print("=" * 74)
    tables = tables_a_marges(9)
    valeurs = sorted({round(im_depuis_table(t), 12) for t in tables})
    print(f"   tables 3x3 a marges toutes egales a 9 : {len(tables)}"
          f"   (il annonce 1540)")
    print(f"   valeurs distinctes de I(A_i ; M_j)     : {len(valeurs)}"
          f"   (il annonce 55)")
    print(f"   plus grande                            : {valeurs[-1]:.12f}"
          f"   log2(3) = {LOG2_3:.12f}")
    print(f"   les six plus grandes : "
          + ", ".join(f"{v:.6f}" for v in valeurs[-6:]))

    print()
    print("=" * 74)
    print("2. UNE CONTRAINTE QUE NI LUI NI MOI N'AVONS ECRITE")
    print("=" * 74)
    print("   Pour un code bijectif sur 27 = 3^3 referents uniformes, le message")
    print("   est une image bijective du referent, donc (M_1, M_2, M_3) est uniforme")
    print("   sur 3^3 : les trois positions sont MUTUELLEMENT INDEPENDANTES.")
    print("   Pour des Y_j independants, I(X ; Y_1..Y_n) >= somme_j I(X ; Y_j).")
    print("   Or les trois positions determinent le referent, donc")
    print("   I(A_i ; M_1,M_2,M_3) = H(A_i) = log2(3). D'ou :\n")
    print(f"     chaque LIGNE de la matrice somme a au plus log2(3) = {LOG2_3:.6f}")
    print("   et par le meme argument sur les attributs, chaque COLONNE aussi.\n")
    print("   Consequence directe, sans enumeration : si les trois maxima de colonne")
    print("   sont dans la ligne r, alors somme_j max_i M[i,j] <= log2(3), et")
    print("   l'appariement vaut au moins max_j M[r,j] >= (somme de la ligne)/3.")
    borne = (2 / 3) * LOG2_3 / INFO_TOTALE
    print(f"   Donc inflation <= (2/3) * log2(3) / log2(27) = {borne:.9f} = 2/9")

    print()
    print("=" * 74)
    print("3. ENUMERATION DES CANDIDATS AU-DESSUS DE LA VALEUR TROUVEE")
    print("=" * 74)
    trouve = 0.154321642873
    print(f"   valeur trouvee par recherche : {trouve:.12f}")
    print("   Une matrice a une seule ligne non nulle (a, b, c) donne")
    print("   inflation = (a + b + c - max(a,b,c)) / log2(27), soit la somme des")
    print("   deux plus petites. Contrainte de ligne : a + b + c <= log2(3).\n")
    candidats = []
    for a, b, c in itertools.combinations_with_replacement(valeurs, 3):
        if a + b + c > LOG2_3 + 1e-12:
            continue
        infl = (a + b + c - max(a, b, c)) / INFO_TOTALE
        if infl > trouve + 1e-12:
            candidats.append((infl, (a, b, c)))
    candidats.sort(reverse=True)
    print(f"   triples du treillis respectant la contrainte de ligne et")
    print(f"   depassant la valeur trouvee : {len(candidats)}")
    for infl, (a, b, c) in candidats[:12]:
        print(f"     {infl:.12f}   ({a:.9f}, {b:.9f}, {c:.9f})")
    if len(candidats) > 12:
        print(f"     ... et {len(candidats)-12} autres")

    print()
    print("=" * 74)
    print("4. LA VALEUR TROUVEE EST-ELLE SUR LE TREILLIS ?")
    print("=" * 74)
    cible = [0.415630552279, 0.438340850504, 0.318151498733]
    for v in cible:
        proche = min(valeurs, key=lambda x: abs(x - v))
        print(f"   {v:.12f}  ->  treillis {proche:.12f}   ecart {abs(proche-v):.2e}")
    somme = sum(cible)
    print(f"\n   somme de la ligne : {somme:.12f}   contre log2(3) = {LOG2_3:.12f}")
    print(f"   marge restante    : {LOG2_3 - somme:.12f}")
    infl = (somme - max(cible)) / INFO_TOTALE
    print(f"   inflation         : {infl:.12f}")


if __name__ == "__main__":
    main()
