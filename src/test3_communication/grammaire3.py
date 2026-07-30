"""
RDTRL — Test 3, etape 1 : le monde, les codes, et la statistique de concentration.

Aucun entrainement ici. Uniquement du calcul exact sur un espace enumerable, pour
que les instruments existent AVANT de lancer quoi que ce soit — c'est ce qui a
rendu le test 2 concluant.

Contenu :
  - les 27 referents (3 attributs x 3 valeurs) et les 27 messages (3 tokens,
    vocabulaire de 3) ;
  - le comptage exact des bijections parfaites et des codes compositionnels,
    verifie par enumeration et pas seulement par formule ;
  - la matrice d'information mutuelle attribut x position, qui est ce qui
    distingue reellement un code compositionnel d'un code holistique ;
  - la statistique de concentration qui resume cette matrice en un scalaire ;
  - la LOI NULLE de cette statistique sur des permutations tirees uniformement.
    C'est la reference de tout le reste : sans elle, une concentration mesuree ne
    veut rien dire.
"""

import argparse
import json
import math
import os
from itertools import permutations, product

import numpy as np

RACINE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DOSSIER_SORTIE = os.path.join(RACINE, "results_test3")

N_ATTRIBUTS = 3
N_VALEURS = 3
N_POSITIONS = 3
N_TOKENS = 3

REFERENTS = list(product(range(N_VALEURS), repeat=N_ATTRIBUTS))
MESSAGES = list(product(range(N_TOKENS), repeat=N_POSITIONS))
INDEX_MESSAGE = {m: i for i, m in enumerate(MESSAGES)}
N = len(REFERENTS)

# Tableaux pour le calcul vectorise :
#   ATTRIBUT[i][r] = valeur de l'attribut i du referent r
#   TOKEN[j][m]    = token en position j du message m
ATTRIBUT = np.array([[ref[i] for ref in REFERENTS] for i in range(N_ATTRIBUTS)])
TOKEN = np.array([[msg[j] for msg in MESSAGES] for j in range(N_POSITIONS)])

# Information portee par un code parfait : log2(27) bits, quel que soit le code.
# Ce qui distingue les codes n'est donc pas la QUANTITE d'information mais sa
# REPARTITION dans la matrice attribut x position.
INFORMATION_TOTALE = math.log2(N)


def codes_compositionnels():
    """Les codes ou le token j encode l'attribut sigma(j), a renommage pres.

    Un code compositionnel est entierement determine par :
      - sigma, l'assignation des positions aux attributs (3! choix) ;
      - pour chaque position, une bijection valeur -> token (3! choix chacune).
    Soit 3! x (3!)^3 = 6 x 216 = 1296 codes.
    """
    codes = []
    for sigma in permutations(range(N_ATTRIBUTS)):
        for f in product(permutations(range(N_TOKENS)), repeat=N_POSITIONS):
            code = np.empty(N, dtype=int)
            for r, referent in enumerate(REFERENTS):
                message = tuple(f[j][referent[sigma[j]]] for j in range(N_POSITIONS))
                code[r] = INDEX_MESSAGE[message]
            codes.append(code)
    return codes


def matrice_information(code):
    """I(A_i ; M_j) en bits, pour un code deterministe (bijection ou non).

    Les referents sont tires uniformement, donc P(r) = 1/27 et la loi jointe de
    (attribut i, token en position j) s'obtient par simple comptage.
    """
    matrice = np.zeros((N_ATTRIBUTS, N_POSITIONS))
    tokens_emis = TOKEN[:, code]           # tokens_emis[j][r] = token j du message de r
    for i in range(N_ATTRIBUTS):
        for j in range(N_POSITIONS):
            indices = ATTRIBUT[i] * N_TOKENS + tokens_emis[j]
            jointe = np.bincount(indices, minlength=N_VALEURS * N_TOKENS)
            jointe = jointe.reshape(N_VALEURS, N_TOKENS) / N
            marge_a = jointe.sum(axis=1, keepdims=True)
            marge_t = jointe.sum(axis=0, keepdims=True)
            produit = marge_a @ marge_t
            non_nuls = jointe > 0
            matrice[i, j] = float(
                (jointe[non_nuls] * np.log2(jointe[non_nuls] / produit[non_nuls])).sum())
    return matrice


def concentration(code=None, matrice=None):
    """Resume la matrice en un scalaire dans [0, 1].

    Pour chaque position, on prend l'attribut dont elle porte le plus
    d'information, et on somme. Un code compositionnel place log2(3) bits sur une
    seule case par colonne, donc atteint le maximum. Un code holistique etale
    l'information et tombe plus bas.

    Le denominateur est 3 x log2(3) = log2(27), l'information totale d'un code
    parfait : la statistique vaut donc exactement 1 pour un code compositionnel.
    """
    if matrice is None:
        matrice = matrice_information(code)
    return float(matrice.max(axis=0).sum() / INFORMATION_TOTALE)


def loi_nulle(n_echantillons, graine=0):
    """Distribution de la concentration sur des bijections tirees uniformement.

    C'est la reference sans laquelle une concentration mesuree ne signifie rien.
    Tester directement « ce code est-il compositionnel ? » n'aurait aucune
    puissance : sous l'hypothese nulle on attend 1296/27! succes, soit zero
    observation en pratique. La concentration, elle, est continue et discriminante.
    """
    generateur = np.random.default_rng(graine)
    valeurs = np.empty(n_echantillons)
    for k in range(n_echantillons):
        valeurs[k] = concentration(generateur.permutation(N))
    return valeurs


def resume_combinatoire():
    total = math.factorial(N)
    compositionnels = math.factorial(N_ATTRIBUTS) * math.factorial(N_TOKENS) ** N_POSITIONS
    return {
        "referents": N,
        "messages": len(MESSAGES),
        "bijections_parfaites": total,
        "codes_compositionnels": compositionnels,
        "fraction_compositionnelle": compositionnels / total,
        "hasard_reconstruction": 1 / N,
        "information_totale_bits": INFORMATION_TOTALE,
    }


if __name__ == "__main__":
    parseur = argparse.ArgumentParser(description="RDTRL — test 3, monde et instruments")
    parseur.add_argument("--echantillons", type=int, default=20000,
                         help="taille de l'echantillon pour la loi nulle")
    parseur.add_argument("--graine", type=int, default=0)
    args = parseur.parse_args()
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)

    print("=" * 78)
    print("TEST 3 — LE MONDE ET LES INSTRUMENTS (aucun entrainement)")
    print("=" * 78)
    infos = resume_combinatoire()
    print(f"Referents  : {infos['referents']} "
          f"({N_ATTRIBUTS} attributs x {N_VALEURS} valeurs)")
    print(f"Messages   : {infos['messages']} "
          f"({N_POSITIONS} positions x {N_TOKENS} tokens)")
    print(f"Hasard (reconstruction) : {100 * infos['hasard_reconstruction']:.2f} %")
    print(f"Information d'un code parfait : log2(27) = "
          f"{infos['information_totale_bits']:.4f} bits")
    print()

    print("-" * 78)
    print("COMPTAGE EXACT")
    print("-" * 78)
    print(f"  bijections parfaites        : 27! = {infos['bijections_parfaites']:,}"
          .replace(",", " "))
    print(f"  codes compositionnels       : 3! x (3!)^3 = "
          f"{infos['codes_compositionnels']}")
    print(f"  fraction compositionnelle   : {infos['fraction_compositionnelle']:.3e}")
    print()
    print("  Tous ces codes rapportent exactement 1. Ils sont a egalite parfaite,")
    print("  donc la recompense ne peut pas les departager.")
    print()

    # Verification par enumeration, pas seulement par formule.
    compositionnels = codes_compositionnels()
    distincts = {tuple(c) for c in compositionnels}
    bijections = all(len(set(c.tolist())) == N for c in compositionnels)
    print(f"  verification par enumeration : {len(compositionnels)} codes generes, "
          f"{len(distincts)} distincts, tous bijectifs : {bijections}")
    assert len(distincts) == infos["codes_compositionnels"], "comptage analytique faux"
    assert bijections, "un code compositionnel n'est pas une bijection"

    concentrations_comp = np.array([concentration(c) for c in compositionnels])
    print(f"  concentration des codes compositionnels : "
          f"min {concentrations_comp.min():.6f}, max {concentrations_comp.max():.6f}")
    assert np.allclose(concentrations_comp, 1.0), "un code compositionnel devrait valoir 1"
    print()

    print("-" * 78)
    print("EXEMPLE — matrice d'information mutuelle attribut x position (bits)")
    print("-" * 78)
    print("  Code compositionnel canonique (token j encode l'attribut j) :")
    canonique = np.array([INDEX_MESSAGE[ref] for ref in REFERENTS])
    print(np.round(matrice_information(canonique), 4))
    print(f"  concentration = {concentration(canonique):.4f}")
    print()
    generateur = np.random.default_rng(args.graine)
    aleatoire = generateur.permutation(N)
    print("  Bijection tiree au hasard :")
    print(np.round(matrice_information(aleatoire), 4))
    print(f"  concentration = {concentration(aleatoire):.4f}")
    print()

    print("-" * 78)
    print(f"LOI NULLE — concentration sur {args.echantillons} bijections uniformes")
    print("-" * 78)
    valeurs = loi_nulle(args.echantillons, args.graine)
    quantiles = {q: float(np.quantile(valeurs, q))
                 for q in (0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999)}
    print(f"  moyenne     : {valeurs.mean():.4f}")
    print(f"  ecart-type  : {valeurs.std():.4f}")
    print(f"  minimum     : {valeurs.min():.4f}")
    print(f"  maximum     : {valeurs.max():.4f}")
    print(f"  quantiles   : " + "  ".join(f"{int(q*1000)/10}% {v:.4f}"
                                          for q, v in quantiles.items()))
    print()
    print(f"  Un code compositionnel vaut 1.0000, soit "
          f"{(1.0 - valeurs.mean()) / valeurs.std():.1f} ecarts-types au-dessus")
    print(f"  de la moyenne nulle. Aucun des {args.echantillons} tirages ne l'atteint.")
    print()
    print("  C'est ce seuil, et non un chiffre arbitraire, qui dira si les codes")
    print("  emergents sont selectionnes par autre chose que la recompense.")

    rapport = {
        "combinatoire": infos,
        "loi_nulle": {
            "n_echantillons": args.echantillons,
            "moyenne": float(valeurs.mean()),
            "ecart_type": float(valeurs.std()),
            "min": float(valeurs.min()),
            "max": float(valeurs.max()),
            "quantiles": quantiles,
        },
        "concentration_compositionnelle": 1.0,
    }
    chemin = os.path.join(DOSSIER_SORTIE, "monde_et_loi_nulle.json")
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False)
    np.save(os.path.join(DOSSIER_SORTIE, "loi_nulle_concentration.npy"), valeurs)
    print(f"\nEcrit dans {DOSSIER_SORTIE}")
