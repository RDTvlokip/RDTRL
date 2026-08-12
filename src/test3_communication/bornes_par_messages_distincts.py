"""
RDTRL — Test 3 : les bornes de degat, refaites dans le regime ou l'experience est.

Les trois bornes de « Bounding the damage » ont ete obtenues par montee locale sur
des PERMUTATIONS. Or §6.5 et §6.7 mesurent que la montee de gradient exacte ne
rejoint pas de bijection depuis l'uniforme : elle se pose sur des codes a 1 a 4
collisions. Les bornes ont donc ete mesurees dans le seul regime ou l'experience
ne se trouve pas.

J'avais ecrit que les trois etaient conditionnelles a la bijectivite, puis je les
avais laissees telles quelles. Dipankar Sarkar (12/08/2026) a fait remarquer que
la bonne suite n'etait pas de les annoter mais de les RELANCER, et a produit les
chiffres. Ce script est ma verification independante.

DIFFERENCE DE GRIMPEUR, ET ELLE COMPTE. Le mien, dans `appariement_vs_distance`,
bougeait par TRANSPOSITIONS, ce qui preserve la bijectivite par construction et
interdit donc toute collision. Ici les mouvements sont des REAFFECTATIONS d'un
referent vers un message quelconque, ce qui autorise les collisions, avec un
plancher R sur le nombre de messages distincts. Seule la contrainte bouge d'une
colonne a l'autre.
"""

import argparse
import json
import os

import numpy as np

from grammaire3 import DOSSIER_SORTIE, N
from loi_nulle_longue import matrices_information_generale, statistiques

PAIRES = [(r, m) for r in range(N) for m in range(N)]


def distincts(code):
    return len(np.unique(code))


def voisins(code):
    """Toutes les reaffectations d'un referent vers un message. 27 x 27 = 729."""
    lot = np.repeat(np.asarray(code)[None, :], len(PAIRES), axis=0)
    for i, (r, m) in enumerate(PAIRES):
        lot[i, r] = m
    return lot


def monter(objectif, plancher, generateur, pas=120):
    """Montee la plus raide sous contrainte de messages distincts >= plancher."""
    # depart : une bijection dont on effondre juste assez de referents
    code = generateur.permutation(N)
    for r in range(N - plancher):
        code[generateur.integers(N)] = code[generateur.integers(N)]
    if distincts(code) < plancher:
        return -np.inf, code
    valeur = objectif(np.asarray(code)[None, :])[0]
    for _ in range(pas):
        lot = voisins(code)
        licites = np.array([distincts(c) >= plancher for c in lot])
        valeurs = np.where(licites, objectif(lot), -np.inf)
        k = int(valeurs.argmax())
        if valeurs[k] <= valeur + 1e-12:
            break
        code, valeur = lot[k].copy(), float(valeurs[k])
    return float(valeur), code


def chercher(objectif, plancher, generateur, restarts, pas):
    meilleur, meilleur_code = -np.inf, None
    for _ in range(restarts):
        v, c = monter(objectif, plancher, generateur, pas)
        if v > meilleur:
            meilleur, meilleur_code = v, c
    return meilleur, meilleur_code


if __name__ == "__main__":
    parseur = argparse.ArgumentParser(description="RDTRL — bornes par messages distincts")
    parseur.add_argument("--planchers", type=int, nargs="*", default=[27, 26, 25, 24, 23])
    parseur.add_argument("--restarts", type=int, default=20)
    parseur.add_argument("--pas", type=int, default=120)
    parseur.add_argument("--graine", type=int, default=0)
    args = parseur.parse_args()
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    generateur = np.random.default_rng(args.graine)

    def ecart(lot):
        cm, ca, _ = statistiques(matrices_information_generale(lot))
        return cm - ca

    def max_en_double(lot):
        cm, _, dc = statistiques(matrices_information_generale(lot))
        return np.where(dc, cm, -1.0)

    print("=" * 78)
    print("LES BORNES DE DEGAT, PAR NOMBRE DE MESSAGES DISTINCTS")
    print("=" * 78)
    print(f"\n  {args.restarts} departs, {args.pas} pas, reaffectation d'un referent.")
    print("  Seule la contrainte change d'une colonne a l'autre.\n")

    print(f"  {'plancher R':>12}  {'ecart max':>10}  {'a concentration':>16}  "
          f"{'appariee':>9}  {'R atteint':>10}")
    table = []
    for plancher in args.planchers:
        v, code = chercher(ecart, plancher, generateur, args.restarts, args.pas)
        cm, ca, _ = statistiques(matrices_information_generale(np.asarray(code)[None, :]))
        table.append({"plancher": plancher, "ecart_max": v,
                      "concentration_max": float(cm[0]),
                      "concentration_appariee": float(ca[0]),
                      "distincts": int(distincts(code)),
                      "code": [int(x) for x in code]})
        print(f"  {plancher:>12}  {v:10.4f}  {float(cm[0]):16.4f}  "
              f"{float(ca[0]):9.4f}  {distincts(code):>10}")

    print(f"\n  {'plancher R':>12}  {'max en double compte':>21}  {'appariee':>9}")
    table2 = []
    for plancher in args.planchers:
        v, code = chercher(max_en_double, plancher, generateur, args.restarts, args.pas)
        cm, ca, _ = statistiques(matrices_information_generale(np.asarray(code)[None, :]))
        table2.append({"plancher": plancher, "max_en_double": v,
                       "concentration_appariee": float(ca[0]),
                       "distincts": int(distincts(code)),
                       "code": [int(x) for x in code]})
        print(f"  {plancher:>12}  {v:21.4f}  {float(ca[0]):9.4f}")

    print("\n  Le code qui maximise l'ecart au plancher R = 26 :")
    gagnant = next(t for t in table if t["plancher"] == 26)
    m = matrices_information_generale(np.array(gagnant["code"])[None, :])[0]
    print("  I(A_i ; M_j) en bits :")
    for ligne in np.round(m, 3):
        print("   ", ligne)
    print(f"  argmax par position : {list(m.argmax(axis=0))}")
    print(f"  messages distincts  : {gagnant['distincts']} sur {N}")
    print(f"  max {gagnant['concentration_max']:.4f}  "
          f"appariee {gagnant['concentration_appariee']:.4f}")

    rapport = {"restarts": args.restarts, "pas": args.pas, "graine": args.graine,
               "ecart_par_plancher": table, "max_en_double_par_plancher": table2}
    nom = f"bornes_par_messages_distincts_g{args.graine}.json"
    with open(os.path.join(DOSSIER_SORTIE, nom), "w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False)
    print(f"\nEcrit dans {DOSSIER_SORTIE} sous {nom}")
