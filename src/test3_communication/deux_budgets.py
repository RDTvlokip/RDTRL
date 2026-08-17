"""Le rapport a un maximum d'echantillon au numerateur ET au denominateur.

Le relecteur montre que 0,1443 / max(nulle) decroit avec n parce que le
denominateur est une statistique d'ordre. Le numerateur en est une aussi :
`recherche_pire_cas(..., n_restarts=24)`. Le 0,1443 est le maximum sur
vingt-quatre montees, et il croit avec le budget de redemarrages.

Ce script mesure les deux cotes :

  A. le numerateur en fonction du nombre de departs
  B. le denominateur en fonction du nombre de tirages
  C. le test n contre n/10 applique a tout le fichier de la loi nulle
"""

import sys
import numpy as np

sys.path.insert(0, "src/test3_communication")
from loi_nulle_longue import N, matrices_information, statistiques
from appariement_vs_distance import recherche_pire_cas

BUDGETS_DEPARTS = [6, 12, 24, 48, 96, 192, 384]
BUDGETS_TIRAGES = [100_000, 1_000_000, 10_000_000]


def objectif_inflation(lot):
    cm, ca, _ = statistiques(matrices_information(lot, verifier_bijectivite=False))
    return cm - ca


def numerateur():
    print("=" * 74)
    print("A. LE NUMERATEUR EST UN MAXIMUM SUR LES DEPARTS")
    print("=" * 74)
    print("   recherche_pire_cas(..., n_restarts=24) : 0,1443 est le meilleur")
    print("   de vingt-quatre montees. Rien dans le nombre ne le dit.\n")
    print(f"   {'departs':>10}{'meilleure inflation':>22}{'rapport a 0,122365':>22}")
    resultats = {}
    for budget in BUDGETS_DEPARTS:
        generateur = np.random.default_rng(0)
        valeur, _ = recherche_pire_cas(objectif_inflation, generateur,
                                       n_restarts=budget, n_pas=60)
        resultats[budget] = valeur
        print(f"   {budget:>10}{valeur:>22.6f}{valeur / 0.122365:>22.3f}")
    return resultats


def denominateur():
    print()
    print("=" * 74)
    print("B. LE DENOMINATEUR EST UN MAXIMUM SUR LES TIRAGES")
    print("=" * 74)
    generateur = np.random.default_rng(0)
    max_courant = 0.0
    somme = 0.0
    n_positifs = 0
    vus = 0
    jalons = {}
    while vus < max(BUDGETS_TIRAGES):
        lot = min(100_000, max(BUDGETS_TIRAGES) - vus)
        codes = np.argsort(generateur.random((lot, N)), axis=1)
        cm, ca, _ = statistiques(
            matrices_information(codes, verifier_bijectivite=False))
        infl = cm - ca
        max_courant = max(max_courant, float(infl.max()))
        somme += float(infl.sum())
        n_positifs += int((infl > 0).sum())
        vus += lot
        if vus in BUDGETS_TIRAGES:
            jalons[vus] = (max_courant, somme / vus, n_positifs / vus)
    print(f"   {'n':>14}{'max':>12}{'E[inflation]':>16}{'P(infl > 0)':>14}")
    for n, (mx, moy, taux) in jalons.items():
        print(f"   {n:>14,}{mx:>12.6f}{moy:>16.7f}{taux:>14.6f}".replace(",", " "))
    print("\n   Le max bouge, les deux autres non. C'est son test n contre n/10,")
    print("   et il separe bien les deux especes.")
    return jalons


def le_rapport(res_num, res_den):
    print()
    print("=" * 74)
    print("C. LE RAPPORT, AVEC SES DEUX BUDGETS")
    print("=" * 74)
    print("   Chaque case est 'meilleur de k departs' divise par 'max de n tirages'.")
    print("   Aucune n'est une propriete de quoi que ce soit.\n")
    ns = sorted(res_den)
    print(f"   {'departs':>10}" + "".join(f"{f'n={n:,}':>18}".replace(",", " ")
                                          for n in ns))
    for budget in sorted(res_num):
        ligne = f"   {budget:>10}"
        for n in ns:
            ligne += f"{res_num[budget] / res_den[n][0]:>18.3f}"
        print(ligne)
    print("\n   Il fait varier une colonne. Le tableau a deux axes, et le nombre")
    print("   publie etait une seule case sans coordonnees.")


def main():
    res_num = numerateur()
    res_den = denominateur()
    le_rapport(res_num, res_den)


if __name__ == "__main__":
    main()
