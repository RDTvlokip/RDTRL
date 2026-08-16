"""L'ecart max - appariee a une masse en zero, et personne ne l'avait ouverte.

Trente pour cent des runs ont un ecart exactement nul, parce que l'argmax non
contraint est deja une bijection : aucune position ne reclame le meme attribut
qu'une autre. La grandeur n'est donc pas continue, c'est un melange.

Trois consequences, dans l'ordre de ce qu'elles coutent :

  1. le rapport publie compare un melange a une conditionnelle
  2. la grandeur se decompose en deux quantites a consommateurs distincts
  3. tous les t de cet echange supposent une normalite que ni la variable ni son
     logarithme ne montrent
"""

import json
import numpy as np
from scipy import stats as st

DECOUVERTE = "results_test3/effet_par_beta_30graines_g0.json"
REPLICATION = "results_test3/effet_par_beta_12graines_g7.json"
BORNE = 0.1443


def charger(chemin):
    with open(chemin, encoding="utf-8") as flux:
        return json.load(flux)["runs"]


def main():
    runs_d = charger(DECOUVERTE)
    runs_r = charger(REPLICATION)
    ecart = np.array([x["ecart"] for x in runs_d])
    beta = np.array([x["beta"] for x in runs_d])
    R = np.array([x["R"] for x in runs_d])
    ecart_r = np.array([x["ecart"] for x in runs_r])
    tous = np.concatenate([ecart, ecart_r])
    positifs = tous[tous > 0]

    print("=" * 74)
    print("1. LA MASSE EN ZERO")
    print("=" * 74)
    print(f"   ecarts exactement nuls : {int((tous == 0).sum())}/{len(tous)} = "
          f"{(tous == 0).mean():.1%}")
    print(f"   plus petit ecart non nul : {positifs.min():.2e}")
    print("   un ecart nul veut dire que l'argmax non contraint est deja une")
    print("   bijection : aucune position ne reclame l'attribut d'une autre.\n")
    reference = 6 / 27
    print(f"   pour reference, sous des argmax independants et uniformes sur")
    print(f"   3 positions x 3 attributs, P(permutation) = 3!/3^3 = {reference:.4f}")
    print(f"   observe {(tous == 0).mean():.4f} sur {len(tous)} runs, binomial "
          f"p = {st.binomtest(int((tous == 0).sum()), len(tous), reference).pvalue:.4f}")
    print("   Cette reference n'est PAS un nul justifie : les trois argmax portent")
    print("   sur des informations mutuelles correlees, pas sur des tirages")
    print("   independants. C'est un point de comparaison, pas un resultat.")

    print()
    print("=" * 74)
    print("2. LE RAPPORT PUBLIE COMPARE UN MELANGE A UNE CONDITIONNELLE")
    print("=" * 74)
    print("   La borne 0,1443 vient du grimpeur par transpositions, qui cherche le")
    print("   PIRE code : il a necessairement une collision d'argmax. C'est donc un")
    print("   pire cas SACHANT collision. Mon 0,0104 est non conditionnel.\n")
    print(f"   {'quantite':<46}{'valeur':>10}{'rapport':>10}")
    lignes = [
        ("E[ecart] sur 210 runs (publie)", tous.mean()),
        ("E[ecart | ecart > 0] — comparaison appariee", positifs.mean()),
        ("mediane des ecarts > 0", float(np.median(positifs))),
        ("q95 des ecarts > 0", float(np.quantile(positifs, 0.95))),
        ("maximum observe sur 210 runs", tous.max()),
    ]
    for nom, valeur in lignes:
        print(f"   {nom:<46}{valeur:>10.5f}{BORNE / valeur:>10.1f}")

    print()
    print("=" * 74)
    print("3. DEUX QUANTITES A CONSOMMATEURS DISTINCTS")
    print("=" * 74)
    proportion = (tous > 0).mean()
    intervalle = st.binomtest(int((tous > 0).sum()), len(tous)).proportion_ci()
    print(f"   P(collision d'argmax)        {proportion:.3f}   "
          f"IC95 [{intervalle[0]:.3f}, {intervalle[1]:.3f}]")
    print(f"   E[inflation | collision]     {positifs.mean():.5f}   "
          f"SE {positifs.std(ddof=1) / np.sqrt(len(positifs)):.5f}")
    print(f"   produit                      {proportion * positifs.mean():.5f}   "
          f"(= la moyenne publiee {tous.mean():.5f})")
    print("\n   P(collision)     : a quelle frequence la statistique publiee est fausse")
    print("   E[. | collision] : de combien quand elle l'est")

    print()
    print("=" * 74)
    print("4. LA FORME, ET CE QU'ELLE INVALIDE")
    print("=" * 74)
    print(f"   n = {len(positifs)}   moyenne {positifs.mean():.5f}   "
          f"sd {positifs.std(ddof=1):.5f}   CV {positifs.std(ddof=1)/positifs.mean():.2f}")
    print(f"   asymetrie {st.skew(positifs):+.2f}   "
          f"aplatissement {st.kurtosis(positifs):+.2f}")
    print(f"   Shapiro sur l'ecart positif : p = {st.shapiro(positifs)[1]:.2e}")
    print(f"   Shapiro sur son logarithme  : p = {st.shapiro(np.log(positifs))[1]:.2e}")
    print("   Ni la variable ni son log ne passent. Ce n'est donc pas une")
    print("   log-normale non plus : c'est une masse en zero plus une partie")
    print("   positive asymetrique a droite, sans forme fermee evidente.\n")
    print("   Effet sur les deux contrastes de l'echange :\n")
    print(f"   {'contraste':<20}{'Student brut':>22}{'log(ecart>0)':>22}{'Mann-Whitney':>16}")
    for nom, etiquettes, a, b in [("R 25 contre 24", R, 25, 24),
                                  ("beta .005 / .03", beta, 0.005, 0.03)]:
        va, vb = ecart[etiquettes == a], ecart[etiquettes == b]
        pa, pb = va[va > 0], vb[vb > 0]
        brut = st.ttest_ind(va, vb, equal_var=True)
        log = st.ttest_ind(np.log(pa), np.log(pb), equal_var=False)
        rangs = st.mannwhitneyu(va, vb)
        print(f"   {nom:<20}t={brut[0]:+.3f} p={brut[1]:.4f}   "
              f"t={log[0]:+.3f} p={log[1]:.4f}   p={rangs[1]:.4f}")

    print()
    print("   decomposition de chaque contraste en part portee par les zeros")
    print("   et part conditionnelle :\n")
    for nom, etiquettes, a, b in [("R 25 contre 24", R, 25, 24),
                                  ("beta .005 / .03", beta, 0.005, 0.03)]:
        va, vb = ecart[etiquettes == a], ecart[etiquettes == b]
        pa, pb = va[va > 0], vb[vb > 0]
        z_a, z_b = (va == 0).mean(), (vb == 0).mean()
        total = va.mean() - vb.mean()
        part_zeros = ((1 - z_a) - (1 - z_b)) * (pa.mean() + pb.mean()) / 2
        part_cond = ((1 - z_a) + (1 - z_b)) / 2 * (pa.mean() - pb.mean())
        fisher = st.fisher_exact([[int((va == 0).sum()), int((va > 0).sum())],
                                  [int((vb == 0).sum()), int((vb > 0).sum())]])[1]
        print(f"   {nom}")
        print(f"     P(ecart = 0)  {z_a:.3f} contre {z_b:.3f}   Fisher p = {fisher:.4f}")
        print(f"     total {total:+.5f} = zeros {part_zeros:+.5f} + "
              f"conditionnel {part_cond:+.5f}   "
              f"({abs(part_zeros)/(abs(part_zeros)+abs(part_cond)):.0%} par les zeros)")


if __name__ == "__main__":
    main()
