"""L'inflation dans la queue de la loi nulle, sans le plafond du reservoir.

`loi_nulle_longue.py` construit `inflation_maximale` a partir du reservoir, qui
cesse de se remplir a 2 000 000 tirages, et l'imprime sous « toute la loi ». Le
maximum publie est donc celui du premier cinquieme du tirage.

Ce script refait le meme tirage, meme graine, meme taille de lot, donc le meme
flux, en suivant :

  1. le vrai maximum courant de l'inflation, par jalons
  2. des comptes de depassement exacts sur toute la loi — le traitement que le
     script d'origine reservait a la concentration
  3. si le pool exact, retenu sur conc_max, porte le vrai maximum d'inflation
  4. la forme de la queue, pour transformer « 0 sur 10^7 » en une estimation
"""

import sys
import time
import numpy as np

sys.path.insert(0, "src/test3_communication")
from loi_nulle_longue import N, matrices_information, statistiques

JALONS = [100_000, 500_000, 2_000_000, 5_000_000, 10_000_000]
SEUILS_INFLATION = [0.03, 0.04, 0.05, 0.05927, 0.06, 0.07, 0.08, 0.09,
                    0.10, 0.10807, 0.11, 0.12, 0.13, 0.1443]
BORNE_RECHERCHE = 0.1443


def tirer(n_total, graine=0, taille_lot=100_000, garder=200_000):
    generateur = np.random.default_rng(graine)
    depassements = {s: 0 for s in SEUILS_INFLATION}
    max_courant = 0.0
    jalons = {}
    somme_infl = 0.0
    n_collision = 0
    somme_infl_pos = 0.0
    pool_max = np.empty(0)
    pool_infl = np.empty(0)
    reservoir_max_infl = 0.0
    n_reservoir = 0
    pas = 0
    debut = time.time()
    while pas < n_total:
        lot = min(taille_lot, n_total - pas)
        codes = np.argsort(generateur.random((lot, N)), axis=1)
        cm, ca, _ = statistiques(
            matrices_information(codes, verifier_bijectivite=False))
        infl = cm - ca

        max_courant = max(max_courant, float(infl.max()))
        for s in SEUILS_INFLATION:
            depassements[s] += int((infl >= s).sum())
        somme_infl += float(infl.sum())
        positifs = infl[infl > 0]
        n_collision += int(positifs.size)
        somme_infl_pos += float(positifs.sum())

        # le pool exact du script d'origine, retenu sur conc_max
        pool_max = np.concatenate([pool_max, cm])
        pool_infl = np.concatenate([pool_infl, infl])
        if pool_max.size > garder:
            indices = np.argpartition(pool_max, -garder)[-garder:]
            pool_max, pool_infl = pool_max[indices], pool_infl[indices]

        # le reservoir du script d'origine, plafonne
        if n_reservoir < 2_000_000:
            reservoir_max_infl = max(reservoir_max_infl, float(infl.max()))
            n_reservoir += lot

        pas += lot
        if pas in JALONS:
            jalons[pas] = (max_courant, reservoir_max_infl, n_reservoir,
                           float(pool_infl.max()))
        if pas % 2_000_000 == 0:
            ecoule = time.time() - debut
            print(f"    {pas:>12,} tirages  {ecoule:6.1f} s".replace(",", " "),
                  flush=True)
    return {
        "n": n_total,
        "max_vrai": max_courant,
        "max_reservoir": reservoir_max_infl,
        "jalons": jalons,
        "depassements": depassements,
        "moyenne_inflation": somme_infl / n_total,
        "taux_collision": n_collision / n_total,
        "moyenne_si_collision": somme_infl_pos / n_collision,
        "pool_max_infl": float(pool_infl.max()),
        "secondes": time.time() - debut,
    }


def main():
    n = 10_000_000
    print(f"tirage de {n:,} bijections, graine 0, lots de 100 000".replace(",", " "))
    res = tirer(n)
    print(f"  termine en {res['secondes']:.0f} s\n")

    print("=" * 74)
    print("1. LE MAXIMUM PUBLIE ETAIT CELUI DU PREMIER CINQUIEME")
    print("=" * 74)
    print(f"   {'n':>12}{'vrai max':>12}{'max reservoir':>16}{'n reservoir':>14}"
          f"{'max du pool':>14}")
    for jalon in JALONS:
        if jalon in res["jalons"]:
            vrai, reserv, nres, pool = res["jalons"][jalon]
            print(f"   {jalon:>12,}{vrai:>12.6f}{reserv:>16.6f}{nres:>14,}"
                  f"{pool:>14.6f}".replace(",", " "))
    print(f"\n   publie dans inflation_maximale : 0.10807050074977963")
    print(f"   vrai maximum sur 10^7           : {res['max_vrai']:.6f}")
    print(f"   tirages au-dessus du publie     : "
          f"{res['depassements'][0.10807]}")

    print()
    print("=" * 74)
    print("2. LE POOL PORTE-T-IL LE VRAI MAXIMUM ? (par chance ou par garantie)")
    print("=" * 74)
    print(f"   max d'inflation dans le pool exact : {res['pool_max_infl']:.6f}")
    print(f"   vrai max sur toute la loi          : {res['max_vrai']:.6f}")
    if abs(res["pool_max_infl"] - res["max_vrai"]) < 1e-12:
        print("   Ils coincident. Mais le pool est retenu sur conc_max, et")
        print("   l'inflation n'est pas monotone en conc_max : c'est une")
        print("   coincidence de ce tirage, pas une garantie du code.")
    else:
        print("   Ils different : le pool NE porte PAS le vrai maximum.")

    print()
    print("=" * 74)
    print("3. COMPTES DE DEPASSEMENT EXACTS — LE TRAITEMENT RESERVE A L'AUTRE COLONNE")
    print("=" * 74)
    print(f"   {'seuil':>10}{'compte sur 10^7':>18}{'p':>14}")
    for s in SEUILS_INFLATION:
        c = res["depassements"][s]
        p = c / n
        marque = ""
        if abs(s - 0.05927) < 1e-6:
            marque = "  <- max observe sur 210 runs"
        if abs(s - 0.10807) < 1e-6:
            marque = "  <- mon inflation_maximale publiee"
        if abs(s - BORNE_RECHERCHE) < 1e-6:
            marque = "  <- pire cas de la recherche adverse"
        p_txt = f"{p:.3e}" if c > 0 else f"< {3/n:.1e}"
        print(f"   {s:>10.5f}{c:>18,}{p_txt:>14}{marque}".replace(",", " "))

    print()
    print("=" * 74)
    print("4. LA QUEUE A-T-ELLE UNE FORME ? (pour estimer plutot que borner)")
    print("=" * 74)
    seuils = [s for s in SEUILS_INFLATION if res["depassements"][s] >= 30]
    comptes = np.array([res["depassements"][s] for s in seuils], dtype=float)
    x = np.array(seuils)
    y = np.log(comptes / n)
    pente, ordonnee = np.polyfit(x, y, 1)
    residus = y - (pente * x + ordonnee)
    r2 = 1 - (residus ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"   ajustement log P(inflation >= s) = {ordonnee:+.3f} {pente:+.3f} s")
    print(f"   R2 = {r2:.5f} sur {len(seuils)} seuils de {min(seuils):.3f} "
          f"a {max(seuils):.5f}")
    print(f"   -> queue exponentielle, longueur caracteristique "
          f"{-1/pente:.5f}\n")
    for cible, nom in [(res["max_vrai"], "max observe sur 10^7"),
                       (BORNE_RECHERCHE, "pire cas de la recherche")]:
        p_est = np.exp(ordonnee + pente * cible)
        print(f"   P(inflation >= {cible:.5f})  estimee {p_est:.2e}   "
              f"soit 1 tirage sur {1/p_est:,.0f}   [{nom}]".replace(",", " "))
    p_borne = np.exp(ordonnee + pente * BORNE_RECHERCHE)
    print(f"\n   La recherche adverse vaut donc environ {1/p_borne:,.0f} tirages "
          f"au hasard.".replace(",", " "))
    print(f"   Borne empirique seule (regle de trois, 0 sur 10^7) : "
          f"p < {3/n:.1e}, soit 1 sur {n/3:,.0f}.".replace(",", " "))

    print()
    print("=" * 74)
    print("5. CE QUE LES TROIS AUTRES CHAMPS DEVIENNENT")
    print("=" * 74)
    print(f"   {'champ':<34}{'publie':>14}{'toute la loi':>16}")
    print(f"   {'taux_global':<34}{0.7464519:>14.7f}{res['taux_collision']:>16.7f}")
    print(f"   {'inflation_moyenne_globale':<34}{0.010049794802284647:>14.7f}"
          f"{res['moyenne_inflation']:>16.7f}")
    print(f"   {'E[inflation | collision]':<34}{0.010049794802284647/0.7464519:>14.7f}"
          f"{res['moyenne_si_collision']:>16.7f}")
    print(f"   {'inflation_maximale':<34}{0.10807050074977963:>14.7f}"
          f"{res['max_vrai']:>16.7f}")


if __name__ == "__main__":
    main()
