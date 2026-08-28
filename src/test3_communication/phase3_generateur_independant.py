"""RDTRL — sa question fermante du tour vingt-quatre, testee plutot que lue.

Il a trouve que phase 3 (« ATTEIGNABLE », le tableau §6.5, 5 % de bijections en
tabulaire) tire ses 30 graines depuis la MEME position de flux que phases 1 et 2
consomment avant elle : 3 permutations (codes temoins) + 24 tirages (phase 1,
`classe(generateur)`/`Recepteur(generateur)` par combinaison) + 48 tirages (phase 2,
quatre `cloner()` par combinaison, chacun tirant une graine derivee puis jetee).
Position exacte confirmee ci-dessous par relecture directe du fichier.

Question posee en retour : le tableau §6.5 bouge-t-il si phase 3 recoit son PROPRE
generateur, independant de ce que 1 et 2 ont consomme au-dessus ?

Pas besoin de rejouer torch pour les phases 1 et 2 : seule la POSITION du flux
numpy compte pour les graines que phase 3 va tirer, et chaque construction
(`classe(generateur)`, `Recepteur(generateur)`, `cloner(..., generateur)`) ne tire
qu'un seul entier. Avancer le flux de 3 permutations + 72 tirages reproduit la
position exacte sans executer un seul pas de montee des phases 1 et 2.
"""

import time

import numpy as np

from grammaire3 import N
from representable_atteignable_stable import (EmetteurFactorise,
                                              EmetteurStructure,
                                              EmetteurTabulaire, Recepteur,
                                              decrire, lire_code, monter)

BETA = 0.02
PAS = 3000
GRAINES = 10
CLASSES = (EmetteurTabulaire, EmetteurFactorise, EmetteurStructure)


def avancer_comme_phases_1_et_2(generateur):
    """Consomme EXACTEMENT ce que le script reel consomme avant phase 3."""
    for _ in range(3):                       # codes_temoins : 3 aleatoire_i
        generateur.permutation(N)
    for _ in range(len(CLASSES) * 4):        # phase 1 : 3 classes x 4 codes x 2 tirages
        generateur.integers(1 << 30)
    for _ in range(len(CLASSES) * 4 * 4):    # phase 2 : 3 classes x 4 codes x 4 cloner()
        generateur.integers(1 << 30)


def phase3(generateur, etiquette):
    print(f"\n  {etiquette}")
    print(f"  {'parametrisation':>16}  {'E[R] moyen':>11}  {'bijections':>11}  "
          f"{'collisions':>11}  {'concentration appariee':>23}")
    resultats = []
    for classe in CLASSES:
        lot = []
        for _ in range(GRAINES):
            emetteur, recepteur = classe(generateur), Recepteur(generateur)
            recompense = monter(emetteur, recepteur, BETA, PAS)
            info = decrire(lire_code(emetteur))
            info["reward"] = recompense
            info["parametrisation"] = classe.nom
            lot.append(info)
        resultats.extend(lot)
        recompenses = np.array([r["reward"] for r in lot])
        bijections = sum(r["bijectif"] for r in lot)
        collisions = np.array([r["collisions"] for r in lot])
        conc = np.array([r["concentration_appariee"] for r in lot])
        print(f"  {classe.nom:>16}  {recompenses.mean():11.4f}  "
              f"{bijections:>4} / {len(lot):<4}  "
              f"{collisions.mean():11.2f}  {conc.mean():13.4f} +/- {conc.std():.4f}")
    return resultats


if __name__ == "__main__":
    t0 = time.time()

    g_expedie = np.random.default_rng(0)
    avancer_comme_phases_1_et_2(g_expedie)
    r_expedie = phase3(g_expedie, "ARM EXPEDIEE : meme position de flux que le script publie")

    g_independant = np.random.default_rng(0)
    r_independant = phase3(g_independant,
                           "ARM INDEPENDANTE : generateur frais, aucun tirage de phase 1/2")

    print(f"\n  temps total : {time.time() - t0:.1f} s")

    print("\n=== LE TABLEAU BOUGE-T-IL ? ===")
    for classe in CLASSES:
        e = [r for r in r_expedie if r["parametrisation"] == classe.nom]
        i = [r for r in r_independant if r["parametrisation"] == classe.nom]
        be, bi = sum(r["bijectif"] for r in e), sum(r["bijectif"] for r in i)
        ce = np.array([r["concentration_appariee"] for r in e])
        ci = np.array([r["concentration_appariee"] for r in i])
        re = np.array([r["reward"] for r in e])
        ri = np.array([r["reward"] for r in i])
        print(f"\n  {classe.nom}")
        print(f"    bijections    expediee {be}/10   independante {bi}/10")
        print(f"    E[R] moyen    expediee {re.mean():.4f}   independante {ri.mean():.4f}"
              f"   ecart {re.mean() - ri.mean():+.4f}")
        print(f"    conc. appariee expediee {ce.mean():.4f} +/- {ce.std():.4f}"
              f"   independante {ci.mean():.4f} +/- {ci.std():.4f}")
