"""
RDTRL — Test 3 : le maximum de la loi nulle est-il un estimateur de quelque chose ?

Le seuil « ~0,35 » de §6.1 est derive du MAXIMUM observe sur 20 000 tirages
(0,3305). Dipankar Sarkar a montre qu'a 10 000 000 de tirages ce maximum monte a
0,3979 pendant que le reste du tableau ne bouge pas. Son diagnostic : le maximum
est la seule ligne encore en mouvement.

Ce script tranche autrement, et plus durement. Au lieu de comparer deux tailles
d'echantillon, il tire des BLOCS INDEPENDANTS de meme taille et regarde de combien
la statistique bouge d'un bloc a l'autre. Un estimateur d'une quantite de la loi
se stabilise quand la taille du bloc grandit ; un maximum d'echantillon, non.

Le point theorique qui explique ce qu'on va voir, et qui ne demande aucun tirage :

    les 1 296 codes compositionnels SONT des bijections. Ils appartiennent donc a
    la loi nulle, avec probabilite 1 296 / 27! ~ 1,19e-25, et ils valent 1.

Le supremum de la loi nulle vaut donc exactement 1 — c'est-a-dire la valeur meme
qu'on veut declarer impossible sous la nulle. Le maximum d'echantillon n'estime
pas un seuil : il estime 1, infiniment lentement. Aucune taille d'echantillon
n'y change rien, et un seuil bati dessus est un seuil bati sur la taille de
l'echantillon qu'on a eu la patience de tirer.
"""

import argparse
import json
import math
import os
import time

import numpy as np

from grammaire3 import DOSSIER_SORTIE, N
from loi_nulle_longue import matrices_information, statistiques

QUANTILES = (0.99, 0.999, 0.9999)


def bloc(n, generateur, taille_lot):
    """Un bloc de n tirages. Renvoie max, quantiles de queue, moyenne, ecart-type."""
    pas = 0
    somme = somme_carres = 0.0
    # le pool doit couvrir le quantile le plus BAS demande, pas une fraction devinee
    garde = max(int(n * (1 - min(QUANTILES))) + 10, 1000)
    pool = np.empty(0)
    while pas < n:
        lot = min(taille_lot, n - pas)
        # permutations par construction, donc verifier serait une perte seche
        codes = np.argsort(generateur.random((lot, N)), axis=1)
        cm, _, _ = statistiques(matrices_information(codes, verifier_bijectivite=False))
        somme += cm.sum()
        somme_carres += (cm * cm).sum()
        pool = np.concatenate([pool, cm])
        if pool.size > garde:
            pool = pool[np.argpartition(pool, -garde)[-garde:]]
        pas += lot
    pool.sort()
    resultat = {"max": float(pool[-1]),
                "moyenne": somme / n,
                "ecart_type": math.sqrt(somme_carres / n - (somme / n) ** 2)}
    for q in QUANTILES:
        rang = int(round((1 - q) * n))
        resultat[f"q{q}"] = float(pool[-rang]) if 1 <= rang <= pool.size else None
    return resultat


if __name__ == "__main__":
    parseur = argparse.ArgumentParser(description="RDTRL — variabilite du maximum")
    parseur.add_argument("--blocs", type=int, default=10)
    parseur.add_argument("--par-bloc", type=int, default=10_000_000)
    parseur.add_argument("--graine", type=int, default=100)
    parseur.add_argument("--lot", type=int, default=100_000)
    args = parseur.parse_args()
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)

    print("=" * 78)
    print(f"TEST 3 — {args.blocs} BLOCS INDEPENDANTS DE {args.par_bloc:,} TIRAGES"
          .replace(",", " "))
    print("=" * 78)
    print("\nSi une ligne du tableau est un estimateur, ses valeurs par bloc se")
    print("resserrent. Si c'est un maximum d'echantillon, elles ne se resserrent pas.\n")

    generateur = np.random.default_rng(args.graine)
    resultats = []
    debut = time.time()
    entete = f"  {'bloc':>4}  {'moyenne':>8}  {'sd':>8}  " + "  ".join(
        f"{'q' + str(100 * q) + '%':>9}" for q in QUANTILES) + f"  {'MAX':>9}"
    print(entete)
    print("  " + "-" * (len(entete) - 2))
    for b in range(args.blocs):
        r = bloc(args.par_bloc, generateur, args.lot)
        resultats.append(r)
        print(f"  {b:>4}  {r['moyenne']:8.4f}  {r['ecart_type']:8.4f}  "
              + "  ".join(f"{r['q' + str(q)]:9.4f}" for q in QUANTILES)
              + f"  {r['max']:9.4f}", flush=True)

    print(f"\n  termine en {time.time() - debut:.0f} s")
    print("\n" + "-" * 78)
    print("DISPERSION ENTRE BLOCS — c'est la seule chose que ce script mesure")
    print("-" * 78)
    print(f"  {'ligne':>12}  {'moyenne':>9}  {'etendue':>9}  "
          f"{'sd entre blocs':>15}  {'sd / moyenne':>13}")
    lignes = [("moyenne", "moyenne"), ("ecart_type", "ecart_type")]
    lignes += [(f"q{100 * q}%", f"q{q}") for q in QUANTILES] + [("MAXIMUM", "max")]
    dispersion = {}
    for nom, cle in lignes:
        v = np.array([r[cle] for r in resultats], dtype=float)
        etendue = v.max() - v.min()
        dispersion[nom] = {"moyenne": float(v.mean()), "etendue": float(etendue),
                           "sd_entre_blocs": float(v.std(ddof=1))}
        print(f"  {nom:>12}  {v.mean():9.4f}  {etendue:9.4f}  "
              f"{v.std(ddof=1):15.4f}  {v.std(ddof=1) / v.mean():13.1%}")

    sd_nulle = float(np.mean([r["ecart_type"] for r in resultats]))
    etendue_max = dispersion["MAXIMUM"]["etendue"]
    print(f"\n  L'etendue du maximum entre blocs vaut {etendue_max:.4f}, soit "
          f"{etendue_max / sd_nulle:.2f} ecart-type de la loi nulle elle-meme.")
    print(f"  Un seuil pose sur cette ligne herite de cette dispersion.")

    rapport = {"blocs": args.blocs, "par_bloc": args.par_bloc, "graine": args.graine,
               "resultats": resultats, "dispersion": dispersion,
               "sd_nulle": sd_nulle, "secondes": time.time() - debut}
    nom = f"variabilite_max_{args.blocs}blocs_de{args.par_bloc}_g{args.graine}.json"
    with open(os.path.join(DOSSIER_SORTIE, nom), "w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False)
    print(f"\nEcrit dans {DOSSIER_SORTIE} sous {nom}")
