"""Le plafond d'une politique sans couplage, et ce que les runs a 12 modes ont appris.

Question posee par dipankarsarkar : les deux coins degeneres contiennent 24
phrases valides chacun, mais pas de la meme facon.

PARTIE A - le plus grand ensemble PRODUIT contenu dans chaque coin.
Une politique sans couplage det -> nom a un support produit. A validite 1 son
support doit donc tenir dans le plus grand produit entierement valide du coin.
C'est un plafond calculable, sans aucun entrainement.

PARTIE B - les runs a 12 modes sont-ils a ce plafond ?
Deux structures differentes donnent 12 modes : un produit verrouille sur un genre
(2 det x 2 noms x 3 verbes) ou une politique couplee qui n'utiliserait qu'un nom
par determinant. La saturation de H(nom | determinant) les separe.
"""

import argparse
import json
import os
from itertools import combinations, product as iproduct

import numpy as np
import torch

from grammaire import Grammaire
from rl_grammaire import (PolitiqueGRU, fixer_graine, entrainer, analyse_exacte,
                          DOSSIER_SORTIE)
from gradient_exact import probabilites_exactes


def phrases_valides_du_coin(g, noms):
    total = 0
    for nom in noms:
        tn = g.traits(nom)
        prod = 1
        for cat in g.structure:
            if cat == "nom":
                continue
            traits = g.traits_a_accorder.get(cat, ())
            prod *= sum(1 for t in g.tokens_par_categorie[cat]
                        if all(g._compatible(g.traits(t)[x], tn[x]) for x in traits))
        total += prod
    return total


def plus_grand_produit(g, noms_du_coin):
    """Plus grand produit entierement valide dont les noms sont dans le coin.

    Un produit est valide ssi chaque categorie ne garde que des tokens
    compatibles avec TOUS les noms retenus : il suffit donc d'enumerer les
    sous-ensembles de noms.
    """
    autres = [c for c in g.structure if c != "nom"]
    meilleur = (0, None)
    for k in range(1, len(noms_du_coin) + 1):
        for bloc in combinations(noms_du_coin, k):
            taille = len(bloc)
            detail = {"nom": list(bloc)}
            for cat in autres:
                traits = g.traits_a_accorder.get(cat, ())
                ok = [t for t in g.tokens_par_categorie[cat]
                      if all(all(g._compatible(g.traits(t)[x], g.traits(n)[x]) for x in traits)
                             for n in bloc)]
                taille *= len(ok)
                detail[cat] = ok
            if taille > meilleur[0]:
                meilleur = (taille, detail)
    return meilleur


def partie_a():
    resultats = {}
    print("=" * 78)
    print("PARTIE A - plafond d'une politique sans couplage, par coin")
    print("=" * 78)
    for longue in (False, True):
        g = Grammaire(longue=longue)
        nom_g = "longue" if longue else "courte"
        resultats[nom_g] = {}
        print(f"\nGrammaire {nom_g}")
        for nb in ("sg", "pl"):
            noms = [n for n in g.tokens_par_categorie["nom"] if g.traits(n)["nombre"] == nb]
            valides = phrases_valides_du_coin(g, noms)
            taille, detail = plus_grand_produit(g, noms)
            resultats[nom_g][nb] = {"valides": valides, "plus_grand_produit": taille,
                                    "plafond_bits": round(float(np.log2(taille)), 4),
                                    "detail": {k: sorted(v) for k, v in detail.items()}}
            print(f"  coin {nb} : {valides:3d} valides | plus grand produit {taille:3d} "
                  f"| plafond {np.log2(taille):.3f} bits | "
                  + " x ".join(f"{c}:{len(v)}" for c, v in detail.items()))
        ecart = (resultats[nom_g]["pl"]["plafond_bits"]
                 - resultats[nom_g]["sg"]["plafond_bits"])
        resultats[nom_g]["ecart_bits"] = round(ecart, 4)
        print(f"  ecart de plafond pluriel - singulier : {ecart:+.4f} bits")
    return resultats


def detail_saturation(politique, g, etiquette):
    ex = analyse_exacte(politique, g)
    print(f"  {etiquette} : {ex['masse_valide_pct']:.2f} % valide | "
          f"{ex['modes_effectifs']} modes | sg {ex['repartition_familles']['sg']:.1f} % "
          f"/ pl {ex['repartition_familles']['pl']:.1f} %")
    print(f"    {'det':>5} {'masse':>9} {'accord%':>8} {'H bits':>8} {'H acc.':>8} "
          f"{'H max':>7} {'satur.%':>8} {'noms ok':>8}")
    for det in sorted(ex["entropie_nom_sachant_det"]):
        e = ex["entropie_nom_sachant_det"][det]
        if e is None:
            print(f"    {det:>5} {'~0':>9} {'n/a':>8}")
            continue
        sat = e["saturation_pct"] if e["saturation_pct"] is not None else 0.0
        print(f"    {det:>5} {e['masse_du_determinant']:>9.5f} "
              f"{e['masse_accordee_pct']:>8.2f} {e['H_bits']:>8.3f} "
              f"{e['H_accorde_bits']:>8.3f} {e['H_max_bits']:>7.3f} "
              f"{sat:>8.1f} {e['noms_compatibles']:>8}")
    return ex


def gradient_exact(g, sequences_t, recompenses_t, beta, graine, etapes=4000, lr=5e-3):
    fixer_graine(graine)
    politique = PolitiqueGRU(g.taille)
    optimiseur = torch.optim.Adam(politique.parameters(), lr=lr)
    for _ in range(etapes):
        log_p = probabilites_exactes(politique, sequences_t)
        p = log_p.exp()
        p = p / p.sum()
        esperance = (p * recompenses_t).sum()
        # entropie en NATS, comme dans gradient_exact.py : beta*H est donc
        # homogene a beta*ln(2) par bit gagne
        entropie = -(p * torch.log(p.clamp_min(1e-30))).sum()
        perte = -(esperance + beta * entropie)
        optimiseur.zero_grad()
        perte.backward()
        optimiseur.step()
    return politique


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20000)
    p.add_argument("--graines", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--sans-entrainement", action="store_true")
    args = p.parse_args()

    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    rapport = {"partie_a": partie_a(), "gradient_exact": {}, "echantillonne": {}}
    if args.sans_entrainement:
        return

    g = Grammaire(longue=False)
    sequences = list(iproduct(range(g.taille), repeat=g.longueur))
    sequences_t = torch.tensor(sequences, dtype=torch.long)
    recompenses_t = torch.tensor([g.recompense_graduee(s) for s in sequences],
                                 dtype=torch.float32)

    print()
    print("=" * 78)
    print("PARTIE B1 - gradient exact : les runs a 12 modes ET leur temoin a 24")
    print("=" * 78)
    # Les quatre runs a 12,0 modes du tableau publie sont (0.01, 0), (0.01, 1),
    # (0.01, 2) et (0.02, 1). Les deux autres, (0.02, 0) et (0.02, 2), font
    # 24,0 modes DANS LE MEME COIN SINGULIER : ils depassent donc le plafond
    # sans couplage, ce qui n'est possible qu'en ayant acquis la conditionnelle.
    # Sans eux le chiffre a 12 n'est pas interpretable, avec eux c'est un
    # contraste a un seul facteur.
    for beta in (0.01, 0.02):
        for graine in args.graines:
            pol = gradient_exact(g, sequences_t, recompenses_t, beta, graine)
            ex = detail_saturation(pol, g, f"gradient exact beta={beta} graine {graine}")
            rapport["gradient_exact"][f"b{beta}_g{graine}"] = ex

    print()
    print("=" * 78)
    print("PARTIE B2 - REINFORCE echantillonne, beta = 0.02, graines 1 et 2")
    print("=" * 78)
    for graine in (1, 2):
        fixer_graine(graine)
        pol = PolitiqueGRU(g.taille)
        entrainer(pol, g, max_episodes=args.episodes, type_recompense="graduee",
                  coef_entropie=0.02, verbeux=False, etiquette=f"g{graine}")
        ex = detail_saturation(pol, g, f"echantillonne beta=0.02 graine {graine}")
        rapport["echantillonne"][str(graine)] = ex

    chemin = os.path.join(DOSSIER_SORTIE, "produit_et_saturation.json")
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nRapport ecrit dans {chemin}")


if __name__ == "__main__":
    main()
